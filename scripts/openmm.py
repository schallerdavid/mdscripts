"""
This script uses OpenMM to run MD simulations with reasonable defaults. A small molecule can be provided separately via
an sdf file. Missing parameters for small molecules are generated with the openmmforcefields package. The results are
processed to center the protein in the water box. The simulation can be restricted to the equilibration by setting
'production_steps' to 0.

Thanks to @glass-w and @hannahbrucemacdonald for inspiration.

Easy installation of all dependencies via conda:
conda create -n openmm -c conda-forge mdanalysis mdtraj numpy openforcefield openmm openmmforcefields
"""
import pathlib

import MDAnalysis as mda
from MDAnalysis import transformations
import mdtraj as md
import numpy as np
from openforcefield.topology import Molecule, Topology
from openmmforcefields.generators import SystemGenerator
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
from simtk.openmm.app import PDBFile, CheckpointReporter
from simtk.openmm.app.statedatareporter import StateDataReporter


# input & output
pdb_path = "path/to/file.pdb"
ligand_path = ""
output_directory = pathlib.Path().cwd() / "output"

# forcefield
protein_forcefield = "amber14/protein.ff14SB.xml"
small_molecule_forcefield = "openff-1.2.0"
solvent_forcefield = "amber14/tip3p.xml"
nonbonded_method = app.PME

# simulation
equilibration_steps = 2500000  # 5 ns
production_steps = 50000000  # 100 ns
pressure = 1.0 * unit.bar
temperature = 300 * unit.kelvin
constraints = app.HBonds
remove_cm_motions = True
collision_rate = 1.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
solvent_padding = 10.0 * unit.angstrom
ionic_strength = 150 * unit.millimolar

# reporter
equilibration_reporter_frequency = round(equilibration_steps / 1000)
equilibration_trajectory_frequency = round(equilibration_steps / 100)
production_reporter_frequency = round(production_steps / 10000)
production_checkpoint_frequency = round(production_steps / 100)
production_trajectory_frequency = round(production_steps / 2000)  # 2000 frames

print("Making output directory ...")
output_directory = pathlib.Path(output_directory)
(output_directory / "equilibration").mkdir(parents=True, exist_ok=True)

print("Reading PDB file ...")
pdb = app.PDBFile(pdb_path)
topology, positions = pdb.topology, pdb.positions

if len(ligand_path) > 0:
    print("Combining topologies ...")  # credit to @hannahbrucemacdonald
    molecule = Molecule.from_file(ligand_path)
    off_ligand_topology = Topology.from_molecules(molecule)
    ligand_topology = off_ligand_topology.to_openmm()
    ligand_positions = molecule.conformers[0]
    md_protein_topology = md.Topology.from_openmm(
        topology
    )  # using mdtraj for protein top
    md_ligand_topology = md.Topology.from_openmm(
        ligand_topology
    )  # using mdtraj for ligand top
    md_complex_topology = md_protein_topology.join(
        md_ligand_topology
    )  # add them together
    complex_topology = md_complex_topology.to_openmm()  # now back to openmm
    total_atoms = len(positions) + len(ligand_positions)
    complex_positions = unit.Quantity(np.zeros([total_atoms, 3]), unit=unit.nanometers)
    complex_positions[0 : len(positions)] = positions
    for index, atom in enumerate(
        ligand_positions, len(positions)
    ):  # openmm works in nm
        coordinates = atom / atom.unit
        complex_positions[index] = (coordinates / 10.0) * unit.nanometers
    topology = complex_topology
    positions = complex_positions
else:
    molecule = None

print("Setting barostat ...")
barostat = mm.MonteCarloBarostat(pressure, temperature)

print("Initializing system generator ...")
system_generator = SystemGenerator(
    forcefields=[protein_forcefield, solvent_forcefield],
    barostat=barostat,
    periodic_forcefield_kwargs={"nonbondedMethod": nonbonded_method},
    small_molecule_forcefield=small_molecule_forcefield,
    molecules=molecule,
)

print("Solvating system ...")
modeller = app.Modeller(topology, positions)
modeller.addSolvent(
    forcefield=system_generator.forcefield,
    model="tip3p",
    ionicStrength=ionic_strength,
    padding=solvent_padding,
)

print("Saving solvated system ...")
with open(output_directory / "solvated_system.pdb", "w") as outfile:
    PDBFile.writeFile(
        modeller.topology, modeller.positions, file=outfile, keepIds=True,
    )

print("Creating OpenMM system ...")
system = system_generator.create_system(modeller.topology)

print("Initializing integrator ...")
integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep)

print("Initializing equilibration simulation ...")
simulation = app.Simulation(modeller.topology, system, integrator)

print("Setting positions ...")
simulation.context.setPositions(modeller.positions)

print("Minimizing energy ...")
initial_energy = (
    simulation.context.getState(getEnergy=True).getPotentialEnergy()
    / unit.kilocalories_per_mole
)
print(f"  initial : {initial_energy:.3f} kcal/mol")
simulation.minimizeEnergy()
final_energy = (
    simulation.context.getState(getEnergy=True).getPotentialEnergy()
    / unit.kilocalories_per_mole
)
print(f"  final : {final_energy:.3f} kcal/mol")

print("Generating random starting velocities ...")
simulation.context.setVelocitiesToTemperature(temperature)

print("Initializing equilibration reporters ...")
simulation.reporters.append(
    StateDataReporter(
        file=str(output_directory / "equilibration/progress.log"),
        reportInterval=equilibration_reporter_frequency,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=equilibration_steps,
        separator="\t",
    )
)

simulation.reporters.append(
    md.reporters.XTCReporter(
        file=str(output_directory / "equilibration/trajectory.xtc"),
        reportInterval=equilibration_trajectory_frequency,
    )
)

print("Running equilibration simulation ...")
simulation.step(equilibration_steps)

print("Saving equilibrated system and state ...")
with open(output_directory / "equilibration/out_state.pdb", "w") as outfile:
    PDBFile.writeFile(
        simulation.topology,
        simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True
        ).getPositions(),
        file=outfile,
        keepIds=True,
    )

state = simulation.context.getState(
    getPositions=True, getVelocities=True, getEnergy=True, getForces=True
)
with open(output_directory / "equilibration/out_state.xml", "w") as outfile:
    xml = mm.XmlSerializer.serialize(state)
    outfile.write(xml)

system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
with open(output_directory / "equilibration/out_system.xml", "w") as outfile:
    xml = mm.XmlSerializer.serialize(system)
    outfile.write(xml)

if production_steps > 0:

    print("Reading equilibrated system and state ...")
    with open(output_directory / "equilibration/out_system.xml", "r") as infile:
        system = mm.XmlSerializer.deserialize(infile.read())

    with open(output_directory / "equilibration/out_state.xml", "r") as infile:
        state = mm.XmlSerializer.deserialize(infile.read())

    print("Initializing integrator ...")
    integrator = mm.LangevinIntegrator(temperature, collision_rate, timestep,)

    print("Initializing production simulation ...")
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())
    simulation.context.setTime(0)

    print("Initializing production reporters ...")
    simulation.reporters.append(
        StateDataReporter(
            file=str(output_directory / "progress.log"),
            reportInterval=production_reporter_frequency,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            speed=True,
            progress=True,
            remainingTime=True,
            totalSteps=production_steps,
            separator="\t",
        )
    )

    simulation.reporters.append(
        CheckpointReporter(
            file=str(output_directory / "checkpoint.chk"),
            reportInterval=production_checkpoint_frequency,
        )
    )

    simulation.reporters.append(
        md.reporters.XTCReporter(
            file=str(output_directory / "trajectory.xtc"),
            reportInterval=production_trajectory_frequency,
        )
    )

    print("Running production simulation ...")
    simulation.step(production_steps)

if production_steps / production_trajectory_frequency >= 1:

    print("Transforming trajectory ...")
    u = mda.Universe(
        str(output_directory / "equilibration/out_state.pdb"),
        str(output_directory / "trajectory.xtc"),
    )
    protein = u.select_atoms("protein or resname ACE or resname NME")
    reference_u = u.copy()
    reference = reference_u.select_atoms("protein or resname ACE or resname NME")
    ag = u.atoms
    workflow = (
        transformations.unwrap(ag),
        transformations.center_in_box(protein, center="mass"),
        transformations.wrap(ag, compound="fragments"),
        transformations.fit_rot_trans(protein, reference),
    )
    u.trajectory.add_transformations(*workflow)

    print("Saving transformed topology and trajectory ...")
    u.atoms.write(str(output_directory / "topology_wrapped.pdb"))
    with mda.Writer(
        str(output_directory / "trajectory_wrapped.xtc"), u.atoms.n_atoms
    ) as W:
        for ts in u.trajectory:
            W.write(u.atoms)

print("Finished")
