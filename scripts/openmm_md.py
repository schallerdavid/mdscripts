"""
This script uses OpenMM to run MD simulations with reasonable defaults. A small molecule can be
provided separately via an sdf file. Missing parameters for small molecules are generated with
the openmmforcefields package. The results are processed to center the protein in the water box.

Thanks to @glass-w, @hannahbrucemacdonald, @sukritsingh and @openforcefield for inspiration.

Easy installation of all dependencies via conda/mamba:
conda create -n openmm -c conda-forge mdanalysis mdtraj openmmforcefields

Optionally install the OpenEye toolkits (free academic licence) to accelerate small molecule parametrization:
conda activate openmm
conda install -c openeye openeye-toolkits
"""
import math
import pathlib
import subprocess

import MDAnalysis as mda
from MDAnalysis import transformations
import mdtraj as md
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmm.app import PDBFile, CheckpointReporter
from openmm.app.statedatareporter import StateDataReporter


# input & output
pdb_path = "path/to/file.pdb"
ligand_path = ""
output_directory = pathlib.Path().cwd() / "output"

# simulation
simulation_steps = 50000000  # 100 ns
pressure = 1.0 * unit.bar
temperature = 310.15 * unit.kelvin  # 37 °C
constraints = app.HBonds
nonbonded_cutoff = 1 * unit.nanometers
nonbonded_method = app.PME
collision_rate = 1.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
solvent_padding = 10.0 * unit.angstrom
ionic_strength = 150 * unit.millimolar
max_number_cpus = 4  # important for e.g. slurm

# reporter
reporter_frequency = math.ceil(simulation_steps / 10000)
checkpoint_frequency = math.ceil(simulation_steps / 100)
trajectory_frequency = math.ceil(simulation_steps / 2000)  # 2000 frames

print("Setting maximum number of CPU threads ...")
platform = mm.Platform.getPlatformByName("CPU")
platform.setPropertyDefaultValue(property="Threads", value=str(max_number_cpus))

print("Making output directory ...")
output_directory = pathlib.Path(output_directory)
output_directory.mkdir(exist_ok=True, parents=True)

print("Writing packages of conda environment ...")
with open(output_directory / "conda_environment.txt", "w") as wf:
    subprocess.run(["conda", "list"], stdout=wf)

print("Reading PDB file ...")
pdb = app.PDBFile(pdb_path)
topology, positions = pdb.topology, pdb.positions
modeller = app.Modeller(topology, positions)

print("Setting up forcefield ...")
forcefield = app.ForceField(
    "amber/protein.ff14SB.xml", "amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml"
)

if len(ligand_path) > 0:
    print("Registering small molecule and combining topologies ...")
    ligand = Molecule.from_file(ligand_path)
    ligand_positions = ligand.conformers[0]
    ligand_topology = ligand.to_topology()
    gaff_template_generator = GAFFTemplateGenerator(molecules=[ligand], forcefield="gaff-2.11")
    forcefield.registerTemplateGenerator(gaff_template_generator.generator)
    modeller.add(ligand_topology.to_openmm(), ligand_positions)

print("Solvating system ...")
modeller.addSolvent(
    forcefield=forcefield,
    model="tip3p",
    ionicStrength=ionic_strength,
    padding=solvent_padding,
)

print("Saving solvated system ...")
with open(output_directory / "solvated_system.pdb", "w") as outfile:
    PDBFile.writeFile(modeller.topology, modeller.positions, file=outfile, keepIds=True)

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=nonbonded_method,
    nonbondedCutoff=nonbonded_cutoff,
    constraints=constraints,
)

print("Minimizing energy ...")
context = mm.Context(
    system,
    mm.LangevinIntegrator(temperature, collision_rate, timestep),
    platform
)
context.setPositions(modeller.getPositions())
initial_energy = context.getState(getEnergy=True).getPotentialEnergy()/unit.kilocalories_per_mole
print(f"  initial : {initial_energy:.3f} kcal/mol")
mm.LocalEnergyMinimizer.minimize(context)
final_energy = context.getState(getEnergy=True).getPotentialEnergy()/unit.kilocalories_per_mole
print(f"  final : {final_energy:.3f} kcal/mol")
with open(output_directory / "minimized_system.pdb", "w") as outfile:
    app.PDBFile.writeFile(
        modeller.topology,
        context.getState(getPositions=True).getPositions(),
        file=outfile,
        keepIds=True
    )

print("Initializing simulation ...")
simulation = app.Simulation(
    modeller.topology,
    system,
    mm.LangevinIntegrator(temperature, collision_rate, timestep)
)

print("Setting positions ...")
simulation.context.setPositions(context.getState(getPositions=True).getPositions())

print("Generating random starting velocities ...")
simulation.context.setVelocitiesToTemperature(temperature)

print("Initializing reporters ...")
simulation.reporters.append(
    StateDataReporter(
        file=str(output_directory / "progress.log"),
        reportInterval=reporter_frequency,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=simulation_steps,
        separator="\t",
    )
)
simulation.reporters.append(
    CheckpointReporter(
        file=str(output_directory / "checkpoint.chk"),
        reportInterval=checkpoint_frequency,
    )
)
simulation.reporters.append(
    md.reporters.XTCReporter(
        file=str(output_directory / "trajectory.xtc"),
        reportInterval=trajectory_frequency,
    )
)

print("Running production simulation ...")
simulation.step(simulation_steps)

print("Writing state and system as XML ...")
state = simulation.context.getState(
    getPositions=True, getVelocities=True, getEnergy=True, getForces=True
)
with open(output_directory / "out_state.xml", "w") as outfile:
    xml = mm.XmlSerializer.serialize(state)
    outfile.write(xml)
system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
with open(output_directory / "out_system.xml", "w") as outfile:
    xml = mm.XmlSerializer.serialize(system)
    outfile.write(xml)

print("Transforming trajectory ...")
u = mda.Universe(
    str(output_directory / "solvated_system.pdb"),
    str(output_directory / "trajectory.xtc"),
    all_coordinates=True,
    in_memory=True,
)
backbone = u.select_atoms("backbone")
not_protein = u.select_atoms("not protein")
workflow = (
    transformations.unwrap(backbone, max_threads=max_number_cpus),
    transformations.center_in_box(backbone),
    transformations.wrap(not_protein, compound="fragments", max_threads=max_number_cpus),
    transformations.fit_rot_trans(backbone, backbone),
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
