import argparse
import random
import mdtraj
import openmm
import numpy as np
from openmm import app, unit
from sys import stdout
from openmm import *                                                                                                
from openmmplumed import PlumedForce

cases  =  ['full', 'start', 'restart']
waters =  ['spce', 'tip3p', 'tip4pew', 'tip5p']
parser = argparse.ArgumentParser()
parser.add_argument('--water', dest='water', help='the water model', choices=waters, default=None)
parser.add_argument('--ff', dest='ff', help='the pepdide force field', default='amber03')
parser.add_argument('--seed', dest='seed', help='the RNG seed', default=None)
parser.add_argument('--platform', dest='platform', help='the computation platform', default='Reference')
parser.add_argument('--case', dest='case', help='the simulation case', choices=cases, default='full')
args = parser.parse_args()

from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

smirnoff = SMIRNOFFTemplateGenerator(forcefield='openff-2.0.0.offxml')

sdf_files = ['bcd_gly.sdf', 'aspirin.sdf']
for i in range(len(sdf_files)):
    offmol = Molecule.from_file(sdf_files[i])
    smirnoff.add_molecules(offmol)

from openmm.app import ForceField

forcefield = app.ForceField('amber14-all.xml', 'tip3p.xml', 'amber/GLYCAM_06j-1.xml')
forcefield.registerTemplateGenerator(smirnoff.generator)

pdb = app.PDBFile('Umb_1.pdb')

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,nonbondedCutoff=1*unit.nanometer,removeCMMotion=False,rigidWater=False,constraints=None)

plumed_script = """

RESTART
UNITS ENERGY=kcal/mol
c1: COM ATOMS=1-147  #CENTER OF MASS OF the host
c2: COM ATOMS=148-168  #CENTER OF MASS OF the guest
d: DISTANCE ATOMS=c1,c2 
OW: GROUP ATOMS=169-8106:3
LIGsolv: COORDINATION GROUPA=c2 GROUPB=OW R_0=0.25 NLIST NL_CUTOFF=1.2 NL_STRIDE=50

METAD ...
 LABEL=mtd
 ARG=d,LIGsolv PACE=8000 HEIGHT=0.5 SIGMA=0.25,0.25 FILE=HILLS BIASFACTOR=6 TEMP=298 ACCELERATION ACCELERATION_RFILE=COLVAR
... METAD

COMMITTOR ...
  ARG=d
  STRIDE=100
  BASIN_LL1=1.0
  BASIN_UL1=1.5

  FILE=commit.log FMT=%8.4f
... COMMITTOR

PRINT STRIDE=100 ARG=d,LIGsolv,mtd.bias,mtd.acc FILE=COLVAR_CC

"""

plumed_force = PlumedForce(plumed_script)

system.addForce(plumed_force)

nsteps=50000000
platform = openmm.Platform.getPlatformByName('CUDA')
integrator = openmm.LangevinMiddleIntegrator(298*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
simulation = app.Simulation(pdb.topology, system, integrator, platform)
#simulation.context.reinitialize(preserveState=True)

if args.case =='restart':
        simulation.loadCheckpoint('metad.chk')
else:

    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(298*unit.kelvin,5)

simulation.reporters.append(app.DCDReporter('metad.dcd', 10000))
simulation.reporters.append(app.StateDataReporter(f'log_{args.case}.csv', 50000, step=True, potentialEnergy=False, density=False,temperature=False, speed=True))
simulation.reporters.append(app.PDBReporter('Umb_2.pdb',nsteps))
simulation.step(nsteps)



