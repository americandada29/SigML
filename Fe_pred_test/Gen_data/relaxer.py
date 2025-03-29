from pymatgen.core import Lattice, Structure
import matgl
from matgl.ext.ase import MolecularDynamics, Relaxer
import warnings
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from matgl.ext.ase import M3GNetCalculator, MolecularDynamics, Relaxer
from matgl.apps.pes import Potential
from matgl.ext.ase import PESCalculator
import numpy as np
import matplotlib.pyplot as plt
from ase.units import GPa
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from ase.io import read, write

radius = 3
def print_nn_dists(struct):
  cnn = CrystalNN()
  natoms = len(struct)
  dists = []
  for i in range(natoms):
    for j in range(i):
      dist = struct.get_distance(i, j)
      dists.append(dist)
  dists = np.array(dists)/0.529177
  print(dists)
  diameter = 2*radius
  passed = True
  for d in dists:
    if d < diameter:
      passed = False
  if not(passed):
    return dists
  else:
    return None

model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
#pot = Potential(model)
#calc = M3GNetCalculator(pot)
calc = PESCalculator(model)

warnings.simplefilter("ignore")


pressure = 300*GPa

vratios = np.linspace(0.3,1.3,50)


fe_ase_org = read("../Poscars/229", format="vasp")

energies = []
volumes = []
for v in vratios:
  newatom = fe_ase_org.copy()
  newatom.set_cell(newatom.get_cell()[:]*(v**(1/3)), scale_atoms=True)
  newatom.calc = calc
  energy = newatom.get_potential_energy() + newatom.get_volume()*pressure
  volume = np.linalg.det(newatom.cell)
  print(volume, energy)
  print_nn_dists(AAA.get_structure(newatom))
  energies.append(energy)
  volumes.append(v)



plt.plot(volumes, energies, marker="o")
plt.show()



