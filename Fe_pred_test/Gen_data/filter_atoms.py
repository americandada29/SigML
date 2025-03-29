from ase.io.trajectory import Trajectory
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
import os 
import pickle
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymatgen.analysis.local_env import CrystalNN

radius = 1.6


def print_nn_dists(struct):
  cnn = CrystalNN()
  natoms = len(struct)
  dists = []
  for i in range(natoms):
    for j in range(i):
      dist = struct.get_distance(i, j)
      dists.append(dist)
  dists = np.array(dists)/0.529177
#   print(dists)
  diameter = 2*radius
  passed = True
  for d in dists:
    if d < diameter:
      passed = False
  if not(passed):
    return dists
  else:
    return None


dirs = sorted(os.listdir("MD_65P"), key=lambda x: float(x.split("_")[0]))
all_atoms = []
for d in dirs:
    traj = Trajectory("MD_65P/" + d, "r")
    atoms = traj[300:]
    good_atoms = [atoms[i] for i in range(0, len(atoms), 61)]
    tgas = []
    for i in range(len(good_atoms)):
        good_atoms[i].wrap()
        dists = print_nn_dists(AAA.get_structure(good_atoms[i]))
        if dists is None:
          tgas.append(good_atoms[i])
    all_atoms.extend(tgas)


print(len(all_atoms))


write("dmft_input_atoms.extxyz", all_atoms, format='extxyz')

