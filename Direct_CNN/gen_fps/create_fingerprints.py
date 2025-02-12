import re
from pathlib import Path
from subprocess import check_output
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import ase
from ase import units
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
from ase.vibrations import Vibrations
import logging
from copy import deepcopy
from ase import Atoms
from ase.optimize.lbfgs import LBFGS
from ase.constraints import ExpCellFilter
from ase.io import read, write
import numpy as np
import fplibmod
import rcovdata



def from_ase(positions, cell, atoms, type_nums):
    lat = cell
    atoms = atoms
    typt = type_nums
    nat = sum(typt)
    #pos = positions
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    rxyz = positions
    znucl = []
    for a in atoms:
        rcov = rcovdata.rcovdata
        for r in rcov:
            if a in r:
                znucl.append(r[0])
    #rxyz = pos
    znucl = np.array(znucl, int)
    return lat, rxyz, types, znucl

def create_fingerprints(atoms, positions, cells, natx):
    fingerprints = []
    dfps = []
    for i, atom in enumerate(atoms):
        all_symbols = atom.get_chemical_symbols()
        unique_symbols = {}
        for a in all_symbols:
            unique_symbols[a] = 0
        unique_symbols = [a for a in unique_symbols]
        type_nums = [all_symbols.count(x) for x in unique_symbols]
        position = np.array(positions[i])
        cell = cells[i]

        lat, rxyz, types, znucl = from_ase(position, cell, unique_symbols, type_nums)
        lmax = 0
        cutoff = 4.0
        contract = False
        ntyp = len(set(types))
        # fp1, dfp1 = fplibmod.get_fp(contract, np.int64(ntyp), natx, lmax, lat, rxyz, types, znucl, cutoff)
        # fp1, dfp1 = fplib3_pure.get_fp(lat, rxyz, types, znucl, contract, 2, np.int64(ntyp), natx, lmax, cutoff)
        fp1, dfp1 = fplibmod.get_fp(lat, rxyz, types, znucl, contract, False, np.int64(ntyp), natx, lmax, cutoff)
        fingerprints.append(fp1)
        dfps.append(dfp1)
    return np.array(fingerprints), np.array(dfps)



#with open("atoms_energies_fenergies.pkl","rb") as f:
#    atoms, es, fs = pickle.load(f)
atoms = read("ran_atoms.extxyz", index=":", format="extxyz")


goodatoms = []
goodfps = []
failcount = 0
passcount = 0
for i in range(len(atoms)):
    try:
      testatom = atoms[i].copy()
      testatom.wrap()
      fingerprint, _ = create_fingerprints([testatom], [testatom.positions], [testatom.get_cell()[:]], natx=50)
      goodfps.append(fingerprint)
      goodatoms.append(testatom)
      print(str(i) + " DONE")
      passcount += 1
      del testatom
    except:
        print(str(i) + " FAILED")
        failcount += 1


print(passcount, failcount)
print(passcount/len(atoms), failcount/len(atoms))


with open("atoms_fingerprints.pkl","wb") as f:
    pickle.dump([goodatoms, goodfps], f)

