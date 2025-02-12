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
import sig_lib
from ase.calculators.singlepoint import SinglePointCalculator



atoms, efs = sig_lib.get_efs()



trainatoms = []
testatoms = []
failcount = 0
passcount = 0

tts = 0.9
Ntrain = tts*len(atoms)


for i in range(len(atoms)):
   try:
     testatom = atoms[i].copy()
     testatom.wrap()
     testatom.calc = SinglePointCalculator(atoms=testatom, energy=efs[i], forces=np.zeros(testatom.positions.shape), stress=None)
     print(testatom.get_potential_energy())
     if i < Ntrain:
       trainatoms.append(testatom)
     if i > Ntrain:
       testatoms.append(testatom)
     print(str(i) + " DONE")
     passcount += 1
     del testatom
   except:
     print(str(i) + " FAILED")
     failcount += 1


print(passcount, failcount)
print(passcount/len(atoms), failcount/len(atoms))

write("train.extxyz", trainatoms, format="extxyz")
write("test.extxyz", testatoms, format="extxyz")






