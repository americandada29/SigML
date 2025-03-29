import pickle
import numpy as np
from nequip.ase.nequip_calculator import nequip_calculator
import matplotlib.pyplot as plt
from ase.io import read, write



ef_calc = nequip_calculator("../../Pred_EF_NEQUIP/deployed_model/efsmodel.pth")

atoms = read("dmft_input_atoms.extxyz", index=":", format='extxyz')

efs = []
for a in atoms:
    a.calc = ef_calc 
    ef = a.get_potential_energy()
    print(ef)
    efs.append(ef)
efs = np.array(efs)
print(efs)
with open("Efs.pkl","wb") as f:
    pickle.dump(efs, f)