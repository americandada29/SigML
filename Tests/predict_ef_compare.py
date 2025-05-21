from EF_Model import get_standard_ef_model
from Sinf_Model import get_standard_sinf_model
from Sig_iws_Model import get_standard_full_sig_model
from utils import build_data, get_average_neighbor_count, get_sig_file_text, parse_sig_file
import pickle 
import os 
import numpy as np
from io import StringIO
from tqdm import tqdm
from nequip.ase.nequip_calculator import nequip_calculator
import torch
from leg_lib import fullatom_giw_from_gl

torch.set_default_dtype(torch.float32) 
device = torch.device("cpu")

### Testing on previous datasets with already provided self energies ###
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)
ave_neighbors = get_average_neighbor_count(dataset)
iws = dataset[0].iws[0].cpu().numpy()
n_matsubara = dataset[0].sig.shape[2]


#n_matsubara = all_data[0].sig.shape[2]
#ave_neighbors = get_average_neighbor_count(all_data)
#
#
#
#ef_model = get_standard_ef_model(ave_neighbors, "../SAVED_MODELS/ef_model.pth")
#sinf_model = get_standard_sinf_model(ave_neighbors, "../SAVED_MODELS/sinf_model.pth")
#full_sig_model = get_standard_full_sig_model(n_matsubara, ave_neighbors, "../SAVED_MODELS/full_sig_model.pth")

# atom = 0
# fig, axs = plt.subplots(5)
# for i in range(5):
#   axs[i].plot(iws, sig_pred[atom, :, i, 0], marker="o", markersize=3, c='blue')
#   axs[i].plot(iws, sig_pred[atom, :, i, 1], marker="o", markersize=3, c='red')

# plt.show()



### Creating new self energies using novel dataset to test performance ###
from ase.io import read
import matplotlib.pyplot as plt

source_save_dir = "../DMFT_4_SIGML_LEG/"
# atoms = read(source_save_dir + "dmft_input_atoms.extxyz", index=":", format="extxyz")
with open(source_save_dir + "atoms_sigs_efs.pkl","rb") as f:
   atoms, sigs, efs = pickle.load(f)
all_data = build_data(atoms, radial_cutoff=5.0)


model_type = "custom"


if model_type == "custom":
   ef_model = get_standard_ef_model(ave_neighbors, weight_path = "SAVED_MODELS/ef_model_fe2o2.pth", cutoff=5.0, device=device)
   amin = 1000000
   amax = -10000000
   mae = 0
   for i in range(len(atoms)):
      ef_pred = ef_model(all_data[i]).cpu().detach().item()
      ef_act = efs[i]
      n_atoms = atoms[i].get_chemical_symbols().count("Fe")
      amin = min(amin, np.amin([ef_pred, ef_act]))
      amax = max(amax, np.amax([ef_pred, ef_act]))
      mae += np.abs(ef_pred - ef_act)
      plt.scatter(ef_pred, ef_act)

   mae /= len(atoms)
   x = np.linspace(amin - 0.5, amax + 0.5)
   plt.xlim(amin-0.5, amax+0.5)
   plt.ylim(amin-0.5, amax+0.5)
   plt.plot(x, x, linestyle="--", c='black')
   plt.text(amin, amax, f"MAE: {mae:.5f}", fontsize=15)
   plt.show()
      
elif model_type == "nequip":
   ef_model = nequip_calculator("SAVED_MODELS/nequip_ef_model_fe2o2.pth")
   amin = 1000000
   amax = -10000000
   mae = 0
   for i in range(len(atoms)):
      atoms[i].calc = ef_model
      ef_pred = atoms[i].get_potential_energy()
      ef_act = efs[i]
      amin = min(amin, np.amin([ef_pred, ef_act]))
      amax = max(amax, np.amax([ef_pred, ef_act]))
      mae += np.abs(ef_pred - ef_act)
      plt.scatter(ef_pred, ef_act)

   mae /= len(atoms)
   x = np.linspace(amin - 0.5, amax + 0.5)
   plt.xlim(amin-0.5, amax+0.5)
   plt.ylim(amin-0.5, amax+0.5)
   plt.plot(x, x, linestyle="--", c='black')
   plt.text(amin, amax, f"MAE: {mae:.5f}", fontsize=15)
   plt.show()















