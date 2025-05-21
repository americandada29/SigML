from EF_Model import get_standard_ef_model
from Sinf_Model import get_standard_sinf_model
from Sig_iws_Model import get_standard_full_sig_model
from utils import build_data, get_average_neighbor_count, get_sig_file_text, parse_sig_file, evaluate_full_sig_legendre
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
test_data = build_data(atoms, sig_texts=sigs, efs=efs, device=device, radial_cutoff=5.0)


full_sig_model = get_standard_full_sig_model(n_matsubara, ave_neighbors, weight_path = "SAVED_MODELS/full_sig_model_fe2o2.pth", radial_cutoff=5.0, device=device)
for o in range(5):
    evaluate_full_sig_legendre(full_sig_model, test_data, orbital=o, atom=1, display=True)
exit()

















