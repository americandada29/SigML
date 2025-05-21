from EF_Model import get_standard_ef_model
from Sinf_Model import get_standard_sinf_model
from Sig_iws_Model import get_standard_full_sig_model
from utils import build_data, get_average_neighbor_count, get_sig_file_text
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

##### IMPORTANT: Be sure to adjust the radial cutoff #####
radial_cutoff = 5.0

source_save_dir = "../DMFT_4_SIGML_LEG/"
atoms = read(source_save_dir + "dmft_input_atoms.extxyz", index=":", format="extxyz")
all_data = build_data(atoms, radial_cutoff=radial_cutoff)



# ef_calc = nequip_calculator("SAVED_MODELS/nequip_ef_model.pth")
ef_model = get_standard_ef_model(ave_neighbor_count=ave_neighbors, weight_path="SAVED_MODELS/ef_model_fe2o2.pth", cutoff=radial_cutoff, device=device)
sinf_model = get_standard_sinf_model(ave_neighbors, weight_path = "SAVED_MODELS/sinf_model_fe2o2.pth", cutoff=radial_cutoff, device=device)
full_sig_model = get_standard_full_sig_model(n_matsubara, ave_neighbors, weight_path = "SAVED_MODELS/full_sig_model_fe2o2.pth", radial_cutoff=radial_cutoff, device=device)



random_plot_inds = np.random.choice(np.arange(0, len(atoms)), 6, replace=False)
fig, axs = plt.subplots(2, 3)
plot_count = 0

save_atoms = []
save_sig_texts = []
save_efs = []
for i in tqdm(range(len(atoms)), desc="Generating self energies..."):
   newatom = atoms[i].copy()
   sinf_pred = sinf_model(all_data[i]).cpu().detach().numpy().flatten()
   sig_pred = full_sig_model(all_data[i]).cpu().detach().numpy()
   sig_pred = fullatom_giw_from_gl(iws, sig_pred)
   ef_pred = ef_model(all_data[i]).cpu().detach().item()

   

   if i in random_plot_inds:
      atom = np.random.randint(0, sig_pred.shape[0])
      orbital = np.random.randint(0, sig_pred.shape[-1])
      axs[plot_count//3, plot_count%3].plot(iws, sig_pred[atom, :, orbital].real, c="red")
      axs[plot_count//3, plot_count%3].plot(iws, sig_pred[atom, :, orbital].imag, c="blue")
      print(plot_count, ef_pred, sinf_pred)
      plot_count += 1

   # newatom.calc = ef_calc 
   # ef_pred = newatom.get_potential_energy()


   sig_lines = get_sig_file_text(iws, sig_pred, sinf_pred, all_data[i])

   save_atoms.append(atoms[i])
   save_sig_texts.append(sig_lines)
   save_efs.append(ef_pred)

plt.show()

with open(source_save_dir + "dmft_input_atoms_sigs_efs.pkl","wb") as f:
   pickle.dump([save_atoms, save_sig_texts, save_efs], f)














