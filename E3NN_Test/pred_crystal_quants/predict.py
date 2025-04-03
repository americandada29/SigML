from EF_Model import get_standard_ef_model
from Sinf_Model import get_standard_sinf_model
from Full_Sig_Model import get_standard_full_sig_model
from utils import build_data, get_average_neighbor_count, get_sig_file_text
import pickle 
import os 
import numpy as np
from io import StringIO
from tqdm import tqdm

### Testing on previous datasets with already provided self energies ###
source_dir = "../ATOMS_SIGS_EFS_DATA/"
patoms = []
psig_texts = []
pefs = []
for pf in os.listdir(source_dir):
   with open(source_dir + pf, "rb") as f:
       tatoms, tsig_texts, tefs = pickle.load(f)
   patoms.extend(tatoms)
   psig_texts.extend(tsig_texts)
   pefs.extend(tefs)

ini_data = np.loadtxt(StringIO(psig_texts[0]), dtype=np.float128)



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

atoms = read("dmft_input_bcc.extxyz", index=":", format="extxyz")
all_data = build_data(atoms)

n_matsubara = 110
ave_neighbors = get_average_neighbor_count(all_data)

ef_model = get_standard_ef_model(ave_neighbors, "../SAVED_MODELS/ef_model.pth")
sinf_model = get_standard_sinf_model(ave_neighbors, "../SAVED_MODELS/sinf_model.pth")
full_sig_model = get_standard_full_sig_model(n_matsubara, ave_neighbors, "../SAVED_MODELS/full_sig_model.pth")

iws = ini_data[:,0]

save_atoms = []
save_sig_texts = []
save_efs = []
for i in tqdm(range(len(atoms)), desc="Generating self energies..."):
  newatom = atoms[i].copy()
  sinf_pred = sinf_model(all_data[i]).cpu().detach().numpy().flatten()
  sig_pred = full_sig_model(all_data[i]).cpu().detach().numpy()
  ef_pred = ef_model(all_data[i]).cpu().detach().item()
  sig_lines = get_sig_file_text(iws, sig_pred, sinf_pred)

  save_atoms.append(newatom)
  save_sig_texts.append(sig_lines)
  save_efs.append(ef_pred)


with open("dmft_input_atoms_sigs_efs.pkl","wb") as f:
   pickle.dump([save_atoms, save_sig_texts, save_efs], f)














