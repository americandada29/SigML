from utils import build_data, get_average_neighbor_count, get_sig_file_text, \
                  train_test_split, train_full_sig, evaluate_full_sig, train_nequip_ef, \
                  eval_nequip_ef, train_sinf, evaluate_sinf
import pickle 
import os 
import numpy as np
from io import StringIO
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from leg_lib import fullatom_gl_from_giw ,fullatom_giw_from_gl #, fullatom_high_freq_moments, fit_hfm
from scipy.signal import savgol_filter

def make_sig_complex(sig):
   newsig = torch.view_as_complex(sig).squeeze(0).cpu().numpy()
   return newsig

def plot_sig(iws, sig, atom=0):
   num_orbitals = sig.shape[-1]
   fig, axs = plt.subplots(num_orbitals)
   for i in range(num_orbitals):
      axs[i].plot(iws, sig.real[atom, :, i], c='blue', marker='o')
      axs[i].plot(iws, sig.imag[atom, :, i], c='red', marker='o')
      axs[i].axhline(0, color='black', linestyle='--')
   plt.show()

def compare_sigs(iws1, sig1, iws2, sig2, atom=0, hfm_matrix=None, smooth=False):
   num_orbitals = sig1.shape[-1]
   fig, axs = plt.subplots(2, 3)
   for i in range(5):
      axs[i//3, i%3].plot(iws1, sig1.real[atom, :, i], c='black', marker='o', markersize=2)
      axs[i//3, i%3].plot(iws2, sig2.real[atom, :, i], c='blue', marker='o'  ,markersize=2)
      axs[i//3, i%3].plot(iws1, sig1.imag[atom, :, i], c='black', marker='o'   ,markersize=2)
      axs[i//3, i%3].plot(iws2, sig2.imag[atom, :, i], c='purple', marker='o',markersize=2)
      axs[i//3, i%3].axhline(0, color='black', linestyle='--')

      if hfm_matrix is not None:
         hind = 35
         halfway = iws2[hind:]
         gf_tail = hfm_matrix[atom, 0, i]/(halfway) 
         gf_tail_diff = np.diff(gf_tail)
         newgftail = [sig2.imag[atom, hind, i]]
         for j in range(0,len(gf_tail_diff)):
            newgftail.append(newgftail[-1] + gf_tail_diff[j])
         newgftail = np.array(newgftail)
         sig2_newtail = np.concatenate((sig2[atom, :hind, i], sig2[atom, hind:, i].real + 1j*newgftail))
         # axs[i//3, i%3].plot(halfway, gf_tail.real, c='black', marker='o')
         axs[i//3, i%3].plot(iws1, sig2_newtail.real, c='black', marker='o', markersize=2)
         axs[i//3, i%3].plot(iws1, sig2_newtail.imag, c='black', marker='o', markersize=2)
      
      if smooth:
         hind = len(iws2)//2
         sig2_first = sig2.imag[atom, :hind, i]
         sig2_last = sig2.imag[atom, hind:, i] 
         # sig2_last_smooth = savgol_filter(sig2_last, window_length=20, polyorder=1)
         kernel = np.ones(30)/30 
         sig2_last_smooth = np.convolve(sig2_last, kernel, mode='same')
         sig2_smooth = np.concatenate((sig2_first, sig2_last_smooth))
         axs[i//3, i%3].plot(iws2, sig2_smooth, c='black', marker='o', markersize=2)
      
      # axs[i].set_ylim(-1, 1)
   plt.show()

def extend_sig(iws, sig):
   beta = np.pi/iws[0]
   extended_iws = np.array([(2*n+1)*np.pi/beta for n in range(len(iws)-1, 400)])
   hfm_matrix = fit_hfm(iws, sig, len(iws)//2, len(iws))
   newsig = np.zeros((sig.shape[0], len(extended_iws)-1 + len(iws), sig.shape[-1]), dtype=np.complex128)
   for i in range(sig.shape[0]):
      for j in range(sig.shape[-1]):
      #   halfway = iws[hind:]
        gf_tail = hfm_matrix[i, 0, j]/extended_iws
        gf_tail_diff = np.diff(gf_tail)
        newgftail = [sig[i, -1, j].imag]
        for k in range(0, len(gf_tail_diff)):
          newgftail.append(newgftail[-1] + gf_tail_diff[k])
        newgftail = np.array(newgftail[1:])
        newsig[i, :, j] = np.concatenate((sig[i, :, j], 1j*newgftail))

   return np.concatenate((iws, extended_iws[1:])), newsig



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# print(f"Using {device}\n")



## Testing on previous datasets with already provided self energies ###
# source_dir = "../ATOMS_SIGS_EFS_DATA/"
# patoms = []
# psig_texts = []
# pefs = []
# for pf in os.listdir(source_dir):
#    if pf != "dmft_2_3_asefs.pkl":
#       continue
#    with open(source_dir + pf, "rb") as f:
#        tatoms, tsig_texts, tefs = pickle.load(f)
#    patoms.extend(tatoms)
#    psig_texts.extend(tsig_texts)
#    pefs.extend(tefs)
# dataset = build_data(patoms[49:51], sig_texts=psig_texts, efs=pefs, device=device)

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)



inds = np.random.choice(np.arange(0, len(dataset)), 5, replace=False)

for i in inds:
   test_data = dataset[i]
   test_sig = test_data.sig_org
   test_iws = test_data.iws[0].cpu().numpy()

   sig_l = fullatom_gl_from_giw(test_iws, test_sig, lmax=30)
   recon_sig = fullatom_giw_from_gl(test_iws, sig_l)
   compare_sigs(test_iws, test_sig, test_iws, recon_sig, hfm_matrix = None, smooth=False)










