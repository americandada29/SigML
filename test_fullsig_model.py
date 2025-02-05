import sig_lib
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from full_sig_test import RealToComplexCNN, create_dataset, custom_unrenorm
from pydlr import dlr

def plot_matsubara(iws, sigs, atom, orbital):
  fig, axs = plt.subplots(2)
  axs[0].plot(iws, sigs[atom, orbital].real, marker="o", c='red')
  axs[1].plot(iws, sigs[atom, orbital].imag, marker="o", c='blue')





    





with open("atoms_fingerprints.pkl","rb") as f:
    atoms, fps = pickle.load(f)
all_iws, all_gxs = sig_lib.read_gxs()
all_iws, all_sigs = sig_lib.get_sigs()

test_fps = fps[80:]
test_gxs = all_sigs[80:]
test_iws = all_iws[80:]


model = RealToComplexCNN()
model.load_state_dict(torch.load("fullsig_cnn.pth", weights_only=True))

test_dataset = create_dataset(test_fps, test_gxs, device="cpu")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for x, y in test_dataloader:
    output = model(x).detach().numpy()[0]

    fig, axs = plt.subplots(2)
    axs[0].plot(test_iws[0], output[0].imag, marker="o", c="blue")
    axs[0].plot(test_iws[0], y[0][0].imag, marker="o", c="red")
    axs[1].plot(test_iws[0], output[0].real, marker="o", c="blue")
    axs[1].plot(test_iws[0], y[0][0].real, marker="o", c="red")
    plt.show()
    exit()
    
    












# Emax, beta = 10, np.pi/iws[0]
# d = dlr(lamb = Emax*beta, eps=1e-10)
# Gx = d.lstsq_dlr_from_matsubara(1j*e_iws, msig, beta)




