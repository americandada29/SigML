import sig_lib
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_fullsig import FullRealToComplexCNN, create_dataset, custom_unrenorm
from pydlr import dlr

def plot_compare_w_orbitals(output, y, iws):
    fig, axs = plt.subplots(5,2)

    for i in range(len(output)):
        axs[i,0].plot(iws, output[i].imag, marker="o", c="blue")
        axs[i,0].plot(iws, y[0,i].imag, marker="o", c="red")
        axs[i,1].plot(iws, output[i].real, marker="o", c="blue")
        axs[i,1].plot(iws, y[0,i].real, marker="o", c="red")

    plt.show()






    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



with open("../Data/atoms_fingerprints.pkl","rb") as f:
    atoms, fps = pickle.load(f)
all_iws, all_gxs = sig_lib.read_gxs()
all_iws, all_sigs = sig_lib.get_sigs()

test_fps = fps[80:]
test_gxs = all_sigs[80:]
test_iws = all_iws[80:]


model = FullRealToComplexCNN().to(device)
model.load_state_dict(torch.load("fullsig_cnn.pth", weights_only=True, map_location=torch.device(device)))

test_dataset = create_dataset(test_fps, test_gxs, device=device)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

for x, y in test_dataloader:
    output = model(x).cpu().detach().numpy()[0]

    plot_compare_w_orbitals(output, y.cpu(), test_iws[0])
    exit()
    
    












# Emax, beta = 10, np.pi/iws[0]
# d = dlr(lamb = Emax*beta, eps=1e-10)
# Gx = d.lstsq_dlr_from_matsubara(1j*e_iws, msig, beta)




