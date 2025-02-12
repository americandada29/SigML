import sig_lib
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_sinf_model_combined import create_dataset, SigInfModel, FullCNNModel
from pydlr import dlr

def plot_matsubara(iws, sigs, atom, orbital):
  fig, axs = plt.subplots(2)
  axs[0].plot(iws, sigs[atom, orbital].real, marker="o", c='red')
  axs[1].plot(iws, sigs[atom, orbital].imag, marker="o", c='blue')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 





#with open("../atoms_fingerprints.pkl","rb") as f:
#    atoms, fps = pickle.load(f)
#all_sinfs, all_edcs = sig_lib.get_sinf_edc()
#
#
#tts = 0.9
#N = int(len(atoms)*tts)
#test_fps = fps[N:]
#test_sinfs = all_sinfs[N:]

with open("test_data_combined.pkl","rb") as f:
    test_fps, test_sinfs = pickle.load(f)


#model = SigInfModel().double()
model = FullCNNModel().double().to(device)
model.load_state_dict(torch.load("siginf_ann_combined.pth", weights_only=True))

test_dataset = create_dataset(test_fps, test_sinfs, device=device)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

adiff = 0
for x, y in test_dataloader:
    output = model(x).detach().numpy()[0]
    print(output.shape, y[0].shape)
    exit()
    adiff += np.abs(y[0].numpy() - output)
    print(y[0].numpy() - output)
    print(y[0].numpy())
    print(output)
    print("#############")
    #fig, axs = plt.subplots(2)
    #axs[0].plot(test_iws[0], output.imag, marker="o", c="blue")
    #axs[0].plot(test_iws[0], y[0].imag, marker="o", c="red")
    #axs[1].plot(test_iws[0], output.real, marker="o", c="blue")
    #axs[1].plot(test_iws[0], y[0].real, marker="o", c="red")
    #plt.show()
    #exit()
adiff = adiff/len(test_dataloader)
print(adiff)

    












# Emax, beta = 10, np.pi/iws[0]
# d = dlr(lamb = Emax*beta, eps=1e-10)
# Gx = d.lstsq_dlr_from_matsubara(1j*e_iws, msig, beta)




