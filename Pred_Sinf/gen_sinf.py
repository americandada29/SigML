import sig_lib
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_sinf_model import create_dataset, SigInfModel
from pydlr import dlr
from create_fingerprints import create_fingerprints
from ase.io import read, write

def plot_matsubara(iws, sigs, atom, orbital):
  fig, axs = plt.subplots(2)
  axs[0].plot(iws, sigs[atom, orbital].real, marker="o", c='red')
  axs[1].plot(iws, sigs[atom, orbital].imag, marker="o", c='blue')




with open("test_atoms_predsigs.pkl","rb") as f:
    atoms, psigs = pickle.load(f)


goodatoms = []
goodfps = []
for i in range(len(atoms)):
    #try:
    testatom = atoms[i].copy()
    testatom.wrap()
    fingerprint, _ = create_fingerprints([testatom], [testatom.positions], [testatom.get_cell()[:]], natx=50)
    goodfps.append(fingerprint)
    goodatoms.append(testatom)
    #except:
    #  print(str(i), "SKIPPING")


model = SigInfModel().double()
model.load_state_dict(torch.load("siginf_ann.pth", weights_only=True))

outstring = "# s_oo= ["
x = torch.from_numpy(goodfps[1]).double()
for i in range(4):
    output = model(x[:, i, :]).cpu().detach().numpy()[0]
    txt1 = "{s1:.14f}, {s2:.14f}, {s3:.14f}, {s4:.14f}, {s5:.14f}, "
    txt2 = "{s1:.14f}, {s2:.14f}, {s3:.14f}, {s4:.14f}, {s5:.14f}"
    if i == 3:
      outstring += txt2.format(s1=output[0],s2=output[1],s3=output[2],s4=output[3], s5=output[4])
    else:
      outstring += txt1.format(s1=output[0],s2=output[1],s3=output[2],s4=output[3], s5=output[4])
outstring = outstring + "]\n"
print(outstring.split())

print(outstring)

exit()





with open("../atoms_fingerprints.pkl","rb") as f:
    atoms, fps = pickle.load(f)
all_sinfs, all_edcs = sig_lib.get_sinf_edc()

test_fps = fps[80:]
test_sinfs = all_sinfs[80:]


model = SigInfModel().double()
model.load_state_dict(torch.load("siginf_ann.pth", weights_only=True))

test_dataset = create_dataset(test_fps, test_sinfs, device="cpu")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

adiff = 0
for x, y in test_dataloader:
    output = model(x).detach().numpy()[0]
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




