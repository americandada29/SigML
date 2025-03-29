import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from SigModel import create_dataset, SigInfModel

def plot_matsubara(iws, sigs, atom, orbital):
  fig, axs = plt.subplots(2)
  axs[0].plot(iws, sigs[atom, orbital].real, marker="o", c='red')
  axs[1].plot(iws, sigs[atom, orbital].imag, marker="o", c='blue')




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("FPS/atoms_fingerprints.pkl","rb") as f:
    test_atoms, test_fps = pickle.load(f)


# model = SigInfModel().double()
model = torch.load("../../Pred_Sinf/siginf_ann.pth", weights_only=False).to(device)
model.eval()

ys = []
for f in test_fps:
   x = torch.from_numpy(f).to(device)
   y = model(x).cpu().detach().numpy()[0]
   ys.append(y)
ys = np.array(ys)

with open("Sinfs.pkl","wb") as f:
   pickle.dump(ys, f)



# test_dataset = create_dataset(test_fps, test_sinfs, device=device)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# adiff = 0
# for x, y in test_dataloader:
#     output = model(x).cpu().detach().numpy()[0]
#     adiff += np.abs(y[0].cpu().numpy() - output)
#     print(y[0].cpu().numpy())
#     print(output)
#     print(y[0].cpu().numpy() - output)
#     print("#############")
#     #fig, axs = plt.subplots(2)
#     #axs[0].plot(test_iws[0], output.imag, marker="o", c="blue")
#     #axs[0].plot(test_iws[0], y[0].imag, marker="o", c="red")
#     #axs[1].plot(test_iws[0], output.real, marker="o", c="blue")
#     #axs[1].plot(test_iws[0], y[0].real, marker="o", c="red")
#     #plt.show()
#     #exit()
# adiff = adiff/len(test_dataloader)
# print(adiff)

    












# Emax, beta = 10, np.pi/iws[0]
# d = dlr(lamb = Emax*beta, eps=1e-10)
# Gx = d.lstsq_dlr_from_matsubara(1j*e_iws, msig, beta)




