import sig_lib
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_tailfit import FullRealToComplexCNN, create_dataset, custom_unrenorm

def plot_compare_w_orbitals(output, y, iws):
    fig, axs = plt.subplots(5,2)

    for i in range(len(output)):
        axs[i,0].plot(iws, output[i].imag, marker="o", c="blue")
        axs[i,0].plot(iws, y[0,i].imag, marker="o", c="red")
        axs[i,1].plot(iws, output[i].real, marker="o", c="blue")
        axs[i,1].plot(iws, y[0,i].real, marker="o", c="red")

    plt.show()






    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("../Pred_Sinf/atoms_fingerprints.pkl","rb") as f:
    atoms, fps = pickle.load(f)
all_iws, all_sigs = sig_lib.get_sigs()
all_tails = sig_lib.fit_sig_tails(all_iws, all_sigs)


N_train = int(0.9*len(atoms))

assert len(all_tails) == len(fps)

train_fps = fps[:N_train]
train_gxs = all_tails[:N_train, :,:,0]

val_fps = fps[N_train:]
val_gxs = all_tails[N_train:,:,:,0]


dataset = create_dataset(train_fps, train_gxs, device=device)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

val_dataset = create_dataset(val_fps, val_gxs, device=device)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


model = FullRealToComplexCNN().to(device)
model.load_state_dict(torch.load("tailfit_fcnn.pth", weights_only=True))
    
preds = []
acts = []

for x, y in dataloader:
    outputs = model(x)
    test_o = outputs.cpu().detach().numpy()
    test_y = y.cpu().detach().numpy()
    print(np.abs(test_o - test_y))









