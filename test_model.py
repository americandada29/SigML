import sig_lib
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn_test_simple import RealToComplexCNN, create_dataset, custom_unrenorm
from pydlr import dlr





## Takes in target (5xM) and output (5xM) and plots predicted and actual self energies
def plot_predictions(target, output, iws):
    t = custom_unrenorm(target.detach().numpy()[0])
    o = custom_unrenorm(output.detach().numpy()[0])
    Emax, beta = 10, np.pi/iws[0]
    d = dlr(lamb = Emax*beta, eps=1e-10)

    act_siws = []
    pred_siws = []

    e_iws, msigs = sig_lib.proj_sig_full_matsubara(iws, t)

    # print(msigs.shape)
    # print(e_iws.shape)
    # print(e_iws)
    # print(msigs)
    # exit()

    fig, axs = plt.subplots(len(t),2)
    for i in range(len(t)):
        act_siw = d.eval_dlr_freq(t[i].reshape((19,1,1)), 1j*e_iws, beta)[:,0,0]
        pred_siw = d.eval_dlr_freq(o[i].reshape((19,1,1)), 1j*e_iws, beta)[:,0,0]
        act_siws.append(act_siw)
        pred_siws.append(pred_siw)

        print(act_siw)
        exit()

        axs[i,0].plot(e_iws, act_siw.real, c="red", marker="o", label="Actual")
        axs[i,0].plot(e_iws, pred_siw.real, c="blue", marker="o", label="Pred")
        axs[i,1].plot(e_iws, act_siw.imag, c="red", marker="o", label="Actual")
        axs[i,1].plot(e_iws, pred_siw.imag, c="blue", marker="o", label="Pred")
    plt.legend()
    plt.show()





with open("atoms_fingerprints.pkl","rb") as f:
    atoms, fps = pickle.load(f)
all_iws, all_gxs = sig_lib.read_gxs()
all_iws, all_sigs = sig_lib.get_sigs()

test_fps = fps[80:]
test_gxs = all_gxs[80:]
test_iws = all_iws[80:]


model = RealToComplexCNN()
model.load_state_dict(torch.load("simple_cnn.pth", weights_only=True))

test_dataset = create_dataset(test_fps, test_gxs, device="cpu")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

count = 0
for x, y in test_dataloader:
    output = model(x)
    plot_predictions(y, output, test_iws[count])
    exit()

    count += 1








# Emax, beta = 10, np.pi/iws[0]
# d = dlr(lamb = Emax*beta, eps=1e-10)
# Gx = d.lstsq_dlr_from_matsubara(1j*e_iws, msig, beta)




