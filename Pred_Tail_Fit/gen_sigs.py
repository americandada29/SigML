import sig_lib
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_fullsig import FullRealToComplexCNN, create_dataset
from pydlr import dlr
import sys
from ase.io import read


def write_sig(iws, sig):
    header1 = "# s_oo= [25.94191367094248, 25.97927152427038, 26.05945867814713, 26.06381971479536, 26.03294235710237, 26.04283272307152, 25.96716318919691, 26.0122262627843, 25.98858687107719, 26.0708059030178, 26.04368953407744, 26.0672438591148, 26.14826552518523, 26.14590929289391, 26.12531807149132, 25.96790067151829, 25.96873105519074, 25.97613803667094, 25.95033170658937, 25.97092521181814]\n"
    header2 = "# Edc= [25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175]\n"
    outdata = np.zeros((sig.shape[-1], 1 + 2*sig.shape[0]*sig.shape[1]))
    outdata[:,0] = iws 
    for a in range(sig.shape[0]):
        for o in range(sig.shape[1]):
            outdata[:,1 + 10*a + 2*o] = sig[a,o].real
            outdata[:,1 + 10*a + 2*o + 1] = sig[a,o].imag
    outfile = "test_sig_outs/0_sig.dat"
    np.savetxt(outfile, outdata)
    f = open(outfile, "r")
    lines = [header1, header2]
    for l in f:
        lines.append(l)
    f.close()
    f = open(outfile, "w")
    for l in lines:
        f.write(l)
    f.close()


def get_sig_file_output(iws, sig):
    header1 = "# s_oo= [25.94191367094248, 25.97927152427038, 26.05945867814713, 26.06381971479536, 26.03294235710237, 26.04283272307152, 25.96716318919691, 26.0122262627843, 25.98858687107719, 26.0708059030178, 26.04368953407744, 26.0672438591148, 26.14826552518523, 26.14590929289391, 26.12531807149132, 25.96790067151829, 25.96873105519074, 25.97613803667094, 25.95033170658937, 25.97092521181814]\n"
    header2 = "# Edc= [25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175, 25.175]\n"
    outdata = np.zeros((sig.shape[-1], 1 + 2*sig.shape[0]*sig.shape[1]))
    outdata[:,0] = iws 
    for a in range(sig.shape[0]):
        for o in range(sig.shape[1]):
            outdata[:,1 + 10*a + 2*o] = sig[a,o].real
            outdata[:,1 + 10*a + 2*o + 1] = sig[a,o].imag
    outfile = "test_sig_outs/temp_sig.dat"
    np.savetxt(outfile, outdata)
    f = open(outfile, "r")
    lines = [header1, header2]
    for l in f:
        lines.append(l)
    f.close()
    return lines

def plot_compare_w_orbitals(output, y, iws):
    fig, axs = plt.subplots(5,2)

    for i in range(len(output)):
        axs[i,0].plot(iws, output[i].imag, marker="o", c="blue")
        axs[i,0].plot(iws, y[0,i].imag, marker="o", c="red")
        axs[i,1].plot(iws, output[i].real, marker="o", c="blue")
        axs[i,1].plot(iws, y[0,i].real, marker="o", c="red")

    plt.show()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_iws, all_sigs = sig_lib.get_sigs()


with open("gen_fps/atoms_fingerprints.pkl", "rb") as f:
    atoms, fps = pickle.load(f)
fps = np.array(fps).astype(np.float32)
fps = fps[:, 0, :, :]
iws = all_iws[0]

model = FullRealToComplexCNN().to(device)
model.load_state_dict(torch.load("fullsig_cnn.pth", weights_only=True))

inds = np.arange(len(fps))
np.random.shuffle(inds)

new_atoms = []
pred_sigs = []
for i in range(5):
    ind = inds[i]
    atom = atoms[ind]
    fp = fps[ind]
    ys = []
    for j in range(len(fp)):
        x = torch.from_numpy(fp[j]).view(1, len(fp[j])).to(device)
        y = model(x).cpu().detach().numpy()[0]
        ys.append(y)
    ys = np.array(ys)
    loutput = get_sig_file_output(iws, ys)
    pred_sigs.append(loutput)
    new_atoms.append(atom)

with open("atoms_predsigs.pkl","wb") as f:
    pickle.dump([new_atoms, pred_sigs], f)


# with open("../Data/atoms_fingerprints.pkl","rb") as f:
#     atoms, fps = pickle.load(f)
# all_iws, all_gxs = sig_lib.read_gxs()
# all_iws, all_sigs = sig_lib.get_sigs()

# test_fps = fps[80:]
# test_gxs = all_sigs[80:]
# test_iws = all_iws[80:]










# test_dataset = create_dataset(test_fps, test_gxs, device="cuda")
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# for x, y in test_dataloader:
#     output = model(x).cpu().detach().numpy()[0]

#     plot_compare_w_orbitals(output, y.cpu(), test_iws[0])
#     exit()
    
    












# Emax, beta = 10, np.pi/iws[0]
# d = dlr(lamb = Emax*beta, eps=1e-10)
# Gx = d.lstsq_dlr_from_matsubara(1j*e_iws, msig, beta)




