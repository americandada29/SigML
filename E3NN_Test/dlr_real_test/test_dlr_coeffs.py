import numpy as np
from pydlr import dlr
from pydlr import kernel
from scipy.integrate import quad
from dlr_testing import parse_sig_file, write_all_dlr_coeffs, read_dlr_coeffs
import pickle
import matplotlib.pyplot as plt

with open("atoms_sigs.pkl","rb") as f:
    atoms, sig_texts = pickle.load(f)
iws, sig = parse_sig_file(sig_texts[0])


def giw_to_gtau(iws, giw):
    iws_input = np.hstack((-np.flip(iws), iws))
    giw_input = np.hstack((np.flip(np.conj(giw)), giw))
    beta = np.pi/iws[0]
    taus = np.linspace(0, beta, 1000)
    gtau = np.zeros(len(taus), dtype=np.complex128)
    for i, t in enumerate(taus):
        gtau[i] = np.sum((1/beta)*giw_input*np.exp(-1j*iws_input*t))
    return taus, gtau
    


## Testing for atom 1 orbital 1
sig_test = sig[0,0]
sig_input = np.zeros((len(sig_test), 1, 1), dtype=np.complex128)
sig_input[:,0,0] = sig_test 

H = np.array([[1.0]])

beta = np.pi/iws[0]
E_max = 130
d = dlr(beta*E_max, eps=1e-12, xi=-1, nmax=len(iws))

# print(d.get_matsubara_frequencies(beta))
# print(iws)
# exit()


G_input = 1/(iws - sig_test)/100
G_input = np.hstack((np.flip(np.conj(G_input)), G_input))
G_input = np.reshape(G_input, (len(G_input), 1, 1))

iws_input = 1j*np.hstack((-np.flip(np.conj(iws)), iws))

G_dlr = d.lstsq_dlr_from_matsubara(iws_input, G_input, beta)

print(G_dlr)





