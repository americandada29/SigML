from pydlr import dlr
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP



def parse_sig_file(sig_text):
  sig_lines = sig_text.split("\n")
  s_infs = np.array([float(x) for x in sig_lines[0].split("[")[1].split("]")[0].split(",")], dtype=np.float64)
  data = np.loadtxt(sig_lines[2:])
  iws = data[:,0]
  sig_data = data[:, 1:]
  adjusted_sig_data = np.zeros((sig_data.shape[0], int(sig_data.shape[1]/2)), dtype=np.complex128)
  for i in range(adjusted_sig_data.shape[1]):
    #adjusted_sig_data[:,i] = sig_data[:,2*i] + s_infs[i] + 1j*sig_data[:,2*i+1]
    adjusted_sig_data[:,i] = sig_data[:,2*i] + 1j*sig_data[:,2*i+1]
  return iws, adjusted_sig_data


def plot_sig(iws, S):
  fig, axs = plt.subplots(2)
  axs[0].plot(iws.imag, S.real, marker="o")
  axs[1].plot(iws.imag, S.imag, marker="o")
  plt.show()

def compare_sigs(iws, S_org, S_test):
  fig, axs = plt.subplots(2)
  axs[0].plot(iws.imag, S_org.real, c='black')
  axs[0].plot(iws.imag, S_test.real, c='red', marker='o')
  axs[1].plot(iws.imag, S_org.imag, c='black')
  axs[1].plot(iws.imag, S_test.imag, c='red', marker='o')
  plt.show() 

def get_dlr_coeffs(iws, S, E_max=50.0):
  beta = np.pi/iws[0]
  S_test = np.hstack((np.flip(np.conj(S)), S))
  S_test = S_test.reshape((S_test.shape[0], 1, 1))
  iws_test = np.hstack((np.flip(-1j*iws), 1j*iws))
  d = dlr(lamb=beta*E_max, eps=1e-10)
  w_x = d.get_dlr_frequencies()
  S_dlr = d.lstsq_dlr_from_matsubara(iws_test, S_test, beta=beta)
  S_recon = d.eval_dlr_freq(S_dlr, iws_test, beta, xi=-1)
  #compare_sigs(iws_test, S_test[:,0,0], S_recon[:,0,0])
  print(S_dlr)

files = ["dmft_2_as.pkl","dmft_as.pkl"]
atoms = []
sigs = []
for f in files:
  with open(f, "rb") as g:
    tatoms, tsigs = pickle.load(g)
  atoms.extend(tatoms)
  sigs.extend(tsigs)



iws, sig_data = parse_sig_file(sigs[0])
get_dlr_coeffs(iws, sig_data[:,0])
