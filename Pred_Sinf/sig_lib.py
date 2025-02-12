import numpy as np
import matplotlib.pyplot as plt
from pydlr import dlr
import os
from scipy.optimize import curve_fit
import pickle

def high_freq_func(wn, c):
  return c/wn


def get_high_freq_tail(iws, sigs):
  tail_threshold = 0.001 ## Threshold where the real part of Sigma is small enough for the tail for be fit
  tails = np.zeros(sigs.shape[:2])
  for i in range(sigs.shape[0]):
    for j in range(sigs.shape[1]):
      reals = sigs[i,j].real
      below_thresh_ind = np.flatnonzero(reals < tail_threshold)[0]
      popt = curve_fit(high_freq_func, iws[below_thresh_ind:], sigs[i,j,below_thresh_ind:].imag)[0][0]
      tails[i,j] = popt
  return tails
      
  

def plot_matsubara(iws, sigs, atom, orbital):
  fig, axs = plt.subplots(2)
  axs[0].plot(iws, sigs[atom, orbital].real, marker="o", c='red')
  axs[1].plot(iws, sigs[atom, orbital].imag, marker="o", c='blue')
  plt.show() 


def get_sigs(path="./"):
  all_sigs = []
  all_iws = []
  for fi in os.listdir(path + "self_energies"):
    data = np.loadtxt(path + "self_energies/" + fi)
    iws = data[:,0]
    all_iws.append(iws)
    sigs = np.zeros((4, 5, data.shape[0])).astype(np.complex128)  # 4 atoms, 5 orbitals, 2 numbers for real and imag parts
    for i in range(4):
      for j in range(5):
        sigs[i,j] = data[:, 1 + 10*i + 2*j] + 1j*data[:, 1 + 10*i + 2*j + 1]
    all_sigs.append(sigs)
  return all_iws, all_sigs

def sig_to_greens(iws, sigs, tails):
  Giws = np.zeros(sigs.shape).astype(np.complex128)
  for i in range(sigs.shape[0]):
    for j in range(sigs.shape[1]):
      Giws[i,j] = 1/(1j*iws - sigs[i,j] + tails[i,j]/(1j*iws)) + 1/(1j*iws + tails[i,j]/(1j*iws))
  return Giws

def giw_to_gtau(iws, Giw, taus):
  beta = np.pi/iws[0]
  Niw = len(iws)
  Gtau = np.zeros(len(taus)).astype(np.complex128)
  for t, tau in enumerate(taus):
    for i in range(Niw):
      Gtau[t] += (1/beta)*np.real(np.exp(-1j*(2*i+1)*np.pi/beta*tau)*Giw[i])/2
  return Gtau

      
def get_real_giws():
  with open("Giws/Giws.pkl","rb") as f:
    data = pickle.load(f)
  Giws = np.zeros((4,5,len(data[0][0]))).astype(np.complex128)
  iws = data[0][0][:,0]

  for k in range(len(data)):
    for i in range(Giws.shape[0]):
      for j in range(Giws.shape[1]):
        Giws[i,j] = data[k][i][:,2*j+1] + 1j*data[k][i][:,2*j+2]
  return iws, Giws


## Expects (5xM) with 5 orbitals and M being number of matsubara points
def proj_sig_full_matsubara(iws, sigs):
  msig = np.zeros((sigs.shape[0], 2*sigs.shape[1])).astype(np.complex128)
  for i in range(len(sigs)):
    msig[i, sigs.shape[1]:] = sigs[i]
    msig[i, :sigs.shape[1]] = np.flip(np.conjugate(sigs[i]))
  e_iws = np.zeros(2*len(iws))
  e_iws[:len(iws)] = -np.flip(iws)
  e_iws[len(iws):] = iws 

  return e_iws, msig



def write_gxs():
  all_iws, all_sigs = get_sigs()
  all_gxs = []
  for a in range(len(all_sigs)):
    sigs = all_sigs[a]
    iws = all_iws[a]
    Gxs = np.zeros((sigs.shape[0], sigs.shape[1], 38)).astype(np.complex128)
    for i in range(sigs.shape[0]):
      for j in range(sigs.shape[1]):
        msig = np.zeros((2*sigs.shape[2], 1, 1)).astype(np.complex128)
        msig[sigs.shape[2]:,0,0] = sigs[i,j]
        msig[:sigs.shape[2],0,0] = np.flip(np.conjugate(sigs[i,j]))
        e_iws = np.zeros(2*len(iws))
        e_iws[:len(iws)] = -np.flip(iws)
        e_iws[len(iws):] = iws 

        Emax, beta = 100, np.pi/iws[0]
        d = dlr(lamb = Emax*beta, eps=1e-10)
        Gx = d.lstsq_dlr_from_matsubara(1j*e_iws, msig, beta)
        Gxs[i,j] = Gx[:,0,0]
    all_gxs.append(Gxs)   
    print(a)
  with open("all_gxs.pkl","wb") as f:
    pickle.dump(all_gxs, f)     
                

def read_gxs():
  all_iws, all_sigs = get_sigs()
  with open("old_all_gxs.pkl","rb") as f:
    all_gxs = pickle.load(f)
  return all_iws, all_gxs


def get_sinf_edc():
  all_sinfs = []
  all_edcs = []
  for fi in os.listdir("../DMFT_SIGS"):
    with open("../DMFT_SIGS/" + fi, "rb") as f:
        atoms, sigs = pickle.load(f)
    for k in range(len(sigs)):
      data = sigs[k].split("\n")
      sinf = eval(data[0].split("=")[1].strip())
      edc = eval(data[1].split("=")[1].strip())
      ao_sinf = np.zeros((4, 5)).astype(np.float64)
      ao_edc = np.zeros((4, 5)).astype(np.float64)
      for i in range(4):
        for j in range(5):
            ao_sinf[i,j] = sinf[5*i + j]
            ao_edc[i,j] = edc[5*i + j]
      all_sinfs.append(ao_sinf)
      all_edcs.append(ao_edc)

  return all_sinfs, all_edcs
    

### Keep in note that for now everything is set up for receiving one structure's green's function and self energy

# all_iws, all_sigs = get_sigs()
# iws = all_iws[0]
# Giws = all_sigs[0]
# Giw_test = np.zeros((2*Giws.shape[2], 1, 1)).astype(np.complex128)
# Giw_test[Giws.shape[2]:,0,0] = Giws[2,3]
# Giw_test[:Giws.shape[2],0,0] = np.flip(np.conjugate(Giws[2,3]))
# e_iws = np.zeros(2*len(iws))
# e_iws[:len(iws)] = -np.flip(iws)
# e_iws[len(iws):] = iws 
# Emax, beta = 10, np.pi/iws[0]
# d = dlr(lamb = Emax*beta, eps=1e-10)
# Gxs = d.lstsq_dlr_from_matsubara(1j*e_iws, Giw_test, beta)
# Giw_recon = d.eval_dlr_freq(Gxs, 1j*e_iws, beta)
# plt.plot(e_iws, Giw_test[:,0,0].imag, marker='o', c='red')
# plt.plot(e_iws, Giw_recon[:,0,0].imag, marker='+', c='blue')
# plt.show()
















