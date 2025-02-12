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


def get_efs():
  all_atoms, all_efs = [], []
  #for fi in os.listdir("Data/"):
  for fi in ["rattle_a_e.pkl"]:
    with open("Data/" + fi, "rb") as f:
      atoms, efs = pickle.load(f)
    all_atoms.extend(atoms)
    all_efs.extend([float(efs[i]) for i in range(len(efs))])
  return all_atoms, all_efs
    




def get_rcovdata():
    dat = \
    [[ 0  , "X" , 1.0],
    [ 1  , "H"  , 0.37],
    [ 2  , "He" , 0.32],
    [ 3  , "Li" , 1.34],
    [ 4  , "Be" , 0.90],
    [ 5  , "B"  , 0.82],
    [ 6  , "C"  , 0.77],
    [ 7  , "N"  , 0.75],
    [ 8  , "O"  , 0.73],
    [ 9  , "F"  , 0.71],
    [ 10 , "Ne" , 0.69],
    [ 11 , "Na" , 1.54],
    [ 12 , "Mg" , 1.30],
    [ 13 , "Al" , 1.18],
    [ 14 , "Si" , 1.11],
    [ 15 , "P"  , 1.06],
    [ 16 , "S"  , 1.02],
    [ 17 , "Cl" , 0.99],
    [ 18 , "Ar" , 0.97],
    [ 19 , "K"  , 1.96],
    [ 20 , "Ca" , 1.74],
    [ 21 , "Sc" , 1.44],
    [ 22 , "Ti" , 1.36],
    [ 23 , "V"  , 1.25],
    [ 24 , "Cr" , 1.27],
    [ 25 , "Mn" , 1.39],
    [ 26 , "Fe" , 1.25],
    [ 27 , "Co" , 1.26],
    [ 28 , "Ni" , 1.21],
    [ 29 , "Cu" , 1.38],
    [ 30 , "Zn" , 1.31],
    [ 31 , "Ga" , 1.26],
    [ 32 , "Ge" , 1.22],
    [ 33 , "As" , 1.19],
    [ 34 , "Se" , 1.16],
    [ 35 , "Br" , 1.14],
    [ 36 , "Kr" , 1.10],
    [ 37 , "Rb" , 2.11],
    [ 38 , "Sr" , 1.92],
    [ 39 , "Y"  , 1.62],
    [ 40 , "Zr" , 1.48],
    [ 41 , "Nb" , 1.37],
    [ 42 , "Mo" , 1.45],
    [ 43 , "Tc" , 1.56],
    [ 44 , "Ru" , 1.26],
    [ 45 , "Rh" , 1.35],
    [ 46 , "Pd" , 1.31],
    [ 47 , "Ag" , 1.53],
    [ 48 , "Cd" , 1.48],
    [ 49 , "In" , 1.44],
    [ 50 , "Sn" , 1.41],
    [ 51 , "Sb" , 1.38],
    [ 52 , "Te" , 1.35],
    [ 53 , "I"  , 1.33],
    [ 54 , "Xe" , 1.30],
    [ 55 , "Cs" , 2.25],
    [ 56 , "Ba" , 1.98],
    [ 57 , "La" , 1.80],
    [ 58 , "Ce" , 1.63],
    [ 59 , "Pr" , 1.76],
    [ 60 , "Nd" , 1.74],
    [ 61 , "Pm" , 1.73],
    [ 62 , "Sm" , 1.72],
    [ 63 , "Eu" , 1.68],
    [ 64 , "Gd" , 1.69],
    [ 56 , "Tb" , 1.68],
    [ 66 , "Dy" , 1.67],
    [ 67 , "Ho" , 1.66],
    [ 68 , "Er" , 1.65],
    [ 69 , "Tm" , 1.64],
    [ 70 , "Yb" , 1.70],
    [ 71 , "Lu" , 1.60],
    [ 72 , "Hf" , 1.50],
    [ 73 , "Ta" , 1.38],
    [ 74 , "W"  , 1.46],
    [ 75 , "Re" , 1.59],
    [ 76 , "Os" , 1.28],
    [ 77 , "Ir" , 1.37],
    [ 78 , "Pt" , 1.28],
    [ 79 , "Au" , 1.44],
    [ 80 , "Hg" , 1.49],
    [ 81 , "Tl" , 1.48],
    [ 82 , "Pb" , 1.47],
    [ 83 , "Bi" , 1.46],
    [ 84 , "Po" , 1.45],
    [ 85 , "At" , 1.47],
    [ 86 , "Rn" , 1.42],
    [ 87 , "Fr" , 2.23],
    [ 88 , "Ra" , 2.01],
    [ 89 , "Ac" , 1.86],
    [ 90 , "Th" , 1.75],
    [ 91 , "Pa" , 1.69],
    [ 92 , "U"  , 1.70],
    [ 93 , "Np" , 1.71],
    [ 94 , "Pu" , 1.72],
    [ 95 , "Am" , 1.66],
    [ 96 , "Cm" , 1.66],
    [ 97 , "Bk" , 1.68],
    [ 98 , "Cf" , 1.68],
    [ 99 , "Es" , 1.65],
    [ 100, "Fm" , 1.67],
    [ 101, "Md" , 1.73],
    [ 102, "No" , 1.76],
    [ 103, "Lr" , 1.61],
    [ 104, "Rf" , 1.57],
    [ 105, "Db" , 1.49],
    [ 106, "Sg" , 1.43],
    [ 107, "Bh" , 1.41],
    [ 108, "Hs" , 1.34],
    [ 109, "Mt" , 1.29],
    [ 110, "Ds" , 1.28],
    [ 111, "Rg" , 1.21],
    [ 112, "Cn" , 1.22]]

    return dat







