import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, eval_legendre
from scipy.integrate import simpson, romb
from scipy.optimize import curve_fit


def func_tail(iws, c1):
  return c1/(iws) 


def xtau(tau, beta):
  return 2*tau/beta - 1

def Tnldag(n, l):
  return (-1)**n*(-1j)**(l+1)*np.sqrt(2*l+1)*spherical_jn(l, (2*n+1)*np.pi/2)

def Tnl(n, l):
  return (-1)**n*(1j)**(l+1)*np.sqrt(2*l+1)*spherical_jn(l, (2*n+1)*np.pi/2)

def calc_high_freq_moments(Gl, beta):
  lmax = len(Gl)
  c1 = 0
  c2 = 0
  c3 = 0
  for l in range(lmax):
    prefact = 2*np.sqrt(2*l+1)/beta
    if l%2 == 1:
      c2 += prefact*Gl[l]*l*(l+1)/(beta)
    elif l%2 == 0:
      c1 += -prefact*Gl[l] 
      c3 += -prefact*Gl[l]*(l+2)*(l+1)*l*(l-1)/(2*beta**2)
  return c1, c2, c3

def convert_to_greens(iws, sig):
  N = len(iws)
  Giw = np.zeros(N).astype(np.complex128)
  for i in range(N):
    Giw[i] = 1/(1j*iws[i] - sig[i])
  return Giw

def gen_gtau(taus, Gl):
  beta = taus[-1]
  Gtau = np.zeros(len(taus)).astype(np.complex128)
  for t in range(len(taus)):
    for l in range(len(Gl)):
      Gtau[t] += np.sqrt(2*l+1)/beta*eval_legendre(l,xtau(taus[t], beta))*Gl[l]
  return Gtau

def glfromgiw(iws, Giw, lmax = 50):
  Gl = np.zeros(lmax).astype(np.complex128)
  for l in range(lmax):
    for n in range(len(iws)):
      # Gl[l] += Tnldag(n, l)*Giw[n]
      # Gl[l] += Tnldag(n, l)*Giw[n] + Tnldag(-n, l)*np.conjugate(Giw[n])
      Gl[l] += Tnldag(n, l)*Giw[n] + Tnl(n, l)*np.conjugate(Giw[n])

  return Gl

def glfromgtau(taus, Gtau, lmax=100):
  beta = taus[-1]
  Gl = np.zeros(lmax).astype(np.complex128)
  for l in range(lmax):
    Gl[l] = np.sqrt(2*l+1)*simpson(Gtau*eval_legendre(l, xtau(taus, beta)), taus)
  return Gl

def gtaufromgl(taus, Gl):
    beta = taus[-1]
    Gtau = np.zeros(len(taus)).astype(np.complex128)
    for t in range(len(taus)):
     for l in range(len(Gl)):
        Gtau[t] += (np.sqrt(2*l+1)/beta)*eval_legendre(l, xtau(taus[t], beta))*Gl[l]
    return Gtau

def gtaufromgiw(taus, iws, Giw):
  Gtau = np.zeros(len(taus)).astype(np.complex128)
  beta = taus[-1]
  for t in range(len(taus)):
    result = np.exp(-1j*iws*taus[t])*Giw
    Gtau[t] = np.sum(2*result.real)
    # for n in range(len(iws)):
    #   Gtau[t] += np.exp(-1j*iws[n]*taus[t])*Giw[n] + np.conjugate(np.exp(-1j*iws[n]*taus[t])*Giw[n])
    #for n in range(len(iws)):
    #  Gtau[t] += np.exp(1j*iws[n]*taus[t])*np.conjugate(Giw[n])
  return Gtau/beta

def giwfromgtau(iws, taus, Gtau):
    beta = taus[-1]
    Giw = np.zeros(len(iws)).astype(np.complex128)
    for n in range(len(iws)):
        Giw[n] = simpson(np.exp(1j*iws[n]*taus)*Gtau, taus)
    return Giw

def giwfromgl(iws, Gl):
  beta = np.pi/iws[0]
  Giw = np.zeros(len(iws)).astype(np.complex128)
  for n in range(len(iws)):
    for l in range(len(Gl)):
      Giw[n] += Tnl(n,l)*Gl[l]
  return Giw

def siwfromgiw(iws, Giw):
  Siw = np.zeros(len(iws)).astype(np.complex128)
  Siw = 1j*iws - 1/Giw
  return Siw

def giwfromsiw(iws, Siw):
  Giw = np.zeros(len(iws)).astype(np.complex128)
  Giw = 1/(1j*iws - Siw)
  return Giw

def fit_hfm(iws, sig, ind_start=20, ind_end=30):
  fit_matrix = np.zeros((sig.shape[0], 1, sig.shape[-1]))
  # halfway = len(iws)//2
  for i in range(sig.shape[0]):
    for j in range(sig.shape[-1]):
      popt, _ = curve_fit(func_tail, iws[ind_start:ind_end], sig[i, ind_start:ind_end, j].imag)
      fit_matrix[i, :, j] = popt
  return fit_matrix

def fullatom_gl_from_giw(iws, giw, lmax=50):
  Gl = np.zeros((giw.shape[0], lmax, giw.shape[-1]), dtype=np.complex128)
  for i in range(giw.shape[0]):
    for j in range(giw.shape[-1]):
      Gl[i, :, j] = glfromgiw(iws, giw[i, :, j], lmax)
  return np.real(Gl)

def fullatom_giw_from_gl(iws, Gl, fit_tail=False, fit_tail_cutoff=None, fit_tail_start=None, fit_tail_end=None):
  Giw = np.zeros((Gl.shape[0], len(iws), Gl.shape[-1]), dtype=np.complex128)
  for i in range(Gl.shape[0]):
    for j in range(Gl.shape[-1]):
      Giw[i, :, j] = giwfromgl(iws, Gl[i, :, j])
  if fit_tail:
    hind = fit_tail_cutoff 
    hfm_matrix = fit_hfm(iws, Giw, fit_tail_start, fit_tail_end)
    for i in range(Gl.shape[0]):
      for j in range(Gl.shape[-1]):
        halfway = iws[hind:]
        gf_tail = hfm_matrix[i, 0, j]/(halfway) 
        gf_tail_diff = np.diff(gf_tail)
        newgftail = [Giw[i, hind, j].imag]
        for k in range(0,len(gf_tail_diff)):
          newgftail.append(newgftail[-1] + gf_tail_diff[k])
        newgftail = np.array(newgftail)
        Giw[i, :, j] = np.concatenate((Giw[i, :hind, j], Giw[i, hind:, j].real + 1j*newgftail))
    return Giw
  return Giw

def fullatom_high_freq_moments(iws, Gl):
  beta = np.pi/iws[0]
  hfm_matrix = np.zeros((Gl.shape[0], Gl.shape[-1], 3), dtype=np.complex128)
  for i in range(Gl.shape[0]):
    for j in range(Gl.shape[-1]):
      hfm_matrix[i, j, :] = calc_high_freq_moments(Gl[i, :, j], beta)
  return hfm_matrix


