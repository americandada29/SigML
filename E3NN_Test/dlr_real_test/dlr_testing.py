from pydlr import dlr
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as SA 
from pymatgen.io.ase import AseAtomsAdaptor as AAA



def parse_sig_file(sig_text, orbital="d"):
  if orbital == "d":
    num_orbitals = 5
  elif orbital == "f":
    num_orbitals = 7
  else:
    raise Exception("Orbital specified not supported, use either d or f")

  sig_lines = sig_text.split("\n")
  s_infs = np.array([float(x) for x in sig_lines[0].split("[")[1].split("]")[0].split(",")], dtype=np.float64)
  data = np.loadtxt(sig_lines[2:])
  iws = data[:,0]
  sig_data = data[:, 1:]
  num_atoms = int(sig_data.shape[1]/10)
  adjusted_sig_data = np.zeros((num_atoms, num_orbitals, sig_data.shape[0]), dtype=np.complex128)
  for i in range(num_atoms):
    for j in range(num_orbitals):
      # adjusted_sig_data[:,i] = sig_data[:,2*i] + s_infs[i] + 1j*sig_data[:,2*i+1]
      adjusted_sig_data[i,j] = sig_data[:,2*num_orbitals*i + 2*j] + 1j*sig_data[:,2*num_orbitals*i+2*j+1]
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
  d = dlr(lamb=beta*E_max, eps=1e-10, python_impl=True)
  w_x = d.get_dlr_frequencies()
  S_dlr = d.lstsq_dlr_from_matsubara(iws_test, S_test, beta=beta)
  # S_recon = d.eval_dlr_freq(S_dlr, iws_test, beta, xi=-1)
  return S_dlr

def get_group_dlr_coeffs(iws, S, E_max=50.0):
  S_input = np.zeros((S.shape[1], S.shape[0], S.shape[0]), dtype=np.complex128)
  for i in range(S.shape[0]):
    S_input[:,i,i] = S[i]
  beta = np.pi/iws[0]
  iws_test = np.hstack((np.flip(-1j*iws), 1j*iws))
  S_test = np.vstack((np.flip(np.conj(S_input), axis=0), S_input))
  d = dlr(lamb=beta*E_max, eps=1e-10, python_impl=True)
  w_x = d.get_dlr_frequencies()
  S_dlr = d.lstsq_dlr_from_matsubara(iws_test, S_test, beta=beta)
  S_out = np.zeros((S_dlr.shape[1], S_dlr.shape[0]), dtype=np.complex128)
  for i in range(S_dlr.shape[1]):
    S_out[i] = S_dlr[:,i,i]
  

  ### Testing to see if DLR coefficients can accurately describe self energy ###
  # S_recon = d.eval_dlr_freq(S_dlr, iws_test, beta, xi=-1)
  # compare_sigs(iws_test, S_test[:, 0,0], S_recon[:,0,0])
  # exit()

  return S_out


# def write_all_dlr_coeffs(sigs, E_max=130.0, fname="dlr_coeffs.pkl"):
#   sig_dlrs = []
#   for sig in tqdm(sigs, desc="Creating DLR coefficients..."):
#     iws, asig = parse_sig_file(sig)
#     beta = np.pi/iws[0]
#     d = dlr(lamb=beta*E_max, eps=1e-10)
#     w_x = d.get_dlr_frequencies()
#     num_atoms = asig.shape[0]
#     num_orbitals = asig.shape[1]
#     sig_dlr = np.zeros((num_atoms, num_orbitals, len(w_x)), dtype=np.complex128)
#     for a in range(num_atoms):
#       sig_dlr[a] = get_group_dlr_coeffs(iws, asig[a], E_max=E_max)      
#     sig_dlrs.append(sig_dlr)
#   # for s in sig_dlrs:
#   #   print(s.shape)
#   # exit()
#   with open(fname,"wb") as f:
#     pickle.dump(sig_dlrs, f)

def write_all_dlr_coeffs(sigs, E_max=130.0, fname="dlr_coeffs.pkl"):
  sig_dlrs = []
  for sig in tqdm(sigs, desc="Creating DLR coefficients..."):
    iws, asig = parse_sig_file(sig)
    beta = np.pi/iws[0]

    asig = asig/13.6

    d = dlr(lamb=beta*E_max, eps=1e-10)
    w_x = d.get_dlr_frequencies()
    num_atoms = asig.shape[0]
    num_orbitals = asig.shape[1]
    sig_dlr = np.zeros((num_atoms, num_orbitals, len(w_x)), dtype=np.complex128)
    for a in range(num_atoms):
      sig_dlr[a] = get_group_dlr_coeffs(iws, asig[a], E_max=E_max)      
    print(sig_dlr)
    exit()
    sig_dlrs.append(sig_dlr)
  # for s in sig_dlrs:
  #   print(s.shape)
  # exit()
  with open(fname,"wb") as f:
    pickle.dump(sig_dlrs, f)

def read_dlr_coeffs(fname):
  with open(fname,"rb") as f:
    sig_dlrs = pickle.load(f)
  return np.array(sig_dlrs)


def get_noneq_atom_inds(atom):
  struct = AAA().get_structure(atom)
  spa = SA(struct, symprec=1e-5)
  sym_struct = spa.get_symmetrized_structure()
  unique_inds = [x[0] for x in sym_struct.equivalent_indices]
  return unique_inds




def create_soap_vecs(atoms, target_chem_syms, r_cut=4.0, n_max=2, l_max=2):
  all_vecs = []
  for atom in atoms:
    noneq_inds = get_noneq_atom_inds(atom)
    syms = atom.get_chemical_symbols()
    soap_centers = []
    for n in noneq_inds:
      if syms[n] in target_chem_syms:
        soap_centers.append(n)
    soap = SOAP(species=atom.get_chemical_symbols(), periodic=True, r_cut=r_cut, n_max=n_max, l_max=l_max, sparse=False)
    vecs = soap.create(atom, centers=soap_centers)
    all_vecs.append(vecs)
  return all_vecs




# files = ["dmft_2_as.pkl","dmft_as.pkl"]
# atoms = []
# sigs = []
# for f in files:
#   with open(f, "rb") as g:
#     tatoms, tsigs = pickle.load(g)
#   atoms.extend(tatoms)
#   sigs.extend(tsigs)

# iws, sig_data = parse_sig_file(sigs[0])

# print(sigs[0])
# print(sig_data[:,1,1])
# exit()

# get_dlr_coeffs(iws, sig_data[:,0])
# create_soap_vecs(atoms[24])
