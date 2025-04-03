import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
import e3nn
from e3nn import o3
from typing import Dict, Union
from BasicNetwork import Network

# crystal structure data
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms

# data pre-processing and visualization
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle 

import os

from dlr_testing import parse_sig_file
from BasicNetwork import PeriodicNetwork, train, evaluate, visualize_layers, train_test_split, PeakEmphasisSmoothLoss, CrystalSelfEnergyNetwork


### Builds self energy matrix in vector form (atoms, N_matsubara*5)
def build_sig_vector(sig_inp):
    sig = np.zeros((sig_inp.shape[0], sig_inp.shape[1]*sig_inp.shape[2])).astype(np.complex128)
    for i in range(len(sig_inp)):
        for j in range(len(sig_inp[i])):
            sig[i, j*sig_inp.shape[2]:(j+1)*sig_inp.shape[2]] = sig_inp[i,j]
    return sig

### Builds self energy matrix in fortran form (atoms, N_matsubara, 5)
def build_sig_matrix_fortran(sig_inp):
    sig = np.zeros((sig_inp.shape[0], sig_inp.shape[2], sig_inp.shape[1])).astype(np.complex128)
    for i in range(len(sig_inp)):
        for j in range(len(sig_inp[i])):
            sig[i, :, j] = sig_inp[i,j]
    return sig


def build_data(atom, ef, type_encoding, type_onehot, am_onehot, r_max=5.0):
    symbols = list(atom.get_chemical_symbols())
    positions = torch.from_numpy(atom.positions).type(torch.float32)
    lattice = torch.from_numpy(atom.get_cell()[:]).unsqueeze(0).type(torch.float32)
    ef = torch.tensor(ef).unsqueeze(0).type(torch.float32)
    
    



    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=atom, cutoff=r_max, self_interaction=True)
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]

    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=torch.float32), lattice[edge_batch])).type(torch.float32)
    
    edge_len = edge_vec.norm(dim=1) 
    x = am_onehot[[type_encoding[s] for s in symbols]].type(torch.float32)
    z = type_onehot[[type_encoding[s] for s in symbols]]
    edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0)
    edge_shift = torch.tensor(edge_shift, dtype=torch.int16)
    
    data = tg.data.Data(pos = positions, lattice = lattice, symbol = symbols, 
                        x = x, z = z, edge_index=edge_index, edge_shift = edge_shift, 
                        edge_vec=edge_vec, edge_len=edge_len, ef = ef)
    return data


def get_average_neighbor_count(all_data):
    neighbor_count = []
    for adata in all_data:
        N = adata.pos.shape[0]
        for i in range(N):
            neighbor_count.append(len((adata.edge_index[0] == i).nonzero()))
    return np.array(neighbor_count)


source_dir = "../ATOMS_EFS_DATA/"
atoms = []
efs = []
for pf in os.listdir(source_dir):
    with open(source_dir + pf, "rb") as f:
        tatoms, tefs = pickle.load(f)
    atoms.extend(tatoms)
    efs.extend(tefs)



torch.set_default_dtype(torch.float32)
radial_cutoff = 3.0

type_encoding = {}
species_am = []
for Z in range(1, 119):
    species = Atom(Z)
    type_encoding[species.symbol] = Z-1
    species_am.append(species.mass)
type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(species_am))

all_data = []
for i in range(len(atoms)):
    adata = build_data(atoms[i], efs[i], type_encoding, type_onehot, am_onehot, radial_cutoff)
    all_data.append(adata)
neighbor_count = get_average_neighbor_count(all_data)


## Dividing out dim by 5 to account for 5D representation of l=2 spherical harmonics 
out_dim = 1
em_dim = 16


model = PeriodicNetwork(in_dim = 118,
                        em_dim= em_dim,
                        irreps_in = str(em_dim) + "x0e",
                        irreps_out = str(out_dim) + "x0e",
                        irreps_node_attr = str(em_dim) + "x0e",
                        layers=2,
                        mul=16,
                        lmax=2,
                        max_radius=radial_cutoff,
                        num_neighbors=neighbor_count.mean(),
                        reduce_output=True)

# (self, in_dim, em_dim, out_dim, radial_cutoff, neighbor_count, orbital_count, layers, mul, lmax): 


### Testing various optimizers ###
opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
# opt = torch.optim.SGD(model.parameters(), lr=0.000001)

### Testing various schedulers ###
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.25)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, threshold=1e-6)


### Testing various loss functions ###
# loss_fn = torch.nn.MSELoss(reduction="sum")
# loss_fn = PeakEmphasisSmoothLoss(out_dim, 5, int(0.2*out_dim))
loss_fn = torch.nn.SmoothL1Loss(reduction="sum")


train_data, test_data = train_test_split(all_data, train_percent=0.9, seed=34533)

save_path = "ef_model.pth"

train(model, opt, train_data, loss_fn, scheduler, save_path= save_path, max_iter=20)

model.load_state_dict(torch.load(save_path))

evaluate(model, test_data)




