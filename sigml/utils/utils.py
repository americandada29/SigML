import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
import e3nn
from e3nn import o3
from typing import Dict, Union
from tqdm import tqdm
from nequip.ase.nequip_calculator import nequip_calculator
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from sigml.utils.leg_lib import fullatom_gl_from_giw, fullatom_giw_from_gl

# crystal structure data
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as SPA

# data pre-processing and visualization
import numpy as np
import matplotlib.pyplot as plt
import copy
import os


"""
utils.py

This file contains the utility functions for processing data, training and evaluating models, and visualizing results.
- parse_sig_file: Parse the sig file text from EDMFTF and return the matsubara frequencies, the self energy data, and the self energy data at infinite frequency
- get_average_neighbor_count: Get the average number of neighbors for a given dataset
- evaluate_sinf: Evaluate the $\Sigma_{\infty}$ model on a dataset
- train_sinf: Train the $\Sigma_{\infty}$ model on a dataset
- train_full_sig: Train the $\Sigma(i\omega)$ model on a dataset
- evaluate_full_sig: Evaluate the $\Sigma(i\omega)$ model on a dataset
- evaluate_full_sig_legendre: Evaluate the $\Sigma(i\omega)$ model on a dataset using the Legendre expansion
- train_ef: Train the $E_f$ model on a dataset
- evaluate_ef: Evaluate the $E_f$ model on a dataset
- build_data: Build a dataset from a list of atoms, self energy text files, and fermi energies
- get_sig_file_text: Get the sig file text from a list of matsubara frequencies, self energy data, and self energy data at infinite frequency
"""


def collate_to_list(batch_list):
    return batch_list


def parse_sig_file(sig_text, orbital="d"):
    r"""
    Parses the sig file text from EDMFTF and returns the matsubara frequencies, the self energy data, and the self energy data at infinite frequency

    Parameters 
    ----------
    sig_text: str
        The text from the sig file
    orbital: str, default = "d"
        The orbital to parse, either "d" or "f"

    Returns
    -------
    iws: np.ndarray
        The matsubara frequencies (upper half of complex plane)
    sig_data: np.ndarray
        The self energy data 
    sinfs: np.ndarray
        The self energy data at infinite frequency
    """
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
    num_atoms = int(sig_data.shape[1]/(2*num_orbitals))
    adjusted_sig_data = np.zeros((num_atoms, num_orbitals, sig_data.shape[0]), dtype=np.complex128)
    adjusted_sinfs = np.zeros((num_atoms, num_orbitals))
    for i in range(num_atoms):
        adjusted_sinfs[i] = s_infs[i*num_orbitals:(i+1)*num_orbitals]
        for j in range(num_orbitals):
            adjusted_sig_data[i,j] = sig_data[:,2*num_orbitals*i + 2*j] + 1j*sig_data[:,2*num_orbitals*i+2*j+1]
    return iws, adjusted_sig_data, adjusted_sinfs


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


def cast_sig_over_symmetric_atoms(atom, sig, cor_atoms):
    spa = SPA(AAA.get_structure(atom), symprec=1e-5)
    sym_struct = spa.get_symmetrized_structure()
    eq_inds = sym_struct.equivalent_indices
    syms = atom.get_chemical_symbols()
    full_sig = np.zeros((len(atom), sig.shape[1], sig.shape[2])).astype(np.complex128)
    count = 0
    cor_atom_inds = []
    for i in range(len(eq_inds)):
        if syms[eq_inds[i][0]] in cor_atoms:
            for eqi in eq_inds[i]:
                full_sig[eqi] = sig[count]
                cor_atom_inds.append(eqi)
            count += 1

    return full_sig, cor_atom_inds, eq_inds


def get_correlated_atoms_inds(atom, cor_atoms):
    spa = SPA(AAA.get_structure(atom), symprec=1e-5)
    sym_struct = spa.get_symmetrized_structure()
    eq_inds = sym_struct.equivalent_indices
    syms = atom.get_chemical_symbols()
    cor_atom_inds = []
    for i in range(len(eq_inds)):
        if syms[eq_inds[i][0]] in cor_atoms:
            for eqi in eq_inds[i]:
                cor_atom_inds.append(eqi)

    return cor_atom_inds, eq_inds


def build_datapoint(atom, type_encoding, type_onehot, am_onehot, sig_text=None, ef=None, r_max=4.0, device="cpu"):
    symbols = list(atom.get_chemical_symbols())
    cor_atoms = list(set([symbols[i] for i in range(len(atom.get_atomic_numbers())) if atom.get_atomic_numbers()[i] > 20]))
    positions = torch.as_tensor(atom.positions,  dtype=torch.float32, device=device)
    lattice   = torch.as_tensor(atom.get_cell()[:],dtype=torch.float32, device=device).unsqueeze(0)
    
    sig = None
    sig_org = None
    iws = None
    sinf = None
    cor_atom_inds = None
    eq_inds = None 
    if sig_text is not None:
        iws, sig, sinf = parse_sig_file(sig_text)
        sig = build_sig_matrix_fortran(sig)
        cor_atom_inds, eq_inds = get_correlated_atoms_inds(atom, cor_atoms)
        sig_org = copy.deepcopy(sig)
        sig = fullatom_gl_from_giw(iws, sig, lmax=30)
        iws = torch.as_tensor(iws, dtype=torch.float32, device=device).unsqueeze(0)
        # sig, cor_atom_inds, eq_inds = cast_sig_over_symmetric_atoms(atom, sig, cor_atoms)
        cor_atom_inds = torch.as_tensor(cor_atom_inds, dtype=torch.int16, device=device)
        eq_inds = [torch.as_tensor(sub, dtype=torch.int16, device=device) for sub in eq_inds]
        sig = torch.as_tensor(sig, dtype=torch.float32, device=device).unsqueeze(0)
        # sig = torch.stack((sig.real, sig.imag), dim=-1).type(torch.float32)
        sinf = torch.as_tensor(sinf, dtype=torch.float32, device=device)
    else:
        cor_atom_inds, eq_inds = get_correlated_atoms_inds(atom, cor_atoms)
        cor_atom_inds = torch.as_tensor(cor_atom_inds, dtype=torch.int16, device=device)
        eq_inds = [torch.as_tensor(sub, dtype=torch.int16, device=device) for sub in eq_inds]
    
    ef_torch = None 
    if ef is not None:
        # ef_torch = torch.tensor(ef).unsqueeze(0).type(torch.float32)
        ef_torch = torch.tensor(ef, dtype=torch.float32, device=device).unsqueeze(0)



    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=atom,
                                                   cutoff=r_max, self_interaction=True)

    edge_src  = torch.from_numpy(edge_src ).long().to(device)
    edge_dst  = torch.from_numpy(edge_dst ).long().to(device)
    edge_shift= torch.from_numpy(edge_shift).to(device, torch.int16) 

    edge_batch= positions.new_zeros(positions.size(0), dtype=torch.long)[edge_src]

    edge_vec  = (positions[edge_dst] - positions[edge_src] +
                 torch.einsum('ni,nij->nj', edge_shift.to(torch.float32), lattice[edge_batch])
                ).to(torch.float32)

    edge_len  = edge_vec.norm(dim=1)

    x = am_onehot[[type_encoding[s] for s in symbols]].to(torch.float32).to(device)
    z = type_onehot[[type_encoding[s] for s in symbols]].to(device)

    edge_index = torch.stack([edge_src, edge_dst], dim=0)            
    
    data = tg.data.Data(pos = positions, lattice = lattice, symbol = symbols, 
                        x = x, z = z, edge_index=edge_index, edge_shift = edge_shift, 
                        edge_vec=edge_vec, edge_len=edge_len, sig=sig, iws=iws, sinf=sinf, ef=ef_torch, 
                        cor_atom_inds = cor_atom_inds, eq_inds = eq_inds, sig_org = sig_org)
    return data


def get_average_neighbor_count(all_data):
    r"""
    Gets the average number of neighbors for a given dataset

    Parameters 
    ----------
      all_data: list
        A list of tg.data.Data objects
    Returns
    ----------
      avg_neighbor_count: float
        The average number of neighbors for the dataset
    """
    neighbor_count = []
    for adata in all_data:
        N = adata.pos.shape[0]
        for i in range(N):
            neighbor_count.append(len((adata.edge_index[0] == i).nonzero()))
    return np.array(neighbor_count).mean()


def visualize_layers(model):
    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['gate', 'tp', 'lin2', 'gate']))
    try: layers = model.mp.layers
    except: layers = model.layers

    num_layers = len(layers)
    num_ops = max([len([k for k in list(layers[i].first._modules.keys()) if k not in ['fc', 'alpha']])
                   for i in range(num_layers-1)])

    fig, ax = plt.subplots(num_layers, num_ops, figsize=(14,3.5*num_layers))
    for i in range(num_layers - 1):
        ops = layers[i].first._modules.copy()
        ops.pop('fc', None); ops.pop('alpha', None)
        for j, (k, v) in enumerate(ops.items()):
            ax[i,j].set_title(k, fontsize=textsize)
            v.cpu().visualize(ax=ax[i,j])
            ax[i,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[i,j].transAxes)

    layer_dst = dict(zip(['sc', 'lin1', 'tp', 'lin2'], ['output', 'tp', 'lin2', 'output']))
    ops = layers[-1]._modules.copy()
    ops.pop('fc', None); ops.pop('alpha', None)
    for j, (k, v) in enumerate(ops.items()):
        ax[-1,j].set_title(k, fontsize=textsize)
        v.cpu().visualize(ax=ax[-1,j])
        ax[-1,j].text(0.7,-0.15,'--> to ' + layer_dst[k], fontsize=textsize-2, transform=ax[-1,j].transAxes)

    fig.subplots_adjust(wspace=0.3, hspace=0.5)


def train_test_split(dataset, train_percent=0.9, seed=None):
    r"""
    Splits a dataset into training and test sets

    Parameters 
    ----------
      dataset: list
        A list of tg.data.Data objects
      train_percent: float, default = 0.9
        The percentage of the dataset to use for training
      seed: int, default = None
        The seed to use for the random number generator

    Returns
    ----------
      train_data: list
        The training data
      test_data: list
        The test data
    """
    rng = None
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()
    N = int(train_percent*len(dataset))
    inds = np.arange(0, len(dataset), 1)
    rng.shuffle(inds)
    train_data = []
    test_data = []
    for i in range(len(inds)):
        if i < N:
            train_data.append(dataset[inds[i]])
        else:
            test_data.append(dataset[inds[i]])
    return train_data, test_data


def evaluate_sinf(model, dataset_org, display=True, img_save_dir = None, device="cpu"):
    r"""
    Evaluate the :math:`\Sigma_{\infty}` of model on a dataset 

    Parameters 
    ----------
      model: sigml.models.Sinf_Model.Sinf_Model
        The :math:`\Sigma_{\infty}` model to evaluate 
      dataset_org: list of tg.data.Data objects
        The dataset to evaluate the :math:`\Sigma_{\infty}` on
      display: bool, default = True
        Whether or not to display the plot. If false, img_save_dir must be provided
      img_save_dir: str, default = None
        The directory to save the plot if not displayed 
      device: str, default = "cpu"
        The device to evaluate the model on
    
    Returns
    -------
      None
    """
    dataset = copy.deepcopy(dataset_org)
    model.eval()
    model.to(device)

    n_atoms = np.amax([dataset[i].sinf.shape[0] for i in range(len(dataset))])    
    n_orbitals = dataset[0].sinf.shape[1]
    fig, axs = plt.subplots(n_atoms)
    # preds = np.zeros((len(dataset), n_atoms, n_orbitals))
    # acts = np.zeros((len(dataset), n_atoms, n_orbitals)) 
    preds = []
    acts = []
    for i in range(len(dataset)):
        d = dataset[i].to(device)
        # preds[i] = model(d).cpu().detach().numpy()
        # acts[i] = dataset[i].sinf.cpu().detach().numpy()
        preds.append(model(d).cpu().detach().numpy())
        acts.append(dataset[i].sinf.cpu().detach().numpy())
    
    colors = ["red", "green", "blue", "orange", "purple"]
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            axs[j].scatter(preds[i][j], acts[i][j], c=colors)

    amin = 100000
    amax = -100000
    for a in acts:
        amin = min(amin, np.amin(a))
        amax = max(amax, np.amax(a))
    print(amin, amax)

    
    
    for i in range(n_atoms):
        x = np.linspace(amin, amax, 1000)
        axs[i].set_xlim(amin-0.5, amax+0.5)
        axs[i].set_ylim(amin-0.5, amax+0.5)
        axs[i].plot(x, x, linestyle="--", c="black")
    if display:
        plt.show()
    elif not(display) and img_save_dir is not None:
        plt.savefig(f'{img_save_dir}/sinf_accuracy.pdf')


def train_sinf(model, optimizer, dataset, loss_fn, scheduler, save_path = None, max_iter=101, val_percent = 0.1, device="cpu", batch_size=1):
    r"""
    Train the :math:`\Sigma_{\infty}` model on a dataset 

    Parameters 
    ----------
      model: sigml.models.Sinf_Model.Sinf_Model
        The :math:`\Sigma_{\infty}` model to evaluate 
      optimizer: torch.optim.Optimizer
        The optimizer to use for training
      dataset: list of tg.data.Data objects
        The dataset to train the :math:`\Sigma_{\infty}` on
      loss_fn: torch.nn.Module
        The loss function to use for training
      scheduler: torch.optim.lr_scheduler
        The scheduler to use for training
      save_path: str, default = None
        The path to save the model. If None, the model will not be saved in a directory
      max_iter: int, default = 101
        The maximum number of iterations to train for
      val_percent: float, default = 0.1
        The percentage of the dataset to use for validation, expressed as a fraction of the total dataset
      device: str, default = "cpu"
        The device to train the model on
    
    Returns
    -------
      None
    """
    model.to(device)

    train_data, val_data = train_test_split(dataset, train_percent= 1-val_percent, seed=53234)

    for step in range(max_iter):

        ## Training 
        model.train()
        dataloader = copy.deepcopy(train_data)
        loss_cumulative = 0.0
        for d in tqdm(dataloader):
            d.to(device)
            output = model(d)
            # if batch_size == 1:
            #     output = output.unsqueeze(0)
            loss = loss_fn(output, d.sinf)
            loss_cumulative += loss.cpu().detach().item()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        ## Valdiation 
        model.eval()
        val_dataloader = copy.deepcopy(val_data)
        loss_val = 0
        for vi, v in enumerate(val_dataloader):
            v.to(device)
            val_out = model(v)
            # if batch_size == 1:
            #     val_out = val_out.unsqueeze(0)
            if vi == 0:
                print(val_out)
                print(v.sinf)
            loss = loss_fn(val_out, v.sinf)
            loss_val += loss.cpu().detach().item()

        print("Epoch", str(step),"Train Loss =", loss_cumulative/len(dataloader), "Val Loss =", loss_val/len(val_dataloader))
        # scheduler.step(loss_val)
        scheduler.step()
    if save_path is not None:
        print("Saving model to", save_path)
        torch.save(model.state_dict(), save_path)


def train_full_sig(model, optimizer, dataset, loss_fn, scheduler, save_path = None, max_iter=101, val_percent = 0.1, device="cpu", batch_size=1):
    r"""
    Train the :math:`\Sigma(i\omega_n)` model on a dataset 

    Parameters 
    ----------
      model: sigml.models.Sig_iws_Model.Sig_iws_Model
        The :math:`\Sigma(i\omega_n)` model to evaluate 
      optimizer: torch.optim.Optimizer
        The optimizer to use for training
      dataset: list of tg.data.Data objects
        The dataset to train the :math:`\Sigma(i\omega_n)` on
      loss_fn: torch.nn.Module
        The loss function to use for training
      scheduler: torch.optim.lr_scheduler
        The scheduler to use for training
      save_path: str, default = None
        The path to save the model. If None, the model will not be saved in a directory
      max_iter: int, default = 101
        The maximum number of iterations to train for
      val_percent: float, default = 0.1
        The percentage of the dataset to use for validation, expressed as a fraction of the total dataset
      device: str, default = "cpu"
        The device to train the model on
      batch_size: int, default = 1
        The batch size to use for training. Currently, only a batch size of 1 is supported
    
    Returns
    -------
      None
    """
    model.to(device)
    train_data, val_data = train_test_split(dataset, train_percent= 1-val_percent, seed=53234)
    for step in range(max_iter):
        ## Training 
        model.train()
        dataloader = copy.deepcopy(train_data)
        loss_cumulative = 0.0
        for d in tqdm(dataloader):
            d.to(device)
            output = model(d)
            if batch_size == 1:
                output = output.unsqueeze(0)
            loss = loss_fn(output, d.sig)
            loss_cumulative += loss.cpu().detach().item()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        ## Valdiation 
        model.eval()
        val_dataloader = copy.deepcopy(val_data)
        loss_val = 0
        for vi, v in enumerate(val_dataloader):
            v.to(device)
            val_out = model(v)
            if batch_size == 1:
                val_out = val_out.unsqueeze(0)
            # if vi == 0:
            #     print(val_out)
            #     print(v.sig)
            loss = loss_fn(val_out, v.sig)
            loss_val += loss.cpu().detach().item()
        print("Epoch", str(step),"Train Loss =", loss_cumulative/len(dataloader), "Val Loss =", loss_val/len(val_dataloader))
        # scheduler.step(loss_val)
        scheduler.step()
    if save_path is not None:
        print("Saving model to", save_path)
        torch.save(model.state_dict(), save_path)


def evaluate_full_sig(model, dataset_org, orbital, atom=1, display=True, img_save_dir = None):
    r"""
    Evaluate the :math:`\Sigma(i\omega_n)` of model on a dataset. CAUTION: This function is depreciated. Use evaluate_full_sig_legendre instead.

    Parameters 
    ----------
      model: sigml.models.Sig_iws_Model.Sig_iws_Model
        The :math:`\Sigma(i\omega_n)` model to evaluate 
      dataset_org: list of tg.data.Data objects
        The dataset to evaluate the :math:`\Sigma(i\omega_n)` on
      orbital: int, default = 0
        The orbital to evaluate the :math:`\Sigma(i\omega_n)` on
      atom: int, default = 0
        The atom number to evaluate the :math:`\Sigma(i\omega_n)` on. The atom number must match a unique correlated atom, symmetrically equivalent atoms are not considered
      display: bool, default = True
        Whether or not to display the plot. If false, img_save_dir must be provided
      img_save_dir: str, default = None
        The directory to save the plot if not displayed 
      device: str, default = "cpu"
        The device to evaluate the model on
    
    Returns
    -------
      None
    """
    dataset = copy.deepcopy(dataset_org)
    model.eval()
    iws = dataset[0].iws.detach().cpu().numpy()

    N1 = 3
    N2 = 3
    fig, axs = plt.subplots(N1, N2)

    colors_gt = ["lightcoral", "firebrick", "lightsalmon", "saddlebrown"]
    colors_pred = ["cornsilk", "khaki", "yellowgreen", "lawngreen"]

    for i in range(N1):
        for j in range(N2):
            output = model([dataset[N2*i+j]])[0].detach().cpu().numpy()
            N_sig_len = int(output.shape[1])
            if i == 0 and j ==0:
                axs[i,j].plot(iws[0], dataset[N2*i + j].sig[0,atom, :, orbital, 0].detach().cpu().numpy(), c="black", label="True DMFT $\Sigma$(i$\omega_n$)")
                axs[i,j].plot(iws[0], output[atom, :, orbital, 0], c="red", marker='o', markersize=2, label="Predicted DMFT Re{$\Sigma$(i$\omega_n$)}")
                axs[i,j].plot(iws[0], dataset[N2*i + j].sig[0,atom, :, orbital, 1].detach().cpu().numpy(), c="black")
                axs[i,j].plot(iws[0], output[atom, :, orbital, 1], c="blue", marker='o', markersize=2, label="Predicted DMFT Im{$\Sigma$(i$\omega_n$)}")
            else:
                axs[i,j].plot(iws[0], dataset[N2*i + j].sig[0,atom, :, orbital, 0].detach().cpu().numpy(), c="black")
                axs[i,j].plot(iws[0], output[atom, :, orbital, 0], c="red", marker='o', markersize=2)
                axs[i,j].plot(iws[0], dataset[N2*i + j].sig[0,atom, :, orbital, 1].detach().cpu().numpy().real, c="black")
                axs[i,j].plot(iws[0], output[atom, :, orbital, 1].real, c="blue", marker='o', markersize=2)
    
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("i$\omega_n$", labelpad=5, fontsize=25)
    plt.ylabel("$\Sigma$(i$\omega_n$)", labelpad=20, fontsize=25)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    # plt.tight_layout()
    if display == True:
        plt.show()
    elif display == False and img_save_dir is not None:
        print(f'Saving orbital {orbital} image to {img_save_dir}/{orbital}_full_sig.pdf')
        plt.savefig(img_save_dir + f"/{orbital}_full_sig.pdf")


def evaluate_full_sig_legendre(model, dataset_org, orbital, atom=0, display=True, img_save_dir = None):
    r"""
    Evaluate the :math:`\Sigma(i\omega_n)` of model on a dataset 

    Parameters 
    ----------
    model: sigml.models.Sig_iws_Model.Sig_iws_Model
      The :math:`\Sigma(i\omega_n)` model to evaluate 
    dataset_org: list of tg.data.Data objects
      The dataset to evaluate the :math:`\Sigma(i\omega_n)` on
    orbital: int, default = 0
      The orbital to evaluate the :math:`\Sigma(i\omega_n)` on
    atom: int, default = 0
      The atom number to evaluate the :math:`\Sigma(i\omega_n)` on. The atom number must match a unique correlated atom, symmetrically equivalent atoms are not considered
    display: bool, default = True
      Whether or not to display the plot. If false, img_save_dir must be provided
    img_save_dir: str, default = None
      The directory to save the plot if not displayed 
    device: str, default = "cpu"
      The device to evaluate the model on
    
    Returns
    -------
      None
    """
    dataset = copy.deepcopy(dataset_org)
    model.eval()
    iws = dataset[0].iws.detach().cpu().numpy()[0]

    N1 = 3
    N2 = 3
    fig, axs = plt.subplots(N1, N2)

    colors_gt = ["lightcoral", "firebrick", "lightsalmon", "saddlebrown"]
    colors_pred = ["cornsilk", "khaki", "yellowgreen", "lawngreen"]

    for i in range(N1):
        for j in range(N2):
            # output = giwfromgl(iws, model(dataset[N2*i+j])[atom, :, orbital].detach().cpu().numpy())
            # sig = giwfromgl(iws, dataset[N2*i+j].sig[0, atom, :, orbital].detach().cpu().numpy())
            sig = dataset[N2*i+j].sig_org[atom, :, orbital]

            output = model(dataset[N2*i+j])[atom:atom+1, :, orbital:orbital+1].detach().cpu().numpy()
            output = fullatom_giw_from_gl(iws, output)[0, :, 0]
            

            if i == 0 and j ==0:
                axs[i,j].plot(iws, sig.real, c="black", label="True DMFT Re{$\Sigma$(i$\omega_n$)}")
                axs[i,j].plot(iws, output.real, c="red", marker='o', markersize=2, label="Predicted DMFT Re{$\Sigma$(i$\omega_n$)}")
                axs[i,j].plot(iws, sig.imag, c="black", label="True DMFT Im{$\Sigma$(i$\omega_n$)}")
                axs[i,j].plot(iws, output.imag, c="blue", marker='o', markersize=2, label="Predicted DMFT Im{$\Sigma$(i$\omega_n$)}")
            else:
                axs[i,j].plot(iws, sig.real, c="black")
                axs[i,j].plot(iws, output.real, c="red", marker='o', markersize=2)
                axs[i,j].plot(iws, sig.imag, c="black")
                axs[i,j].plot(iws, output.imag, c="blue", marker='o', markersize=2)

    ### Code for adding common x and y labels ###  
    # fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel("i$\omega_n$", labelpad=5, fontsize=25)
    # plt.ylabel("$\Sigma$(i$\omega_n$)", labelpad=20, fontsize=25)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    # plt.tight_layout()
    if display == True:
        plt.show()
    elif display == False and img_save_dir is not None:
        print(f'Saving orbital {orbital} image to {img_save_dir}/{orbital}_full_sig.pdf')
        plt.savefig(img_save_dir + f"/{orbital}_full_sig.pdf")


def train_ef(model, optimizer, dataset, loss_fn, scheduler, save_path = None, max_iter=101, val_percent = 0.1, device="cpu", batch_size=1):
    r"""
    Train the :math:`E_f` model on a dataset 

    Parameters 
    ----------
      model: sigml.models.Ef_Model.Ef_Model
        The :math:`E_f` model to evaluate 
      optimizer: torch.optim.Optimizer
        The optimizer to use for training
      dataset: list of tg.data.Data objects
        The dataset to train the :math:`E_f` on
      loss_fn: torch.nn.Module
        The loss function to use for training
      scheduler: torch.optim.lr_scheduler
        The scheduler to use for training
      save_path: str, default = None
        The path to save the model. If None, the model will not be saved in a directory
      max_iter: int, default = 101
        The maximum number of iterations to train for
      val_percent: float, default = 0.1
        The percentage of the dataset to use for validation, expressed as a fraction of the total dataset
      device: str, default = "cpu"
        The device to train the model on
      batch_size: int, default = 1
        The batch size to use for training. Currently, only a batch size of 1 is supported

    Returns
    -------
      None
    """
    model.to(device)

    train_data, val_data = train_test_split(dataset, train_percent= 1-val_percent, seed=53234)

    for step in range(max_iter):

        ## Training 
        model.train()
        dataloader = copy.deepcopy(train_data)
        loss_cumulative = 0.0
        for d in tqdm(dataloader):
            d.to(device)
            output = model(d)
            # if batch_size == 1:
            #     output = output.unsqueeze(0)
            loss = loss_fn(output, d.ef[0])
            loss_cumulative += loss.cpu().detach().item()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        ## Valdiation 
        model.eval()
        val_dataloader = copy.deepcopy(val_data)
        loss_val = 0
        for vi, v in enumerate(val_dataloader):
            val_out = model(v)
            # if batch_size == 1:
            #     val_out = val_out.unsqueeze(0)
            # if vi == 0:
            #     print(val_out)
            #     print(v.ef)
            loss = loss_fn(val_out, v.ef[0])
            loss_val += loss.cpu().detach().item()

        print("Epoch", str(step),"Train Loss =", loss_cumulative/len(dataloader), "Val Loss =", loss_val/len(val_dataloader))
        # scheduler.step(loss_val)
        scheduler.step()
    if save_path is not None:
        print("Saving model to", save_path)
        torch.save(model.state_dict(), save_path)


def evaluate_ef(model, dataset_org):
    r"""
    Evaluate the :math:`E_f` of model on a dataset 

    Parameters 
    ----------
      model: sigml.models.Ef_Model.Ef_Model
        The :math:`E_f` model to evaluate 
      dataset_org: list of tg.data.Data objects
        The dataset to evaluate the :math:`E_f` on
    
    Returns
    -------
      None
    """
    dataset = copy.deepcopy(dataset_org)
    model.eval()
    
    preds = []
    acts = []
    mae = 0
    for i in range(len(dataset)):
        pred = model(dataset[i]).cpu().detach().item()
        act = dataset[i].ef[0].item()
        mae += np.abs(pred - act)
        preds.append(pred)
        acts.append(act)
    mae /= len(dataset)
    plt.scatter(preds, acts)
    x = np.linspace(np.amin(acts), np.amax(acts), 1000)
    plt.plot(x, x, linestyle="--", c="black")
    plt.xlim(np.amin(acts)-1 , np.amax(acts) + 1)
    plt.ylim(np.amin(acts)-1 , np.amax(acts) + 1)
    plt.text(np.amin(acts), np.amax(acts), f"MAE: {mae:.5f}", fontsize=15)
    plt.show()


def assert_sig_conditions(sig_text):
    _, sig, _ = parse_sig_file(sig_text)
    passed = True
    for a in range(sig.shape[0]):
        for o in range(sig.shape[1]):
            if np.any(sig[a, o].imag > 0):
                passed = False 
                break
    return passed


def build_data(atoms, sig_texts=None, efs=None, radial_cutoff=3.0, device="cpu"):
    r"""
    Build a dataset from a list of atoms and optional self-energy text files and fermi energies

    Parameters 
    ----------
      atoms: list of tg.data.Data objects
        The atoms to build the dataset from
      sig_texts: list of str, optional, default = None
        The self-energy text files to build the dataset from
      efs: list of float, optional, default = None
        The fermi energies to build the dataset from
      radial_cutoff: float, default = 3.0
        The radial cutoff to build the dataset from
      device: str, default = "cpu"
        The device to build the dataset on
    
    Returns
    -------
      dataset: list of tg.data.Data objects
        The dataset built from the atoms, self-energy text files, and fermi energies
    """
    type_encoding = {}
    species_am = []
    for Z in range(1, 119):
        species = Atom(Z)
        type_encoding[species.symbol] = Z-1
        species_am.append(species.mass)
    type_onehot = torch.eye(len(type_encoding), device=device)
    am_onehot = torch.diag(torch.tensor(species_am, device=device))
    all_data = []
    for i in tqdm(range(len(atoms)), desc="Building dataset..."):
        tsig_text = None 
        tef = None 
        passed = True
        if sig_texts is not None:
            tsig_text = sig_texts[i]
            passed = assert_sig_conditions(tsig_text)
        if efs is not None:
            tef = efs[i]
        if passed:
            adata = build_datapoint(atoms[i], type_encoding, type_onehot, am_onehot, tsig_text, tef, radial_cutoff, device=device)
            all_data.append(adata)
        else:
            print("Skipping atom", i, "due to not satisfying self-energy conditions")
    return all_data
    

def get_sig_file_text(iws, sig, sinf, U, J, nf):
    r"""
    Generate sig.inp file text for input into EDMFTF

    Parameters 
    ----------
      iws: np.ndarray
        Matsubara frequencies over which $\Sigma$ is defined
      sig: np.ndarray of shape (N_inequivalent_atoms, N_matsubara, N_orbitals)
        The complex valued self-energy 
      sinf: np.ndarray of shape (N_inequivalent_atoms*N_orbitals)
        Self-energy at infinite frequency 
      U: float
        The columb interaction strength in eV to be used in the DMFT calculation
      J: float
        The Hund's coupling in eV to be used in the DMFT calculation
      nf: float
        The nominal occupancy of the correlated sites
    
    Returns
    -------
      lines: list of strings
        A list of all the relevant lines needed to construct the sig.inp text file
    """
    # header1 = "# s_oo= [25.94191367094248, 25.97927152427038, 26.05945867814713, 26.06381971479536, 26.03294235710237, 26.04283272307152, 25.96716318919691, 26.0122262627843, 25.98858687107719, 26.0708059030178, 26.04368953407744, 26.0672438591148, 26.14826552518523, 26.14590929289391, 26.12531807149132, 25.96790067151829, 25.96873105519074, 25.97613803667094, 25.95033170658937, 25.97092521181814]\n"
    header1 = "# s_oo= [" + ', '.join(f'{x:.14f}' for x in sinf) + "]\n"
    # header2 = "# Edc= [52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 52.5]\n"
    Edc = U*(nf - 0.5) - 0.5*J*(nf-1.0)
    header2 = "# Edc= [" + ', '.join(f'{Edc}' for _ in range(len(sinf))) + "]\n"


    # valid_indices = []
    # for i in range(len(data.eq_inds)):
    #     first_idx = data.eq_inds[i][0]
    #     if first_idx in data.cor_atom_inds:
    #         valid_indices.append(first_idx.numpy())
    # valid_indices = np.array(valid_indices)
    # sig = sig[valid_indices]

    outdata = np.zeros((sig.shape[1], 1 + 2*sig.shape[0]*sig.shape[2]))
    outdata[:,0] = iws 

    ## sig in form [n_atoms, n_matsubara, n_orbitals, [real, imag]]
    for a in range(sig.shape[0]):
        for o in range(sig.shape[2]):
            outdata[:,1 + 10*a + 2*o] = sig[a,:,o].real
            outdata[:,1 + 10*a + 2*o + 1] = sig[a,:, o].imag
    lines = [header1, header2]
    for i in range(len(outdata)):
        sig_line = ' '.join(f'{num:.18e}' for num in outdata[i]) + "\n"
        lines.append(sig_line)
    return lines


def plot_predictions(dataset, ind_restrict=None, orbital=0):
    ### Filter out irrelevant points ###
    filtered_dataset = []
    if ind_restrict is not None:
        for i in range(len(dataset)):
            if i not in ind_restrict:
                filtered_dataset.append(dataset[i])
    else:
        filtered_dataset = dataset

    select_inds = np.arange(0, len(filtered_dataset), 1)
    np.random.shuffle(select_inds)


    ### Plot 9 random datapoints ### 
    iws = dataset[0].iws
    N1 = 3
    N2 = 3
    fig, axs = plt.subplots(N1, N2)

    colors_gt = ["lightcoral", "firebrick", "lightsalmon", "saddlebrown"]
    colors_pred = ["cornsilk", "khaki", "yellowgreen", "lawngreen"]

    for i in range(N1):
        for j in range(N2):
            ind = select_inds[N2*i + j]
            if i == 0 and j ==0:
                axs[i,j].plot(iws[0], dataset[ind].sig[0,0, :, orbital, 0].numpy(), c="red", label="Pred Re{$\Sigma$(i$\omega_n$)}")
                axs[i,j].plot(iws[0], dataset[ind].sig[0,0, :, orbital, 1].numpy(), c="blue", label="Pred Im{$\Sigma$(i$\omega_n$)}")
            else:
                axs[i,j].plot(iws[0], dataset[ind].sig[0,0, :, orbital, 0].numpy(), c="red")
                axs[i,j].plot(iws[0], dataset[ind].sig[0,0, :, orbital, 1].numpy(), c="blue")
            axs[i,j].axhline(y=0, linestyle="--", c="black")

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("i$\omega_n$", labelpad=5, fontsize=25)
    plt.ylabel("$\Sigma$(i$\omega_n$)", labelpad=20, fontsize=25)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    # plt.tight_layout()
    plt.show()


def train_nequip_ef(config_path, dataset):
    atoms = []
    for d in dataset:
        cell = d.lattice[0].cpu().numpy()
        pos = d.pos.cpu().numpy()
        ef = d.ef[0].item()
        tatom = Atoms(symbols = d.symbol, positions = pos, cell = cell, pbc=True)
        tatom.calc = SinglePointCalculator(atoms=tatom, energy=ef, forces=None, stress=None)
        atoms.append(tatom)
    write("dmft_atoms_efs.extxyz", atoms, format='extxyz')
    os.system("rm -r results")
    os.system("python utils_nequip.py " + config_path + " " + str(len(atoms)))
    os.system("rm dmft_atoms_efs.extxyz")


def eval_nequip_ef(model_path, dataset, display=True, img_save_dir = None):
    if os.path.isfile(model_path):
        print("Deployed model already exists, using that")
    else:
        print("Deploying model...")
        os.system("nequip-deploy build --train-dir results/dmft/fe-dmft-ef-model/ " + model_path)
    calc = nequip_calculator(model_path)
    pred_efs = []
    act_efs = []
    mae = 0
    for d in tqdm(dataset, desc="Evalulating NequIP fermi energy predictions..."):
        cell = d.lattice[0].cpu().numpy()
        pos = d.pos.cpu().numpy()
        ef = d.ef[0].item()
        tatom = Atoms(symbols = d.symbol, positions = pos, cell = cell, pbc=True)
        tatom.calc = calc 
        pred_efs.append(tatom.get_potential_energy())
        act_efs.append(ef)
        mae += np.abs(pred_efs[-1] - act_efs[-1])
    mae /= len(dataset)
    plt.scatter(pred_efs, act_efs, c="red")
    plt.xlim(np.amin(act_efs)-3, np.amax(act_efs)+3)
    plt.ylim(np.amin(act_efs)-3, np.amax(act_efs)+3)
    x = np.linspace(np.amin(act_efs), np.amax(act_efs), 1000)
    plt.plot(x, x, linestyle="--", c='black')
    plt.xlabel("E$_f$ Predictions (eV)")
    plt.ylabel("E$_f$ Actual (eV)")
    plt.text(np.amin(act_efs), np.amax(act_efs), f"MAE: {mae:.5f}", fontsize=15)
    if display:
        plt.show()
    elif not(display) and img_save_dir is not None:
        plt.savefig(f'{img_save_dir}/ef_accuracy.pdf')

    