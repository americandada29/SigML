from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import copy
from sigml.models.network import Network
from torch_geometric.data import Batch


"""
Sig_iws_Model.py

This file contains the definition of the CrystalSelfEnergyNetwork class, which is a PyTorch model for predicting the self energy of a material.
- CrystalSelfEnergyNetwork: The CrystalSelfEnergyNetwork class, which is a PyTorch model for predicting the self energy of a material.
- get_standard_full_sig_model: Get a standard $\Sigma(i\omega)$ model
"""

class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        self.n_matsubara = self.irreps_out[0][0]//2

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)

        ## Learnable scaling factor
        # self.gamma_real = torch.nn.Parameter(torch.tensor(1.0))
        # self.gamma_imag = torch.nn.Parameter(torch.tensor(1.0))
        ## Dropout layer
        # self.dropout = nn.Dropout(p=0.1)


        ### The PReLU activation, sent from the heavens for machine learning the self energy ###
        self.prelu_real = nn.PReLU()
        self.prelu_imag = nn.PReLU()

    def forward(self, data_inp: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data = copy.deepcopy(data_inp)
        data.x = F.sigmoid(self.em(data.x))
        data.z = F.sigmoid(self.em(data.z))
        output = super().forward(data)


        ### Only use correlated inequivalent atoms ###
        valid_indices = []
        for i in range(len(data.eq_inds)):
            first_idx = data.eq_inds[i][0]
            if first_idx in data.cor_atom_inds:
                valid_indices.append(first_idx)
        valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=output.device)
        output = output[valid_indices]
        

        # output = output.view(output.shape[0], self.n_matsubara, 2)
        # real = self.prelu_real(output[:, :, 0]) 
        # imag = self.prelu_imag(output[:, :, 1]) 
        # output = torch.stack((real, imag), dim=-1)
        # output = torch.view_as_complex(output)

        output = self.prelu_real(output)
        
        return output


class CrystalSelfEnergyNetwork(torch.nn.Module):
    def __init__(self, in_dim, em_dim, out_dim, radial_cutoff, neighbor_count, orbital_count, layers, mul, lmax, **kwargs) -> None:
        super().__init__()
        self.orbital_count = orbital_count
        models = []
        for o in range(orbital_count):
            model = PeriodicNetwork(in_dim = in_dim,
                            em_dim= em_dim,
                            irreps_in = str(em_dim) + "x0e",
                            irreps_out = str(out_dim) + "x0e",
                            irreps_node_attr = str(em_dim) + "x0e",
                            layers=layers,
                            mul=mul,
                            lmax=lmax,
                            max_radius=radial_cutoff,
                            num_neighbors=neighbor_count.mean(),
                            reduce_output=False,
                            **kwargs)

            models.append(model)
        self.models = nn.ModuleList(models)


    def forward(self, data_inp: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data = copy.deepcopy(data_inp)
        outputs = []
        for o in range(self.orbital_count):
            outputs.append(self.models[o](data))
        output = torch.stack(outputs, dim=-1)
        return output
    
    
def get_standard_full_sig_model(ave_neighbor_count, leg_lmax, em_dim=64, mul=64, \
                                interaction_layers=2, radial_layers=2, radial_neurons=64, lmax=2, orbital_count=5, \
                                cutoff=4.0, weight_path=None, device="cpu"):
    r"""
    Get a standard :math:`\Sigma(i\omega)` model

    Parameters
    ----------
    ave_neighbor_count: int
        The average number of neighbors per atom
    leg_lmax: int
        The maximum angular momentum number :math:`l_max` used for the Legendre expansion
    em_dim: int
        Embedding dimension, the atomic one-hot representations for each atom are converted to from :math:`\mathbb{R}^{N_{atoms}\times 118` to 
        :math:`\mathbb{R}^{N_{atoms}\times em\_dim}` before being fed into network
    mul: int
        Multiplicity of neurons within equivariant layers of the neural network. Higher means larger network
    interaction_layers: int
        Number of convolutional interaction layers 
    radial_layers: int
        Number of radial layers for encoding of atomic positions into spherical harmonics representation
    radial_neurons: int
        Number of neurons for each radial layer
    lmax: int
        Maximum irrep order to consider when building equivariant layers 
    orbital_count: int
        Number of orbitals within correlated subspace for each atoms. Currently only supports atoms of the same orbital count
    cutoff: float, optional, default = 4.0
        The cutoff radius of the model
    weight_path: str, optional, default = None
        The path to the weights of the model if model has been previously trained and saved
    device: str, optional, default = "cpu"
        The device to build the model on

    Returns
    -------
    model: CrystalSelfEnergyNetwork
        The standard :math:`\Sigma(i\omega)` model
    """
    out_dim = leg_lmax
    em_dim = em_dim
    model = CrystalSelfEnergyNetwork(in_dim=118,
                            em_dim=em_dim,
                            out_dim=out_dim,
                            layers=interaction_layers,
                            mul=mul,
                            lmax=lmax,
                            orbital_count=orbital_count,
                            radial_cutoff=cutoff,
                            neighbor_count=ave_neighbor_count,
                            radial_layers = radial_layers,
                            radial_neurons = radial_neurons).to(device)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model.to(torch.float32)


