from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import copy
from sigml.models.network import Network

"""
Sinf_Model.py

This file contains the definition of the Sinf_Model class, which is a PyTorch model for predicting the self energy at infinite frequency of a material.
- Sinf_Model: The Sinf_Model class, which is a PyTorch model for predicting the self energy at infinite frequency of a material.
- get_standard_sinf_model: Get a standard $\Sigma_{\infty}$ model
"""

class Sinf_Model(Network):
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)
        
        self.n_matsubara = self.irreps_out[0][0]

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)
     


    def forward(self, data_inp: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data = copy.deepcopy(data_inp)
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
        output = super().forward(data)
        output = F.softplus(output)
        # if self.pool == True:
        #     output = torch.sum(output)
        
        # Collect indices of first elements in each equivalent group that are in correlated atoms
        valid_indices = []
        for i in range(len(data.eq_inds)):
            first_idx = data.eq_inds[i][0]
            if first_idx in data.cor_atom_inds:
                valid_indices.append(first_idx)
        valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=output.device)
        output = output[valid_indices]

        
        return output


def get_standard_sinf_model(ave_neighbor_count, em_dim=16, mul=32, \
                            interaction_layers=2, radial_layers=2, radial_neurons=64, lmax=2, \
                            orbital_count=5, cutoff=4.0, weight_path=None, device="cpu"):
    r"""
    Get a standard :math:`\Sigma_{\infty}` model

    Parameters
    ----------
    ave_neighbor_count: int
        The average number of neighbors per atom
    em_dim: int
        Embedding dimension, the atomic one-hot representations for each atom are converted to from :math:`\mathbb{R}^{N_{atoms}\times 118` to 
        :math:`\mathbb{R}^{N_{atoms}\times em_dim` before being fed into network
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
    model: Sinf_Model
        The standard :math:`\Sigma_{\infty}` model
    """
    out_dim = orbital_count
    em_dim = em_dim
    model = Sinf_Model(in_dim = 118,
						em_dim= em_dim,
						irreps_in = str(em_dim) + "x0e",
						irreps_out = str(out_dim) + "x0e",
						irreps_node_attr = str(em_dim) + "x0e",
						layers=interaction_layers,
						mul=mul,
						lmax=lmax,
						max_radius=cutoff,
						num_neighbors=ave_neighbor_count,
						radial_layers = radial_layers,
						radial_neurons = radial_neurons,
						reduce_output=False).to(device)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model.to(torch.float32)






    




