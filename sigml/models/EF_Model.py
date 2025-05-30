from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import copy
from sigml.models.network import Network

"""
EF_Model.py

This file contains the definition of the EF_Model class, which is a PyTorch model for predicting the fermi energy of a material.
- EF_Model: The EF_Model class, which is a PyTorch model for predicting the fermi energy of a material.
- get_standard_ef_model: Get a standard $E_f$ model
"""

class EF_Model(Network):
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)


        ## Trainable normalization layer
        self.n_matsubara = self.irreps_out[0][0]      

        self.prelu = torch.nn.PReLU()  


    def forward(self, data_inp: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data = copy.deepcopy(data_inp)
        data.x = F.sigmoid(self.em(data.x))
        data.z = F.sigmoid(self.em(data.z))

        output = super().forward(data)
        output = self.prelu(output)

        if self.pool == True:
            output = torch.sum(output)

        return output


def get_standard_ef_model(ave_neighbor_count, em_dim=32, mul=16, \
                            interaction_layers=2, radial_layers=2, radial_neurons=64, \
                            lmax=2, weight_path=None, cutoff=4.0, device="cpu"):
    r"""
    Get a standard :math:`E_f` model

    Parameters
    ----------
    ave_neighbor_count: int
        The average number of neighbors per atom
    em_dim: int
        Embedding dimension, the atomic one-hot representations for each atom are converted to from :math:`\mathbb{R}^{N_{atoms}\times 118}` to 
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
    weight_path: str, optional, default = None
        The path to the weights of the model if model has been previously trained and saved
    cutoff: float, optional, default = 4.0
        The cutoff radius of the model when considering neighbors for the graph neural network
    device: str, optional, default = "cpu"
        The device to build the model on
        
    Returns
    -------
    model: `sigml.models.EF_Model.EF_Model`
        The standard :math:`E_f` model
    """
    out_dim = 1
    em_dim = em_dim
    model = EF_Model(in_dim = 118,
                    em_dim= em_dim,
                    irreps_in = str(em_dim) + "x0e",
                    irreps_out = str(out_dim) + "x0e",
                    irreps_node_attr = str(em_dim) + "x0e",
                    layers=interaction_layers,
                    mul=mul,
                    lmax=lmax,
                    max_radius=cutoff,
                    num_neighbors=ave_neighbor_count,
                    radial_layers=radial_layers, 
                    radial_neurons = radial_neurons,
                    reduce_output=True).to(device)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model




    




