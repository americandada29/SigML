from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import copy
from network import Network


class PeriodicNetwork(Network):
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

        ## Learnable scaling factor
        self.gamma = torch.nn.Parameter(torch.tensor(1.0))


    def forward(self, data_inp: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data = copy.deepcopy(data_inp)
        data.x = F.sigmoid(self.em(data.x))
        data.z = F.sigmoid(self.em(data.z))
        output = super().forward(data)
        output = torch.abs(F.silu(output))
        eps = 1e-8
        norm = torch.norm(output, p=2, dim=1, keepdim=True)
        output = self.gamma * output / (norm + eps)
        output = -1*output
        
        return output


class CrystalSelfEnergyNetwork(torch.nn.Module):
    def __init__(self, in_dim, em_dim, out_dim, radial_cutoff, neighbor_count, orbital_count, layers, mul, lmax) -> None:
        super().__init__()
        self.orbital_count = orbital_count
        models_real = []
        models_imag = []
        for o in range(orbital_count):
            model_real = PeriodicNetwork(in_dim = in_dim,
                            em_dim= em_dim,
                            irreps_in = str(em_dim) + "x0e",
                            irreps_out = str(out_dim) + "x0e",
                            irreps_node_attr = str(em_dim) + "x0e",
                            layers=layers,
                            mul=mul,
                            lmax=lmax,
                            max_radius=radial_cutoff,
                            num_neighbors=neighbor_count.mean(),
                            reduce_output=False)
            model_imag = PeriodicNetwork(in_dim = in_dim,
                            em_dim= em_dim,
                            irreps_in = str(em_dim) + "x0e",
                            irreps_out = str(out_dim) + "x0e",
                            irreps_node_attr = str(em_dim) + "x0e",
                            layers=layers,
                            mul=mul,
                            lmax=lmax,
                            max_radius=radial_cutoff,
                            num_neighbors=neighbor_count.mean(),
                            reduce_output=False)
            models_real.append(model_real)
            models_imag.append(model_imag)

        self.models_real = nn.ModuleList(models_real)
        self.models_imag = nn.ModuleList(models_imag)



    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        # Each model outputs (N_atoms, N_matsubara)
        outputs_real = []
        outputs_imag = []
        for i in range(self.orbital_count):
            out_real = self.models_real[i](data)  # shape: (N_atoms, N_matsubara)
            out_imag = self.models_imag[i](data)
            outputs_real.append(out_real.unsqueeze(-1))
            outputs_imag.append(out_imag.unsqueeze(-1))

        # Stack along the last dimension -> (N_atoms, N_matsubara, N_orbitals)
        output = torch.stack((torch.cat(outputs_real, dim=-1), torch.cat(outputs_imag, dim=-1)), dim=-1)
 
        return output
    

def get_standard_full_sig_model(n_matsubara, ave_neighbor_count, weight_path=None):
    out_dim = n_matsubara
    em_dim = 32
    model = CrystalSelfEnergyNetwork(in_dim=118,
                            em_dim=em_dim,
                            out_dim=out_dim,
                            layers=2,
                            mul=64,
                            lmax=2,
                            orbital_count=5,
                            radial_cutoff=3.0,
                            neighbor_count=ave_neighbor_count)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


