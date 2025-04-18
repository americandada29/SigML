from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import copy
from network import Network
from torch_geometric.data import Batch


class PeriodicNetworkReal(Network):
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

        ## Dropout layer
        # self.dropout = nn.Dropout(p=0.1)


    def forward(self, data_inp: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data = copy.deepcopy(data_inp)
        data.x = F.sigmoid(self.em(data.x))
        data.z = F.sigmoid(self.em(data.z))
        output = super().forward(data)
        # output = torch.abs(F.silu(output))
        # output = torch.tanh(output)
        output = F.silu(output)
        # output = self.dropout(output)
        eps = 1e-8
        norm = torch.norm(output, p=2, dim=1, keepdim=True)
        output = self.gamma * output / (norm + eps)
        #output = -1*output
        
        return output


class PeriodicNetworkImag(Network):
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

        ## Dropout layer
        # self.dropout = nn.Dropout(p=0.1)


    def forward(self, data_inp: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data = copy.deepcopy(data_inp)
        data.x = F.sigmoid(self.em(data.x))
        data.z = F.sigmoid(self.em(data.z))
        output = super().forward(data)
        output = torch.abs(F.silu(output))
        # output = self.dropout(output)
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
            model_real = PeriodicNetworkReal(in_dim = in_dim,
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
            model_imag = PeriodicNetworkImag(in_dim = in_dim,
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




    def forward(self, data: tg.data.Batch) -> list[torch.Tensor]:
        # — Accept either a Python list of Data *or* a Batch —
        if isinstance(data, list):
            # collate to a Batch so .ptr/.batch get created
            batch = Batch.from_data_list(data, exclude_keys=["sig", "iws"])
        elif isinstance(data, tg.data.Batch):
            batch = data
        else:
            raise TypeError(f"Expected list[Data] or Batch, got {type(data)}")

        # 1) get the ptr vector (graph boundaries) and total_atoms
        ptr = batch.ptr                  # shape: [batch_size+1]
        total_atoms = int(ptr[-1].item())

        # 2) run each orbital‐model exactly once over the *flat* Batch
        #    each m(batch) -> (total_atoms, n_mats)
        reals = [m(batch).unsqueeze(-1) for m in self.models_real]
        imags = [m(batch).unsqueeze(-1) for m in self.models_imag]

        # 3) stack orbitals -> (total_atoms, n_mats, n_orb)
        real_full = torch.cat(reals, dim=-1)
        imag_full = torch.cat(imags, dim=-1)

        # 4) pack real & imag -> (total_atoms, n_mats, n_orb, 2)
        combined = torch.stack((real_full, imag_full), dim=-1)

        # 5) split back into per-graph pieces
        sizes = (ptr[1:] - ptr[:-1]).tolist()   # [n_atoms₀, n_atoms₁, …]
        per_graph = torch.split(combined, sizes, dim=0)

        # 6) return a Python list of Tensors
        return list(per_graph)
    

def get_standard_full_sig_model(n_matsubara, ave_neighbor_count, weight_path=None, device="cpu"):
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
                            neighbor_count=ave_neighbor_count).to(device)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model.to(torch.float32)


