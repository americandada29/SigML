from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import copy
from network import Network



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
        if self.pool == True:
            output = torch.sum(output)
        return output


def get_standard_sinf_model(ave_neighbor_count, weight_path=None):
    out_dim = 5
    em_dim = 16
    model = Sinf_Model(in_dim = 118,
                            em_dim= em_dim,
                            irreps_in = str(em_dim) + "x0e",
                            irreps_out = str(out_dim) + "x0e",
                            irreps_node_attr = str(em_dim) + "x0e",
                            layers=2,
                            mul=32,
                            lmax=2,
                            max_radius=3.0,
                            num_neighbors=ave_neighbor_count,
                            reduce_output=False)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model.to(torch.float32)






    




