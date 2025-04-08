from EF_Model import get_standard_ef_model
from Sinf_Model import get_standard_sinf_model
from Sig_iws_Model import get_standard_full_sig_model
from utils import build_data, get_average_neighbor_count, get_sig_file_text, \
                  train_test_split, train_full_sig, evaluate_full_sig, train_nequip_ef, \
                  eval_nequip_ef, train_sinf, evaluate_sinf
import pickle 
import os 
import numpy as np
from io import StringIO
from tqdm import tqdm
import torch

### Testing on previous datasets with already provided self energies ###
source_dir = "../ATOMS_SIGS_EFS_DATA/"
patoms = []
psig_texts = []
pefs = []
for pf in os.listdir(source_dir):
   with open(source_dir + pf, "rb") as f:
       tatoms, tsig_texts, tefs = pickle.load(f)
   patoms.extend(tatoms)
   psig_texts.extend(tsig_texts)
   pefs.extend(tefs)

dataset = build_data(patoms, sig_texts=psig_texts, efs=pefs)
ave_neighbors = get_average_neighbor_count(dataset)
n_matsubara = len(dataset[0].iws[0])
train_data, test_data = train_test_split(dataset, train_percent=0.9, seed=34533)
print("Training on dataset of length", len(train_data))



### Training for the Sig(iwn) - Sig(iwn -> infty) model ###
# full_sig_model = get_standard_full_sig_model(n_matsubara, ave_neighbors)
# opt = torch.optim.AdamW(full_sig_model.parameters(), lr=0.01, weight_decay=0.05)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.25)
# loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
# save_path = "SAVED_MODELS/full_sig_model.pth"

# full_sig_model.load_state_dict(torch.load(save_path))
# train_full_sig(full_sig_model, opt, train_data, loss_fn, scheduler, save_path = save_path, max_iter=15, val_percent = 0.1, device="cpu", batch_size=1)
# full_sig_model.load_state_dict(torch.load(save_path))
# for o in range(5):
#     evaluate_full_sig(full_sig_model, test_data, orbital=o)
# exit()
############################################################




### ### Two options for Ef model: Using NequIP or using custom E3NN model ### ###

### Custom E3NN Model ###
# ef_model = get_standard_ef_model(ave_neighbors)

### NequIP Model ###
# train_nequip_ef(config_path="./default_config.yaml", dataset=train_data)
# eval_nequip_ef(model_path="SAVED_MODELS/nequip_ef_model.pth", dataset=test_data)
# exit()
####################################################################################


############## Training for the S_infty model ############################
sinf_model = get_standard_sinf_model(ave_neighbors)
opt = torch.optim.AdamW(sinf_model.parameters(), lr=0.01, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
save_path = "SAVED_MODELS/sinf_model.pth"

sinf_model.load_state_dict(torch.load(save_path))
train_sinf(sinf_model, opt, train_data, loss_fn, scheduler, save_path, max_iter=5)
sinf_model.load_state_dict(torch.load(save_path))
evaluate_sinf(sinf_model, test_data)
exit()
##########################################################################











