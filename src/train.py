from EF_Model import get_standard_ef_model
from Sinf_Model import get_standard_sinf_model
from Sig_iws_Model import get_standard_full_sig_model
from utils import build_data, get_average_neighbor_count, get_sig_file_text, \
                  train_test_split, train_full_sig, evaluate_full_sig, train_nequip_ef, \
                  eval_nequip_ef, train_sinf, evaluate_sinf, evaluate_full_sig_legendre, \
                  train_ef, evaluate_ef
import pickle 
import os 
import numpy as np
from io import StringIO
from tqdm import tqdm
import torch

### just for testing ###
from torch.utils.data import DataLoader
from utils import collate_to_list
#######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}\n")

read_dataset = True
# if "dataset.pkl" in os.listdir():
#     read_dataset = True 


### Testing on previous datasets with already provided self energies ###
if not(read_dataset):
    source_dir = "../ATOMS_SIGS_EFS_DATA/"
    # source_dir = "/home/akldfas/Documents/DMFT/SIG_ML/Fe/SigML/E3NN_Test/ATOMS_SIGS_EFS_DATA/"
    patoms = []
    psig_texts = []
    pefs = []
    for pf in os.listdir(source_dir):
        if pf != "fe2o2_asefs.pkl":
            continue
        with open(source_dir + pf, "rb") as f:
            tatoms, tsig_texts, tefs = pickle.load(f)
        patoms.extend(tatoms)
        psig_texts.extend(tsig_texts)
        pefs.extend(tefs)

    dataset = build_data(patoms, sig_texts=psig_texts, efs=pefs, device=device, radial_cutoff=5.0)
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)


ave_neighbors = get_average_neighbor_count(dataset)
# n_matsubara = len(dataset[0].iws[0])
n_matsubara = dataset[0].sig.shape[2]
train_data, test_data = train_test_split(dataset, train_percent=0.9, seed=34533)
print("Training on dataset of length", len(train_data))



### Training for the Sig(iwn) - Sig(iwn -> infty) model ###
# full_sig_model = get_standard_full_sig_model(n_matsubara, ave_neighbors, radial_cutoff=5.0, device=device)
# opt = torch.optim.AdamW(full_sig_model.parameters(), lr=0.01, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
# loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
# # loss_fn = torch.nn.L1Loss(reduction="sum")
# save_path = "SAVED_MODELS/full_sig_model_fe2o2.pth"

# # full_sig_model.load_state_dict(torch.load(save_path))
# train_full_sig(full_sig_model, opt, train_data, loss_fn, scheduler, save_path = save_path, max_iter=50, val_percent = 0.1, device=device, batch_size=1)
# full_sig_model.load_state_dict(torch.load(save_path))
# for o in range(5):
#     evaluate_full_sig_legendre(full_sig_model, test_data, orbital=o, atom=1, display=True)
# exit()
############################################################




### ### Two options for Ef model: Using NequIP or using custom E3NN model ### ###

### Custom E3NN Model ###
# ef_model = get_standard_ef_model(ave_neighbors, cutoff=5.0, device=device)
# opt = torch.optim.AdamW(ef_model.parameters(), lr=0.01/4, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
# loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
# save_path = "SAVED_MODELS/ef_model_fe2o2.pth"

# ef_model.load_state_dict(torch.load(save_path))
# train_ef(ef_model, opt, train_data, loss_fn, scheduler, save_path = save_path, max_iter=10, val_percent = 0.1, device=device, batch_size=1)
# evaluate_ef(ef_model, test_data)

## NequIP Model ###
# train_nequip_ef(config_path="./default_config.yaml", dataset=train_data)
# eval_nequip_ef(model_path="SAVED_MODELS/nequip_ef_model_fe2o2.pth", dataset=test_data, display=True, img_save_dir="output_images")
# exit()
####################################################################################


############## Training for the S_infty model ############################
# sinf_model = get_standard_sinf_model(ave_neighbors, cutoff=5.0, device=device)
# opt = torch.optim.AdamW(sinf_model.parameters(), lr=0.01, weight_decay=0.05)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.25)
# loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
# save_path = "SAVED_MODELS/sinf_model_fe2o2.pth"

# sinf_model.load_state_dict(torch.load(save_path))
# train_sinf(sinf_model, opt, train_data, loss_fn, scheduler, save_path, max_iter=50)
# sinf_model.load_state_dict(torch.load(save_path))
# evaluate_sinf(sinf_model, test_data, display=True, img_save_dir="output_images", device=device)
# exit()
##########################################################################











