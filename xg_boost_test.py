import pickle
import numpy as np
import matplotlib.pyplot as plt
import sig_lib 
import os
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils



# Log with negatives for a 2D array
def custom_log(xs):
    newxs = np.zeros(xs.shape).astype(np.complex128)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            for k in range(2):
                x = 0
                ope = 0
                if k == 0:
                    x = xs[i,j].real
                    ope = 1.0
                else:
                    x = xs[i,j].imag
                    ope = 1j
                if x > 0:
                    newxs[i,j] += ope*np.log(x)
                elif x < 0:
                    newxs[i,j] += -ope*np.log(-x)
                elif x == 0:
                    newxs[i,j] = np.nan
    return newxs

def custom_renorm(xs):
    newxs = np.zeros(xs.shape).astype(np.complex128)
    newxs = xs.real/1000000 + 1j*xs.imag*10
    return newxs

def custom_unrenorm(xs):
    newxs = np.zeros(xs.shape).astype(np.complex128)
    newxs = 1000000 * xs.real + 1j*xs.imag/10
    return newxs

# Convert complex output to (radius, angle) representation
def radius_angle_repr(xs):
    newxs = np.zeros((xs.shape[0], xs.shape[1], 2))
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            x = xs[i,j]
            ang = np.angle(x)
            rad = np.abs(x)
            if np.abs(ang) > 1:
                if ang < 0:
                    ang = ang + np.pi 
                else:
                    ang = ang - np.pi
            newxs[i,j] = np.array([rad, ang])
    return newxs

# Dataset class for holding self energies (y) and fingerprints (x)
class SigmlDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, fps, gxs, transform=None):
        self.fps = fps 
        self.gxs = gxs

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        return self.fps[idx], self.gxs[idx]


def create_dataset(fps, gxs, device="cpu"):
    xs = []
    ys = []
    for i in range(len(fps)):
        for j in range(len(fps[i][0])):
            xs.append(fps[i][0][j])
            # ys.append(custom_renorm(gxs[i][j]))
            ys.append(gxs[i][j])
    xs = np.array(xs)
    ys = np.array(ys)
    xs = torch.Tensor.float(torch.from_numpy(xs)).to(device)
    ys = torch.from_numpy(ys).to(device)
    dataset = SigmlDataset(xs, ys)
    return dataset

def create_dataset_numpy(fps, gxs):
    xs = []
    ys = []
    for i in range(len(fps)):
        for j in range(len(fps[i][0])):
            xs.append(fps[i][0][j])
            # ys.append(custom_renorm(gxs[i][j]))
            # ys.append(gxs[i][j])
            ys.append(np.stack((gxs[i][j].real, gxs[i][j].imag), axis=-1))
    xs = np.array(xs)
    ys = np.array(ys).reshape(len(xs), -1)
    return xs, ys

def complex_mse_loss(output, target):
    return nn.MSELoss()(output.real, target.real) + nn.MSELoss()(output.imag, target.imag)

class PrintLossCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        """
        Called after each boosting iteration.
        
        Parameters:
            model     : the Booster instance being trained.
            epoch     : the current boosting round (0-indexed).
            evals_log : dict mapping dataset names to dicts of metric lists.
                        For example: {'train': {'rmse': [val1, val2, ...]}, 
                                      'validation': {'rmse': [val1, val2, ...]}}
        Returns:
            False (training continues).
        """
        msgs = []
        # Iterate over each dataset (e.g., "train", "validation")
        for data_name, metric_dict in evals_log.items():
            # metric_dict is a dictionary mapping metric names to lists of values.
            for metric_name, metric_values in metric_dict.items():
                # Check that there's at least one recorded value.
                if len(metric_values) > 0:
                    msgs.append(f"{data_name}-{metric_name}={metric_values[-1]:.4f}")
        print(f"Iteration {epoch + 1}: " + ", ".join(msgs))
        return False  # Returning True would stop training.



if __name__ == "__main__":

    with open("atoms_fingerprints.pkl","rb") as f:
        atoms, fps = pickle.load(f)
    all_iws, all_gxs = sig_lib.read_gxs()
    all_iws, all_sigs = sig_lib.get_sigs()

    print(all_gxs[10])
    exit()

    xs_train, ys_train = create_dataset_numpy(fps[:80], all_gxs[:80])
    xs_test, ys_test = create_dataset_numpy(fps[80:], all_gxs[80:])



    d_train = xgb.DMatrix(xs_train, label=ys_train)
    d_test = xgb.DMatrix(xs_test, label=ys_test)
    params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "eta": 0.1,
    "eval_metric": "rmse"           
    }
    evals = [(d_train, "train"), (d_test, "validation")]

    print("Starting training...\n")
    model = xgb.train(
    params=params,
    dtrain=d_train,
    num_boost_round=500,  ## Training steps essentially
    evals=evals,
    verbose_eval=False,            
    callbacks=[PrintLossCallback()]  # custom print callback
    )

    # -------------------------------
    # 5. Save the Trained Model
    # -------------------------------

    model_filename = "xgb_model.json"
    model.save_model(model_filename)
    print(f"\nModel saved to '{model_filename}'.")
    


    


 




















