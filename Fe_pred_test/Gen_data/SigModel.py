import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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
            # ys.append(gxs[i][j])
            ys.append(gxs[i][j])
    xs = np.array(xs)
    ys = np.array(ys)
    xs = torch.Tensor.float(torch.from_numpy(xs)).to(device).double()
    ys = torch.from_numpy(ys).to(device).double()


    dataset = SigmlDataset(xs, ys)
    return dataset

class SigInfModel(nn.Module):
    def __init__(self, input_length = 50, output_length = 5, num_filters=100, fc_bias=True, mean=0.0, std=0.0):
        super(SigInfModel, self).__init__()
        self.fc1 = nn.Linear(input_length, num_filters, bias=fc_bias).double()
        self.fc2 = nn.Linear(num_filters, output_length, bias=fc_bias).double()

        self.mean = mean 
        self.std = std

    def forward(self, x):
        y = F.tanh(self.fc1(x))
        y = self.fc2(y)
        y = (y + self.mean)*self.std
        return y
    


class RealToComplexCNN(nn.Module):
    def __init__(self, input_length=50, output_length=110, num_filters=32):
        super(RealToComplexCNN, self).__init__()
        self.input_length = input_length
        self.output_length = output_length

        # 1D Convolution Layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
        self.batchnorm = nn.BatchNorm1d(num_filters)
        self.fc = nn.Linear(input_length * num_filters, output_length)

        self.conv3 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm1d(num_filters)
        self.fc2 = nn.Linear(input_length * num_filters, output_length)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0], 1, x.shape[1])
 
        x1 = F.relu(self.conv1(x))  # (batch_size, num_filters, 50)
        x1 = F.relu(self.batchnorm(self.conv2(x1)))  # (batch_size, num_filters, 50)
        x1 = x1.view(batch_size, -1)  # Shape: (batch_size, 50 * num_filters)
        x1 = self.fc(x1)  # Shape: (batch_size, 110)

        x2 = F.relu(self.conv3(x))  # (batch_size, num_filters, 50)
        x2 = F.relu(self.batchnorm2(self.conv4(x2)))  # (batch_size, num_filters, 50)
        x2 = x2.view(batch_size, -1)  # Shape: (batch_size, 50 * num_filters)
        x2 = self.fc(x2)  # Shape: (batch_size, 110)
  
        y = torch.complex(x1, x2)

        return y.to(torch.complex128)
    

class FullRealToComplexCNN(nn.Module):
    def __init__(self, input_length=50, output_length=110, num_filters=32):
        super(FullRealToComplexCNN, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = RealToComplexCNN().to(device)
        self.model2 = RealToComplexCNN().to(device)
        self.model3 = RealToComplexCNN().to(device)
        self.model4 = RealToComplexCNN().to(device)
        self.model5 = RealToComplexCNN().to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        out4 = self.model4(x)
        out5 = self.model5(x)

        y = torch.stack((out1, out2, out3, out4, out5), dim=1)
        return y.to(torch.complex128)
    

def create_dataset_fullsig(fps, gxs, device="cpu"):
    xs = []
    ys = []
    for i in range(len(fps)):
        for j in range(len(fps[i][0])):
            xs.append(fps[i][0][j])
            # ys.append(custom_renorm(gxs[i][j]))
            # ys.append(gxs[i][j])
            ys.append(gxs[i][j])
    xs = np.array(xs)
    ys = np.array(ys)
    xs = torch.Tensor.float(torch.from_numpy(xs)).to(device)
    ys = torch.from_numpy(ys).to(device)

    dataset = SigmlDataset(xs, ys)
    return dataset