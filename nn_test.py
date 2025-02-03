import pickle
import numpy as np
import matplotlib.pyplot as plt
import sig_lib 
import os
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


def create_dataset(fps, gxs):
    xs = []
    ys = []
    for i in range(len(fps)):
        for j in range(len(fps[i][0])):
            xs.append(fps[i][0][j])
            ys.append(custom_log(gxs[i][j]))
    xs = np.array(xs)
    ys = np.array(ys)
    xs = torch.Tensor.float(torch.from_numpy(xs))
    ys = torch.from_numpy(ys)
    dataset = SigmlDataset(xs, ys)
    return dataset

def complex_mse_loss(output, target):
    return nn.MSELoss()(output.real, target.real) + nn.MSELoss()(output.imag, target.imag)

class RealToComplexCNN(nn.Module):
    def __init__(self, input_dim=50, output_shape=(5, 19), hidden_dim=128):
        super(RealToComplexCNN, self).__init__()
        self.output_shape = output_shape  # (5, 19)
        
        # self.fc = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 5 * 5 * 32)  # Reshape into a 2D latent space
        # )
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.lin3 = nn.Linear(hidden_dim//2, 5*5*32)
        
        self.conv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 2, kernel_size=(3, 7), stride=(1, 3), padding=(1, 0))
        # self.conv3 = nn.ConvTranspose2d(16, 2, kernel_size=(3, 5), stride=(1, 2), padding=(1, 1))

    def forward(self, x):
        batch_size = x.shape[0]

        # Use fully connected NN first
        # x = self.fc(x)  # Output shape: (batch_size, 5*5*32)

        x = nn.ReLU()(self.lin1(x))
        x = nn.ReLU()(self.lin2(x))
        x = self.lin3(x)

        # Reshape into (batch, 32, 5, 5)
        x = x.view(batch_size, 32, 5, 5)

        # Transform into a 5x19 matrix
        x = torch.relu(self.conv1(x))  # (batch, 32, 5, 5)
        x = torch.relu(self.conv2(x))  # (batch, 16, 5, 9)


        # Real and imaginary parts
        real_part = x[:, 0, :, :].to(torch.float64)
        imag_part = x[:, 1, :, :].to(torch.float64)
        complex_output = torch.complex(real_part, imag_part)
        
        return complex_output # (batch, 5, 9)




with open("atoms_fingerprints.pkl","rb") as f:
    atoms, fps = pickle.load(f)
all_iws, all_gxs = sig_lib.read_gxs()
all_iws, all_sigs = sig_lib.get_sigs()
dataset = create_dataset(fps, all_gxs)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


model = RealToComplexCNN()

# example_input = torch.randn(8, 50)  
# output = model(example_input)
# print("Output shape:", output.shape) 
# print("Output dtype:", output.dtype)  
# exit()




batch_size = 1

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 500
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = complex_mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
















