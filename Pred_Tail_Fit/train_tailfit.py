import pickle
import numpy as np
import matplotlib.pyplot as plt
import sig_lib 
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



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
        for j in range(len(fps[i])):
            xs.append(fps[i][j])
            # ys.append(custom_renorm(gxs[i][j]))
            # ys.append(gxs[i][j])
            ys.append(gxs[i][j][0])
        # xs.append(fps[i])
        # ys.append(gxs[i])
    xs = np.array(xs)
    ys = np.array(ys)

    xs = torch.Tensor.float(torch.from_numpy(xs)).to(device)
    ys = torch.from_numpy(ys).to(device)

    dataset = SigmlDataset(xs, ys)
    return dataset

def complex_mse_loss(output, target):
    return nn.MSELoss()(output.real, target.real) + nn.MSELoss()(output.imag, target.imag)



# class RealToComplexCNN(nn.Module):
#     def __init__(self, input_length=50, output_length=110, num_filters=32):
#         super(RealToComplexCNN, self).__init__()
#         self.input_length = input_length
#         self.output_length = output_length

#         # 1D Convolution Layers
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
#         self.batchnorm = nn.BatchNorm1d(num_filters)
#         self.fc = nn.Linear(input_length * num_filters, output_length)

#         self.conv3 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
#         self.conv4 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
#         self.batchnorm2 = nn.BatchNorm1d(num_filters)
#         self.fc2 = nn.Linear(input_length * num_filters, output_length)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.view(x.shape[0], 1, x.shape[1])
 
#         x1 = F.relu(self.conv1(x))  # (batch_size, num_filters, 50)
#         x1 = F.relu(self.batchnorm(self.conv2(x1)))  # (batch_size, num_filters, 50)
#         x1 = x1.view(batch_size, -1)  # Shape: (batch_size, 50 * num_filters)
#         x1 = self.fc(x1)  # Shape: (batch_size, 110)

#         x2 = F.relu(self.conv3(x))  # (batch_size, num_filters, 50)
#         x2 = F.relu(self.batchnorm2(self.conv4(x2)))  # (batch_size, num_filters, 50)
#         x2 = x2.view(batch_size, -1)  # Shape: (batch_size, 50 * num_filters)
#         x2 = self.fc(x2)  # Shape: (batch_size, 110)
  
#         y = torch.complex(x1, x2)

#         return y.to(torch.complex128)
    

# class FullRealToComplexCNN(nn.Module):
#     def __init__(self, input_length=50, output_length=110, num_filters=32):
#         super(FullRealToComplexCNN, self).__init__()

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model1 = RealToComplexCNN().to(device)
#         self.model2 = RealToComplexCNN().to(device)
#         self.model3 = RealToComplexCNN().to(device)
#         self.model4 = RealToComplexCNN().to(device)
#         self.model5 = RealToComplexCNN().to(device)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         out1 = self.model1(x)
#         out2 = self.model2(x)
#         out3 = self.model3(x)
#         out4 = self.model4(x)
#         out5 = self.model5(x)

#         y = torch.stack((out1, out2, out3, out4, out5), dim=1)
#         return y.to(torch.complex128)



# class FullRealToComplexCNN(nn.Module):
#     def __init__(self, input_length=100, output_length=5, num_filters=25):
#         super(FullRealToComplexCNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels = 4, out_channels = 4, kernel_size = int(input_length/2 + 1))
#         self.conv2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=int(input_length/2) - 5 + 1)
#         self.batchnorm = nn.BatchNorm1d(4)
#         self.act1 = F.tanh

#     def forward(self, x):
#         y = self.act1(self.batchnorm(self.conv1(x)))
#         y = self.conv2(y)
#         return y.double()

class FullRealToComplexCNN(nn.Module):
    def __init__(self, input_length=18, output_length=1, hl1 = 10):
        super(FullRealToComplexCNN, self).__init__()
        self.fc1 = nn.Linear(input_length, hl1)
        self.fc2 = nn.Linear(hl1, output_length)

        self.act1 = F.softplus
        self.act2 = F.softplus

        self.norm = F.normalize

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x.double()




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sig_lib.write_atoms_lmbtrs()
    # atoms, fps = sig_lib.get_atoms_lmbtrs()
    sig_lib.write_atoms_soaps()
    atoms, fps = sig_lib.get_atoms_soaps()


    all_iws, all_sigs = sig_lib.get_sigs()
    all_tails = sig_lib.fit_sig_tails(all_iws, all_sigs)

    all_tails, fps = sig_lib.shuffle_data(all_tails, fps)

    N_train = int(0.9*len(atoms))

    assert len(all_tails) == len(fps)

    train_fps = fps[:N_train]
    train_gxs = all_tails[:N_train, :,:,0]

    val_fps = fps[N_train:]
    val_gxs = all_tails[N_train:,:,:,0]


    dataset = create_dataset(train_fps, train_gxs, device=device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    val_dataset = create_dataset(val_fps, val_gxs, device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    model = FullRealToComplexCNN().to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    epochs = 30
    loss_fn = nn.MSELoss()
    loss_fn_2 = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        tloss = 0
        count = 0
        vloss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            tloss = tloss + loss.item()
            count += 1

        scheduler.step()

        for vali, valt in val_dataloader:
                val_output = model(vali)
                vloss += loss_fn_2(val_output, valt)

        print(f"Epoch {epoch}, Training Loss: {tloss/count:.6f}, Validation Loss: {vloss/len(val_dataset)}")

    torch.save(model.state_dict(), "tailfit_fcnn.pth")
    with open("val_dataset.pkl","wb") as f:
        pickle.dump(val_dataset, f)





















