import pickle
import numpy as np
import matplotlib.pyplot as plt
import sig_lib 
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
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


def create_dataset(fps, gxs, mean=None, std=None, device="cpu"):
    xs = []
    ys = gxs
    for i in range(len(fps)):
        tfps = []
        for j in range(len(fps[i][0])):
            tfps.append(fps[i][0][j])
        xs.append(tfps)
    xs = np.array(xs)
    ys = np.array([[y] for y in ys])

    xs = torch.Tensor.float(torch.from_numpy(xs)).to(device).double()
    ys = torch.from_numpy(ys).to(device).double()

    dataset = SigmlDataset(xs, ys)
    return dataset

#def complex_mse_loss(output, target):
#    return nn.MSELoss()(output.real, target.real) + nn.MSELoss()(output.imag, target.imag)






class SigInfModel(nn.Module):
    def __init__(self, input_length = 50, output_length = 1, atoms=4, num_filters=100, fc_bias=True):
        super(SigInfModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels = 4, out_channels = 2, kernel_size = 26)
        self.conv1d1 = nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 16)
        self.batchnorm = nn.BatchNorm1d(1, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(10, 1)

    def forward(self, x):
        # y = F.relu(self.batchnorm(self.conv1d(x)))
        y = F.relu(self.conv1d(x))
        y = F.tanh(self.dropout(self.batchnorm(self.conv1d1(y))))
        y = self.fc1(y)
        return y


#class RealToComplexCNN(nn.Module):
#    def __init__(self, input_length=50, output_length=110, num_filters=32):
#        super(RealToComplexCNN, self).__init__()
#        self.input_length = input_length
#        self.output_length = output_length
#
#        # 1D Convolution Layers
#        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
#        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
#        self.batchnorm = nn.BatchNorm1d(num_filters)
#        self.fc = nn.Linear(input_length * num_filters, output_length)
#
#        self.conv3 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
#        self.conv4 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5, stride=1, padding=2)
#        self.batchnorm2 = nn.BatchNorm1d(num_filters)
#        self.fc2 = nn.Linear(input_length * num_filters, output_length)
#
#    def forward(self, x):
#        batch_size = x.shape[0]
#        x = x.view(x.shape[0], 1, x.shape[1])
# 
#        x1 = F.relu(self.conv1(x))  # (batch_size, num_filters, 50)
#        x1 = F.relu(self.batchnorm(self.conv2(x1)))  # (batch_size, num_filters, 50)
#        x1 = x1.view(batch_size, -1)  # Shape: (batch_size, 50 * num_filters)
#        x1 = self.fc(x1)  # Shape: (batch_size, 110)
#
#        x2 = F.relu(self.conv3(x))  # (batch_size, num_filters, 50)
#        x2 = F.relu(self.batchnorm2(self.conv4(x2)))  # (batch_size, num_filters, 50)
#        x2 = x2.view(batch_size, -1)  # Shape: (batch_size, 50 * num_filters)
#        x2 = self.fc(x2)  # Shape: (batch_size, 110)
#  
#        y = torch.complex(x1, x2)
#
#        return y.to(torch.complex128)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("atoms_fingerprints.pkl","rb") as f:
        atoms, fps = pickle.load(f)
    # all_sinfs, all_edcs = sig_lib.get_sinf_edc()
    _, all_efs = sig_lib.get_efs()


    inds = np.arange(0, len(atoms))
    np.random.shuffle(inds)
    newatoms = []
    newefs= []
    for i, ind in enumerate(inds):
        newatoms.append(atoms[ind])
        newefs.append(all_efs[ind])
    all_efs = np.array(newefs)
    atoms = newatoms

    dmean = np.mean(all_efs)
    dstd = np.std(all_efs)

    all_efs = (all_efs - dmean)/dstd
    


    tts = 0.9
    N = int(len(atoms)*tts)

    train_fps = fps[:N]
    train_efs= all_efs[:N]

    val_fps = fps[N:]
    val_efs= all_efs[N:]


    dataset = create_dataset(train_fps, train_efs, mean=dmean, std=dstd, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    val_dataset = create_dataset(val_fps, val_efs, mean=dmean, std=dstd, device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)


    model = SigInfModel().to(device).double()

    # example_input = torch.randn(8, 50)  
    # output = model(example_input)
    # print("Output shape:", output.shape) 
    # print("Output dtype:", output.dtype)  
    # exit()



    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    epochs = 100
    losses = []
    loss_fn = nn.MSELoss()
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
            for vali, valt in val_dataloader:
                val_output = model(vali)
                # print("Output:", val_output)
                # print("Valid Out:", valt)
                # print("##################################")
                vloss += loss_fn(val_output, valt)
            count += 1
        #scheduler.step(vloss)
        # if epoch % 50 == 0:
        print(f"Epoch {epoch}, Training Loss: {tloss/count:.6f}, Validation Loss: {vloss/len(val_dataset)}")
        # losses.append(tloss/count)
    for vali, valt in val_dataloader:
                val_output = model(vali)
                print("Output:", val_output)
                print("Valid Out:", valt)
                print("##################################")
                vloss += loss_fn(val_output, valt)

    torch.save(model.state_dict(), "siginf_ann.pth")

    # plt.plot(losses, marker="o")
    # plt.show()




















