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
            ys.append(custom_renorm(gxs[i][j]))
            # ys.append(gxs[i][j])
    xs = np.array(xs)
    ys = np.array(ys)
    xs = torch.Tensor.float(torch.from_numpy(xs)).to(device)
    ys = torch.from_numpy(ys).to(device)
    dataset = SigmlDataset(xs, ys)
    return dataset

def complex_mse_loss(output, target):
    return nn.MSELoss()(output.real, target.real) + nn.MSELoss()(output.imag, target.imag)



class RealToComplexCNN(nn.Module):
    def __init__(self, input_dim=50, output_shape=(5, 110), latent_channels=128, dropout_prob=0.3):
        """
        Maps a 1D real vector (length input_dim) to a 2D complex matrix of shape (5, 110).
        Uses a fully connected layer to project the input to a latent space of shape (latent_channels, 5, 110),
        then applies a 2D convolution with a (1x10) kernel to scan vertically across each row.
        """
        super(RealToComplexCNN, self).__init__()
        self.output_shape = output_shape  # (rows, columns) = (5, 110)
        self.num_rows = output_shape[0]   # 5 rows
        self.num_cols = output_shape[1]   # 110 columns
        self.latent_channels = latent_channels

        # Fully connected layer: projects input to (batch, latent_channels, 5, 110)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.num_rows * self.num_cols * latent_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

        # 2D convolution block with a (1x10) kernel scanning vertically across each row
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=latent_channels, out_channels=2, kernel_size=(1, 9), stride=1, padding=(0, 4)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Fully connected transformation: project input to (batch, 5*110*latent_channels)
        x = self.fc(x)
        x = x.view(batch_size, self.latent_channels, self.num_rows, self.num_cols)  # Reshape to (batch, latent, 5, 110)

        # Apply the 2D convolution block
        x = self.conv2d(x)  # Output: (batch, 2, 5, 110)

        # Split into real and imaginary parts and combine into a complex tensor.
        real_part = x[:, 0, :, :]  # Shape: (batch, 5, 110)
        imag_part = x[:, 1, :, :]  # Shape: (batch, 5, 110)
        complex_output = torch.complex(real_part, imag_part)
        return complex_output.to(torch.complex128) 



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("atoms_fingerprints.pkl","rb") as f:
        atoms, fps = pickle.load(f)
    all_iws, all_gxs = sig_lib.read_gxs()
    all_iws, all_sigs = sig_lib.get_sigs()



    train_fps = fps[:80]
    train_gxs = all_sigs[:80]

    val_fps = fps[80:]
    val_gxs = all_sigs[80:]

    dataset = create_dataset(train_fps, train_gxs, device=device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    val_dataset = create_dataset(val_fps, val_gxs, device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    model = RealToComplexCNN().to(device)

    # example_input = torch.randn(8, 50)  
    # output = model(example_input)
    # print("Output shape:", output.shape) 
    # print("Output dtype:", output.dtype)  
    # exit()




    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    epochs = 50
    losses = []
    for epoch in range(epochs):
        model.train()
        tloss = 0
        count = 0
        vloss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = complex_mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            tloss = tloss + loss.item()
            for vali, valt in val_dataloader:
                val_output = model(vali)
                vloss += complex_mse_loss(val_output, valt)
            count += 1
        scheduler.step()
        # if epoch % 50 == 0:
        print(f"Epoch {epoch}, Training Loss: {tloss/count:.6f}, Validation Loss: {vloss/len(val_dataset)}")
        # losses.append(tloss/count)

    torch.save(model.state_dict(), "fullsig_cnn.pth")

    # plt.plot(losses, marker="o")
    # plt.show()




















