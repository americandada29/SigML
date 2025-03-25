import torch
import torch.nn as nn
import math
import dlr_test
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt



def pull_train_test_data(train_test_percent = 0.9):
#   files = ["dmft_2_as.pkl","dmft_as.pkl"]
  files = ["dmft_as.pkl"]
  atoms = []
  sigs = []
  all_dlr_coeffs = []
  soaps = []
  for f in files:
    with open(f, "rb") as g:
      tatoms, tsigs = pickle.load(g)
    atoms.extend(tatoms)
    sigs.extend(tsigs)
    fstem = f.split(".")[0]
    dlr_f_name = fstem + "_dlr_coeffs.pkl"
    if dlr_f_name not in os.listdir("."):
        dlr_test.write_all_dlr_coeffs(sigs, E_max=50.0, fname=dlr_f_name)
    dlr_coeffs = dlr_test.read_dlr_coeffs(dlr_f_name)
    all_dlr_coeffs.extend(dlr_coeffs)
    soap_vecs = dlr_test.create_soap_vecs(tatoms, ["Fe"])
    soaps.extend(soap_vecs)
  
  for i in range(len(atoms)):
    assert soaps[i].shape[0] == all_dlr_coeffs[i].shape[1]
  
  x_data = []
  y_data = []
  for i in range(len(atoms)):
    for j in range(soaps[i].shape[0]):
        x_data.append(soaps[i][j])
        y_data.append(all_dlr_coeffs[i][:,j])
  N_train = int(train_test_percent*len(x_data))
  inds = np.arange(0, len(x_data), 1, dtype=np.int32)
  np.random.shuffle(inds)
  train_x = []
  train_y = []
  test_x = []
  test_y = []
  for i, ind in enumerate(inds):
    if i < N_train:
      train_x.append(x_data[ind])
      train_y.append(y_data[ind])
    else:
      test_x.append(x_data[ind])
      test_y.append(y_data[ind])
  
  train_x = torch.from_numpy(np.array(train_x)).type(torch.float32)
  train_y = torch.from_numpy(np.array(train_y))
  test_x = torch.from_numpy(np.array(test_x)).type(torch.float32)
  test_y = torch.from_numpy(np.array(test_y))


  return train_x, train_y, test_x, test_y

class ComponentNormalization(nn.Module):
    """
    Custom normalization layer that handles different scale magnitudes
    and allows for adaptive scaling of input features
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        
        # Learnable log-scale parameters
        self.log_scale = nn.Parameter(torch.zeros(num_features))
        
        # Learnable shift parameters
        self.shift = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # Apply log scaling to handle wide ranges
        scaled_x = torch.sign(x) * torch.log(torch.abs(x) + 1)
        
        # Compute mean and standard deviation
        mean = scaled_x.mean(dim=0)
        std = scaled_x.std(dim=0)
        
        # Normalize
        normalized = (scaled_x - mean) / (std + self.eps)
        
        # Apply learnable scaling and shifting
        return normalized * torch.exp(self.log_scale) + self.shift

class ComponentPredictionNetwork(nn.Module):
    """
    Network for predicting either real or imaginary components
    with adaptive normalization and residual connections
    """
    def __init__(self, input_size, hidden_layers=[64, 128, 64], output_size=165):
        super().__init__()
        
        # Input normalization
        self.input_norm = ComponentNormalization(input_size)
        
        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_layers
        
        for i in range(len(layer_sizes) - 1):
            layer = nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.ELU(),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.Dropout(0.1)
            )
            self.layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        
    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)
        
        # Residual connections
        identity = x
        for layer in self.layers:
            x = layer(x)
            # Optional residual connection
            if x.shape[-1] == identity.shape[-1]:
                x = x + identity
                identity = x
        
        # Final output
        return self.output_layer(x)

class ComplexTransformationModel(nn.Module):
    """
    Combined model with separate networks for real and imaginary components
    """
    def __init__(self, input_size=30, output_size=[33, 5]):
        super().__init__()
        
        # Separate networks for real and imaginary parts
        self.real_network = ComponentPredictionNetwork(input_size)
        self.imag_network = ComponentPredictionNetwork(input_size)
        
        # Output size configuration
        self.output_rows, self.output_cols = output_size
    
    def forward(self, x):
        # Predict real and imaginary components
        real_output = self.real_network(x)
        imag_output = self.imag_network(x)
        
        # Reshape to final matrix
        real_matrix = real_output.view(-1, self.output_rows, self.output_cols)
        imag_matrix = imag_output.view(-1, self.output_rows, self.output_cols)
        
        # Combine as complex tensor
        return torch.complex(real_matrix, imag_matrix)



def train_model(model, train_x, train_y, test_x, test_y, 
                num_epochs=100, learning_rate=0.1, 
                batch_size=2, device=None):
    """
    Training function with flexible data handling
    
    Args:
    - model: PyTorch model to train
    - train_x: Input training data [num_samples, input_size]
    - train_y: Output training data [num_samples, output_rows, output_cols]
    - test_x: Input test data [num_test_samples, input_size]
    - test_y: Output test data [num_test_samples, output_rows, output_cols]
    - num_epochs: Number of training epochs
    - learning_rate: Optimizer learning rate
    - batch_size: Training batch size
    - device: torch device (cuda/cpu)
    
    Returns:
    - Trained model
    - Training history dictionary
    """
    # Use GPU if available and not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model and data to device
    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    # Loss function
    def complex_matrix_mse_loss(pred, target):
        return torch.mean(torch.abs(pred - target)**2)

    def imag_mse_loss(pred, target):
        return torch.mean(torch.abs(pred.imag - target.imag)**2)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'learning_rate': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Shuffle training data
        indices = torch.randperm(train_x.size(0))
        train_x_shuffled = train_x[indices]
        train_y_shuffled = train_y[indices]
        
        # Batch training
        epoch_train_losses = []
        for i in range(0, train_x.size(0), batch_size):
            batch_x = train_x_shuffled[i:i+batch_size]
            batch_y = train_y_shuffled[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_x)
            # loss = complex_matrix_mse_loss(outputs, batch_y)
            loss = imag_mse_loss(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        # Compute test loss
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_x)
            test_loss = complex_matrix_mse_loss(test_outputs, test_y)
            imag_test_loss = imag_mse_loss(test_outputs, test_y)
        
        # Scheduler step
        scheduler.step(test_loss)
        
        # Record history
        avg_train_loss = torch.tensor(epoch_train_losses).mean().item()
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(test_loss.item())
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        if (epoch + 1) % 3 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Test Loss: {test_loss.item():.4f}')
            print(f'Imaginary Test Loss: {imag_test_loss.item():.4f}')
            print('-' * 40)
    
    return model, history


# Input: real-valued vector [30]
# Output: complex-valued matrix [33,5]
train_x, train_y, test_x, test_y = pull_train_test_data()


input_size = train_x.shape[1]
output_rows = train_y.shape[1]
output_cols = train_y.shape[2]
batch_size = 4
learning_rate = 0.001
num_epochs = 100

# Create model
model = ComplexTransformationModel(
    input_size=input_size, 
    output_size=(output_rows, output_cols)
)


trained_model, training_history = train_model(model,train_x,train_y,test_x,test_y)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(training_history['train_loss'], label='Train Loss')
plt.plot(training_history['test_loss'], label='Test Loss')
plt.title('Model Losses')
plt.legend()

plt.subplot(1,2,2)
plt.plot(training_history['learning_rate'])
plt.title('Learning Rate')
plt.tight_layout()
plt.show()






