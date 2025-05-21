################################################################
### Import Library
################################################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from timeit import default_timer
import torch.nn.functional as F
import pyekfmm as fmm
from model import *

# Choose GPU:0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


################################################################
### Network Training Function 
################################################################
def count_params(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, train_loader, valid_loader, epochs, optimizer, scheduler, criterion, device, ntrain, nvalid, batch_size):
    model.train()
    losses = {'train': [], 'valid': []}
    best_valid_loss = float('inf')  # Initialize the best validation loss to a very high value
    best_model_state = None  # To store the best model's state dictionary

    for epoch in range(epochs):
        # Training phase
        t1 = default_timer()
        train_loss = 0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels= batch_inputs.float(), batch_labels.float()
            optimizer.zero_grad()
            batch_outputs = model(batch_inputs)
            loss = criterion(batch_outputs.view(batch_size, -1), batch_labels.view(batch_size, -1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        
        # Validation phase
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.float(), labels.float()
                outputs = model(inputs)
                loss = criterion(outputs.view(batch_size, -1), labels.view(batch_size, -1))
                valid_loss += loss.item()
        model.train()
        
        # Average losses
        train_loss /= ntrain
        valid_loss /= nvalid
        
        losses['train'].append(train_loss)
        losses['valid'].append(valid_loss)
        t2 = default_timer()
        
        # Print losses very 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.3e}, Val Loss: {valid_loss:.3e}, Time: {t2 - t1:.3f} s")

    return model, losses

######################################################################################
### Data Generation  Pyekfmm: https://github.com/aaspip/pyekfmm/tree/main/pyekfmm
######################################################################################
# Hyperparameters
n_sources = 100  # Number of sources
nz, nx = 240, 737  # Grid dimensions
dx, dz = 0.0125, 0.0125  # Grid spacing in km

# Initialize data containers
T0_data_all = np.zeros((n_sources, nz, nx), dtype='float32')
T_data_all = np.zeros((n_sources, nz, nx), dtype='float32')
vel_data_all = np.zeros((n_sources, nz, nx), dtype='float32')
src_data_all = np.zeros((n_sources, nz, nx), dtype='float32')

# Load velocity model
try:
    with open('./vel_model/marmousi/marmvz.bin', 'rb') as fd:
        velzz = np.fromfile(fd, dtype=np.float32).reshape([nz, nx], order='F')  # [z, x]
    velz = velzz / 1000  # Convert velocity to km/s
    print(f"Velocity model loaded. Shape: {velz.shape}")
except FileNotFoundError:
    print("Error: Velocity model file not found. Using random data for debugging.")
    velz = np.random.rand(nz, nx) * 3 + 1  # Random velocity between 1 and 4 km/s

# Generate random sources
np.random.seed(202506)  # For reproducibility
source_locs = np.random.randint(0, [nx, nz], size=(n_sources, 2)).astype('float32')  # [x, z]

# Convert to physical coordinates (km)
source_locs_physical = source_locs * [dx, dz]


# Calculate traveltime fields for each source
for src_idx, (sx, sz) in enumerate(source_locs):
    sx, sz = int(sx), int(sz)  # Convert to integer indices
    src_physical = np.array([sz * dz, 0, sx * dx], dtype='float32')  # [x, y, z] in km

    # Populate velocity and source data
    vel_data_all[src_idx] = velz
    src_data_all[src_idx, sz, sx] = 1

    # Heterogeneous velocity computation
    velz_1d = velz.transpose().flatten(order='F')  # [z, x] -> [x, z]
    t = fmm.eikonal(
        velz_1d,
        xyz=src_physical,  # [x, y, z]
        ax=[0, dz, nz],  # x-axis
        ay=[0, dx, 1],   # y-axis (not used in 2D)
        az=[0, dx, nx],  # z-axis
        order=2
    )
    T_data_all[src_idx] = t.reshape(nz, nx, order='F')  # Reshape to [z, x]

    # Homogeneous velocity computation
    vel_homo = velz[sz, sx] * np.ones([nz * nx], dtype='float32')
    # print(vel_homo.shape)
    t1 = fmm.eikonal(
        vel_homo,
        xyz=src_physical,  # [x, y, z]
        ax=[0, dz, nz],  # x-axis
        ay=[0, dx, 1],   # y-axis (not used in 2D)
        az=[0, dx, nx],  # z-axis
        order=2
    )
    T0_data_all[src_idx] = t1.reshape(nz, nx, order='F')  # Reshape to [z, x]

# Compute Tau data
Tau_data_all = T_data_all - T0_data_all  # Shape: (n_sources, nz, nx)

# Print final shapes
print(f"T0_data_all: {T0_data_all.shape}")
print(f"T_data_all: {T_data_all.shape}")
print(f"vel_data_all: {vel_data_all.shape}")
print(f"src_data_all: {src_data_all.shape}")
print(f"Tau_data_all: {Tau_data_all.shape}")

################################################################
### Training dataset splitting
################################################################
batch_size = 4
ntrain = 84
nvalid = n_sources - ntrain 

# Convert to tensors
inputs = torch.tensor(T0_data_all.reshape(T0_data_all.shape[0], T0_data_all.shape[1], T0_data_all.shape[2], 1), 
                      dtype=torch.float, requires_grad=True, device=device)
labels = torch.tensor(Tau_data_all, dtype=torch.float, requires_grad=True, device=device)

train_inputs = inputs[:ntrain, :, :, :]
val_inputs = inputs[ntrain:, :, :, :]
train_labels = labels[:ntrain, :, :]
val_labels = labels[ntrain:, :, :]

 
# Create DataLoaders
train_dataset = TensorDataset(train_inputs, train_labels)
valid_dataset = TensorDataset(val_inputs, val_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


################################################################
### Network training 
################################################################
learning_rate = 1.5e-3
epochs = 3000
modes1, modes2 = 8, 8
width = 8
scheduler_step = epochs//5
scheduler_gamma = 0.5
print(epochs, learning_rate, scheduler_step, scheduler_gamma)


# Model, optimizer, and scheduler
model = KAGFNO2d(modes1=modes1, modes2=modes2, width=width).to(device)
print(f"Model parameters: {count_params(model):,}")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * (ntrain // batch_size))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
criterion = nn.MSELoss(reduction='sum')
# criterion = LpLoss(size_average=False)

# Training
start_time = default_timer()
model, losses = train(model, train_loader, valid_loader, epochs, optimizer, scheduler, criterion, device, ntrain, nvalid, batch_size)
# model, losses = train(model, train_loader, epochs, optimizer, scheduler, criterion, device, ntrain, batch_size)
elapsed = default_timer() - start_time
print(f"[Training] Finished. Time: {elapsed:.2f}s ({elapsed / 3600:.2f}h)")

# Save model
save_dir = "./model/Marmousi/KAGFNO/"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"KAGFNO_marmousi_ep{epochs}_modes{modes1}{modes2}_width{width}_0519.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}")
model.eval()


################################################################
### Loss curve visualization
################################################################
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), losses['train'], label='Training Loss', color='blue')
plt.plot(range(1, epochs + 1), losses['valid'], label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Optional: Log scale if losses vary widely
plt.yscale('log')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()