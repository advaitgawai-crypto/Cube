import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from networks import CubeNet

torch.manual_seed(0)
# 1. Setup (Clean Baseline: no dropout, no weight decay)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

model = CubeNet(dropout_p=0.3).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

epochs = 50
best_val_loss = float('inf')
patience = 5
epochs_no_improve = 0

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# 2. Load data
with h5py.File("data/cross_data_v2.h5", "r") as f:
    states = torch.tensor(f["states"][:], dtype=torch.int64)
    distances = torch.tensor(f["distances"][:], dtype=torch.float32)
    groups = f["groups"][:]

distances = distances.unsqueeze(1)

# split by start (trajectory), so neighbors never land in both train and val
rng = np.random.default_rng(0)
ug = rng.permutation(np.unique(groups))
n = len(ug)
train_g, val_g, test_g = ug[:int(.8*n)], ug[int(.8*n):int(.9*n)], ug[int(.9*n):]
train_indices = torch.tensor(np.where(np.isin(groups, train_g))[0])
val_indices   = torch.tensor(np.where(np.isin(groups, val_g))[0])
test_indices  = torch.tensor(np.where(np.isin(groups, test_g))[0])

# Save validation and test indices
torch.save(val_indices, "models/cross_val_indices.pt")
torch.save(test_indices, "models/cross_test_indices.pt")

# Slice data into train, validation, and test sets
train_states = states[train_indices]
train_distances = distances[train_indices]

val_states = states[val_indices]
val_distances = distances[val_indices]

# Wrap into DataLoaders for batching
batch_size = 64
train_loader = DataLoader(TensorDataset(train_states, train_distances), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_states, val_distances), batch_size=batch_size, shuffle=False)

# 4. Training loop
for epoch in range(1, epochs + 1):
    model.train()
    train_loss_total = 0.0
    
    for batch_states, batch_distances in train_loader:
        batch_states = batch_states.to(device)
        batch_distances = batch_distances.to(device)
        
        optimizer.zero_grad()
        prediction = model(batch_states)
        loss = loss_fn(prediction, batch_distances)
        
        loss.backward()
        optimizer.step()
        
        train_loss_total += loss.item() * batch_states.size(0)
        
    avg_train_loss = train_loss_total / len(train_loader.dataset)

    model.eval()
    val_loss_total = 0.0
    
    with torch.no_grad():
        for batch_states, batch_distances in val_loader:
            batch_states = batch_states.to(device)
            batch_distances = batch_distances.to(device)
            
            prediction = model(batch_states)
            loss = loss_fn(prediction, batch_distances)
            val_loss_total += loss.item() * batch_states.size(0)
            
    val_loss = val_loss_total / len(val_loader.dataset)

    scheduler.step(val_loss)

    print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/cross_net_dropout03.pt")
        print(f"--> Saved new best model (Val Loss: {val_loss:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break