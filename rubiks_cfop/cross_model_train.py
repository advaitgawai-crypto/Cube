import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from networks import CubeNet

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CubeNet(dropout_p=0.1).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Add Learning Rate Scheduler: cuts lr in half if val_loss doesn't improve for 3 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

epochs = 50
best_val_loss = float('inf')

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# 2. Load data
with h5py.File("data/cross_data.h5", "r") as f:
    states = torch.tensor(f["states"][:], dtype=torch.int64)
    distances = torch.tensor(f["distances"][:], dtype=torch.float32)

# Reshape distances to match model output shape: (60000, 1)
distances = distances.unsqueeze(1)

# 3. Split data
dataset_size = len(states)
indices = torch.randperm(dataset_size)

train_size = int(0.9 * dataset_size)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Save the validation indices to the models folder for testing consistency
torch.save(val_indices, "models/cross_val_indices.pt")

# Slice data into train and validation sets
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
    # Training phase
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

    # Validation phase
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

    # Update scheduler with the current validation loss
    scheduler.step(val_loss)

    print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/cross_net.pt")
        print(f"--> Saved new best model (Val Loss: {val_loss:.4f})")