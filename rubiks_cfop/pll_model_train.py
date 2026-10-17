import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from networks import CubeNet

torch.manual_seed(0)

#  settings 
DROPOUT = 0
EPOCHS = 1000
PATIENCE = 200
BATCH_SIZE = 64
LR = 1e-3
DATA_PATH = "data/pll_data_v2.h5"
MODEL_PATH = f"models/pll_net_dropout{DROPOUT}.pt"

# CPU keeps runs repeatable. Switch to CUDA only for final training if speed matters.
device = torch.device("cpu")
print(f"Using device: {device}")

os.makedirs("models", exist_ok=True)

model = CubeNet(dropout_p=DROPOUT).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

best_val_loss = float("inf")
epochs_no_improve = 0

#  load data 
with h5py.File(DATA_PATH, "r") as f:
    states = torch.tensor(f["states"][:], dtype=torch.int64)
    distances = torch.tensor(f["distances"][:], dtype=torch.float32)
    groups = f["groups"][:]

distances = distances.unsqueeze(1)
# Train on ALL states: the PLL space is closed and fully enumerated,
# so a held-out state would just be one the network was never taught.
all_indices = torch.arange(len(states))

train_indices = all_indices
val_indices = all_indices   # kept only so the loop still runs; same states as training
test_indices = torch.tensor([], dtype=torch.long)


torch.save(val_indices, "models/pll_val_indices.pt")
torch.save(test_indices, "models/pll_test_indices.pt")

train_states, train_distances = states[train_indices], distances[train_indices]
val_states, val_distances = states[val_indices], distances[val_indices]

train_loader = DataLoader(TensorDataset(train_states, train_distances), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_states, val_distances), batch_size=BATCH_SIZE, shuffle=False)

#  training loop 
for epoch in range(1, EPOCHS + 1):
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

    print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"--> Saved new best model (Val Loss: {val_loss:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break