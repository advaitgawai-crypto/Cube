import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from networks import CubeNet

torch.manual_seed(0)

# --- settings (change only DROPOUT between experiments) ---
DROPOUT = 0.3
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 64
LR = 1e-3

# paths are relative to this file, so it works no matter where you run it from
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "oll_data_v2.h5")
MODELS_DIR = os.path.join(BASE, "models")
tag = f"dropout{int(round(DROPOUT * 100)):02d}"
MODEL_PATH = os.path.join(MODELS_DIR, f"oll_net_{tag}.pt")
VAL_IDX_PATH = os.path.join(MODELS_DIR, "oll_val_indices.pt")
TEST_IDX_PATH = os.path.join(MODELS_DIR, "oll_test_indices.pt")

# 1. Setup (CPU so runs are exactly repeatable)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

model = CubeNet(dropout_p=DROPOUT).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_loss = float('inf')
epochs_no_improve = 0

os.makedirs(MODELS_DIR, exist_ok=True)

# 2. Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Generate oll_data_v2.h5 first (needs states, distances, groups).")

with h5py.File(DATA_PATH, "r") as f:
    for key in ("states", "distances", "groups"):
        if key not in f:
            raise KeyError(f"'{key}' missing in {DATA_PATH}. Found: {list(f.keys())}")
    states = torch.tensor(f["states"][:], dtype=torch.int64)
    distances = torch.tensor(f["distances"][:], dtype=torch.float32)
    groups = f["groups"][:]

print(f"Loaded {len(states)} samples, {len(np.unique(groups))} groups, "
      f"labels {int(distances.min())}..{int(distances.max())}")

distances = distances.unsqueeze(1)

# split by walk (group), so neighbors never land in both train and val
rng = np.random.default_rng(0)
ug = rng.permutation(np.unique(groups))
n = len(ug)
train_g, val_g, test_g = ug[:int(.8*n)], ug[int(.8*n):int(.9*n)], ug[int(.9*n):]
train_indices = torch.tensor(np.where(np.isin(groups, train_g))[0])
val_indices   = torch.tensor(np.where(np.isin(groups, val_g))[0])
test_indices  = torch.tensor(np.where(np.isin(groups, test_g))[0])
print(f"train {len(train_indices)} | val {len(val_indices)} | test {len(test_indices)}")

# Save validation and test indices
torch.save(val_indices, VAL_IDX_PATH)
torch.save(test_indices, TEST_IDX_PATH)

# Slice data into train and validation sets (test set is NOT touched here)
train_states = states[train_indices]
train_distances = distances[train_indices]

val_states = states[val_indices]
val_distances = distances[val_indices]

train_loader = DataLoader(TensorDataset(train_states, train_distances), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_states, val_distances), batch_size=BATCH_SIZE, shuffle=False)

# 3. Training loop
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

print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {MODEL_PATH}")