import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from networks import CubeNet

device = torch.device("cpu")

model = CubeNet(dropout_p=0.3).to(device)
model.load_state_dict(torch.load("models/cross_net_dropout03.pt", map_location=device))
model.eval()

val_indices = torch.load("models/cross_val_indices.pt")

with h5py.File("data/cross_data_v2.h5", "r") as f:
    states = torch.tensor(f["states"][:], dtype=torch.int64)
    distances = torch.tensor(f["distances"][:], dtype=torch.float32)

distances = distances.unsqueeze(1)

val_states = states[val_indices]
val_distances = distances[val_indices]

val_loader = DataLoader(
    TensorDataset(val_states, val_distances),
    batch_size=64,
    shuffle=False
)

loss_fn = nn.MSELoss()
val_loss_total = 0.0

with torch.no_grad():
    for batch_states, batch_distances in val_loader:
        batch_states = batch_states.to(device)
        batch_distances = batch_distances.to(device)

        prediction = model(batch_states)
        loss = loss_fn(prediction, batch_distances)
        val_loss_total += loss.item() * batch_states.size(0)

avg_val_loss = val_loss_total / len(val_loader.dataset)

print(f"Validation MSE: {avg_val_loss:.4f}")