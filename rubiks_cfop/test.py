import h5py
import torch
from networks import CubeNet

device = torch.device("cpu")

with h5py.File("data/pll_data_v2.h5", "r") as f:
    states = torch.tensor(f["states"][:], dtype=torch.int64)
    distances = torch.tensor(f["distances"][:], dtype=torch.float32)

model = CubeNet(dropout_p=0.0).to(device)
model.load_state_dict(torch.load("models/pll_net_dropout0.pt", map_location=device))
model.eval()

with torch.no_grad():
    preds = model(states.to(device)).squeeze(1).cpu()

exact = (torch.round(preds).long() == distances.long()).sum().item()
max_err = (preds - distances).abs().max().item()

print(f"Exact matches: {exact}/{len(states)}")
print(f"Max absolute error: {max_err:.4f}")