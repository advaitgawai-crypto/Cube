import torch

sd = torch.load("models/pll_net_dropout0.pt", map_location="cpu")
for k, v in sd.items():
    print(k, tuple(v.shape))