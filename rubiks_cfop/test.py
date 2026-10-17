import torch
val_indices = torch.load("models/cross_val_indices.pt")
test_indices = torch.load("models/cross_test_indices.pt")
print("val:", len(val_indices), "test:", len(test_indices))
print("overlap:", len(set(val_indices.tolist()) & set(test_indices.tolist())))