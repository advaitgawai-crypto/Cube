import h5py
import numpy as np

with h5py.File("data/cross_data.h5", "r") as f:
    distances = f["distances"][:]

print("min:", distances.min())
print("max:", distances.max())
print("mean:", distances.mean())

# Baseline: MSE if you always predict the mean (which is equivalent to the variance of the distances)
baseline_mse = np.mean((distances - distances.mean()) ** 2)
print("baseline MSE (variance):", baseline_mse)