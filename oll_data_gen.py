from notations import Cube
import random as rd
import numpy as np
from cube_utils import flatten_cube
import h5py
import time

MOVES = [
    "R", "R_Prime", "L", "L_Prime",
    "U", "U_Prime", "D", "D_Prime",
    "F", "F_Prime", "B", "B_Prime"
]

def oll_solved(cube: Cube):
    # If cross is on DOWN, OLL checks if the entire UPPER face is solid
    if (cube.upper == cube.upper[1, 1]).all():
        return True
    else:
        return False

def random_walk(n):
    cube = Cube()
    samples = []
    for step in range(1, n + 1):
        pick = rd.choice(MOVES)
        getattr(cube, pick)()
        samples.append((flatten_cube(cube), step))
    return samples

all_samples = []
start = time.perf_counter()

# 5,000 walks * 6 moves = 30,000 samples for OLL
for i, _ in enumerate(range(5000), start=1):
    all_samples.extend(random_walk(6))
    if i % 500 == 0:
        elapsed = time.perf_counter() - start
        left = elapsed / i * (5000 - i)
        print(f"{i}/5000 done, about {left:.1f}s left", flush=True)

states_list, distances_list = zip(*all_samples)
states = np.array(states_list)
distances = np.array(distances_list)

with h5py.File("data/oll_data.h5", "w") as f:
    f.create_dataset("states", data=states)
    f.create_dataset("distances", data=distances)

with h5py.File("data/oll_data.h5", "r") as f:
    print(f["states"].shape, f["distances"].shape)