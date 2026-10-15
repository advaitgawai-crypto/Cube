from notations import Cube
import random as rd
import numpy as np
from collections import deque
import copy
from cube_utils import flatten_cube
import h5py
from oll_safe_algorithms import OLLSafe
import time 

OLLSafeMoves = ["algorithm_18","algorithm_19","algorithm_1", "algorithm_2", "algorithm_3", "algorithm_4", "algorithm_5", "algorithm_6", "algorithm_7", "algorithm_8", "algorithm_9", "algorithm_10", "algorithm_11", "algorithm_12", "algorithm_13", "algorithm_14", "algorithm_15", "algorithm_16", "algorithm_17"]

def walk_from(n):
    cube = Cube()
    cube.X()
    cube.X()
    algs = OLLSafe()
    samples = []
    step = 0
    for _ in range(n):
        step += 1
        getattr(algs, rd.choice(OLLSafeMoves))(cube)
        samples.append((flatten_cube(cube), step))
    return samples

all_samples = []
for _ in range(3400):
    all_samples.extend(walk_from(15))

states_list, distances_list = zip(*all_samples)
states = np.array(states_list)
distances = np.array(distances_list)

with h5py.File("data/pll_data.h5", "w") as f:
    f.create_dataset("states", data=states)
    f.create_dataset("distances", data=distances)

with h5py.File("data/pll_data.h5", "r") as f:
    print(f["states"].shape, f["distances"].shape)