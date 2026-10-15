from notations import Cube
import random as rd
import numpy as np
from collections import deque
import copy
from cube_utils import flatten_cube
import h5py
from cross_safe_algorithms import CrossSafe
import time

RandomMoves = ["R","R_Prime","L","L_Prime","U","U_Prime","D","D_Prime","F","F_Prime","B","B_Prime","M_prime","M","E_prime","E","S","S_prime"]
CrossSafeMoves = ["algorithm_1","algorithm_2","algorithm_3","algorithm_4","algorithm_5", "algorithm_6"]

def make_cross_start(num_algs):
    algs = CrossSafe()
    cube = Cube()
    for _ in range(num_algs):
        name = rd.choice(CrossSafeMoves)
        getattr(algs, name)(cube)
    return cube

def walk_from(start, n_cross, n_full):
    cube = copy.deepcopy(start)
    algs = CrossSafe()
    samples = []
    step = 0
    for _ in range(n_cross):
        step += 1
        name = rd.choice(CrossSafeMoves)
        getattr(algs, name)(cube)
        samples.append((flatten_cube(cube), step))
    for _ in range(n_full):
        step += 1
        getattr(cube, rd.choice(RandomMoves))()
        samples.append((flatten_cube(cube), step))
    return samples

starts = []
for _ in range(4000):
    starts.append(make_cross_start(15))

all_samples = []
for start in starts:
    all_samples.extend(walk_from(start, 0, 15))

states_list, distances_list = zip(*all_samples)
states = np.array(states_list)
distances = np.array(distances_list)

with h5py.File("data/cross_data.h5", "w") as f:
    f.create_dataset("states", data=states)
    f.create_dataset("distances", data=distances)

with h5py.File("data/cross_data.h5", "r") as f:
    print(f["states"].shape, f["distances"].shape)