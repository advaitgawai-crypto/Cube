from notations import Cube
import random as rd
import numpy as np    
from collections import deque
import copy
from cube_utils import flatten_cube
import h5py

MOVES = ["R","R_Prime","L","L_Prime","U","U_Prime","D","D_Prime","F","F_Prime","B","B_Prime","X","X_prime","Y","Y_prime","Z","Z_prime","M_prime","M","E_prime","E","S","S_prime"]
# cube = Cube()

def cross_solved(cube: Cube):
    if (
        cube.upper[0,1]==cube.upper[1,1] and
        cube.upper[1,0]==cube.upper[1,1] and
        cube.upper[1,2]==cube.upper[1,1] and
        cube.upper[2,1]==cube.upper[1,1] and
        cube.front[0,1]==cube.front[1,1] and
        cube.right[0,1]==cube.right[1,1] and
        cube.left[0,1] ==cube.left[1,1]  and
        cube.back[0,1] ==cube.back[1,1]
    ):
        return True
    else:
        return False

depth = 10

def scramble(depth):
    cube = Cube()
    for _ in range(depth):
        pick = rd.choice(MOVES)
        getattr(cube,pick)()

    cube.display()
    return cube

def cross_distance(start):
    line = deque()
    seen = set()
    seen.add(flatten_cube(start).tobytes())
    line.append((start, 0))
    while line:
        cube, steps = line.popleft()
        if cross_solved(cube):                    
            return steps             
        for name in MOVES:
            new = copy.deepcopy(cube)
            getattr(new, name)()
            key = flatten_cube(new).tobytes()
            if key not in seen:
                seen.add(key)
                line.append((new, steps+1))

def random_walk(n):
    cube = Cube()
    samples = []
    for step in range(1, n + 1):
        pick = rd.choice(MOVES)                      # which function picks a random move?
        getattr(cube, pick)()
        samples.append((flatten_cube(cube), step))     # what goes in the first slot?
    return samples



import time
all_samples = []
start = time.perf_counter()

for i, _ in enumerate(range(6250), start=1):
    all_samples.extend(random_walk(8))
    if i % 500 == 0:
        elapsed = time.perf_counter() - start
        left = elapsed / i * (6250 - i)                      # average time per walk * walks left
        print(f"{i}/6250 done, about {left:.1f}s left", flush=True)


states_list, distances_list = zip(*all_samples)
states = np.array(states_list)
distances = np.array(distances_list)

with h5py.File("data/cross_data.h5", "w") as f:
    f.create_dataset("states", data=states)
    f.create_dataset("distances", data=distances)

with h5py.File("data/cross_data.h5", "r") as f:
    print(f["states"].shape, f["distances"].shape)

