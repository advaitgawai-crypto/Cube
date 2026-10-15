from notations import Cube
from cube_utils import flatten_cube
import random as rd
import numpy as np
import h5py
import time
import os

# White UP (cross + F2L = top two layers), orange FRONT, yellow DOWN (last layer).
# Standard algs flipped over: U->D, D->U, F->B, B->F. They keep the top two layers intact.
def algorithm_1(cube: Cube):   # sune
    cube.R(); cube.D(); cube.R_Prime(); cube.D(); cube.R(); cube.D(); cube.D(); cube.R_Prime()
def algorithm_2(cube: Cube):   # sune inverse
    cube.R(); cube.D_Prime(); cube.D_Prime(); cube.R_Prime(); cube.D_Prime(); cube.R(); cube.D_Prime(); cube.R_Prime()
def algorithm_3(cube: Cube):   # edge flipper
    cube.B(); cube.R(); cube.D(); cube.R_Prime(); cube.D_Prime(); cube.B_Prime()
def algorithm_4(cube: Cube):   # edge flipper inverse
    cube.B(); cube.D(); cube.R(); cube.D_Prime(); cube.R_Prime(); cube.B_Prime()
def algorithm_5(cube: Cube):   # mirror of 3
    cube.B_Prime(); cube.L_Prime(); cube.D_Prime(); cube.L(); cube.D(); cube.B()
def algorithm_6(cube: Cube):   # mirror of 4
    cube.B_Prime(); cube.D_Prime(); cube.L_Prime(); cube.D(); cube.L(); cube.B()
def algorithm_7(cube: Cube):   cube.D()
def algorithm_8(cube: Cube):   cube.D_Prime()
# 9-10: T-perm and its inverse. They shuffle the last-layer pieces, which 1-8 can't do on their own
# (without them only 1/3 of last-layer states are reachable).
def algorithm_9(cube: Cube):
    cube.R(); cube.D(); cube.R_Prime(); cube.D_Prime(); cube.R_Prime(); cube.B(); cube.R(); cube.R()
    cube.D_Prime(); cube.R_Prime(); cube.D_Prime(); cube.R(); cube.D(); cube.R_Prime(); cube.B_Prime()
def algorithm_10(cube: Cube):
    cube.B(); cube.R(); cube.D_Prime(); cube.R_Prime(); cube.D(); cube.R(); cube.D(); cube.R(); cube.R()
    cube.B_Prime(); cube.R(); cube.D(); cube.R(); cube.D_Prime(); cube.R_Prime()

ALGS = [algorithm_1, algorithm_2, algorithm_3, algorithm_4, algorithm_5,
        algorithm_6, algorithm_7, algorithm_8, algorithm_9, algorithm_10]
# index of the algorithm that undoes each one
INVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8}


def f2l_solved(c: Cube):
    if not np.all(c.upper == c.upper[1, 1]):
        return False
    for face in [c.front, c.right, c.back, c.left]:
        if not np.all(face[0:2, :] == face[1, 1]):
            return False
    return True


def oll_solved(c: Cube):
    return f2l_solved(c) and np.all(c.lower == c.lower[1, 1])


def random_walk(n):
    cube = Cube()
    samples = []
    last = None
    for step in range(1, n + 1):
        choices = [i for i in range(len(ALGS)) if last is None or i != INVERSE[last]]
        pick = rd.choice(choices)
        ALGS[pick](cube)
        last = pick
        assert f2l_solved(cube), f"algorithm_{pick + 1} broke cross/F2L"
        dist = 0 if oll_solved(cube) else step
        samples.append((flatten_cube(cube), dist))
    return samples


if __name__ == "__main__":
    WALK_LEN = 10
    TARGET = 30000
    MAX_WALKS = 200_000     # hard stop so the loop can never run forever
    best = {}

    solved = flatten_cube(Cube())
    best[solved.tobytes()] = (solved, 0)

    start = time.perf_counter()
    walks = 0
    while len(best) < TARGET and walks < MAX_WALKS:
        for state, d in random_walk(WALK_LEN):
            key = state.tobytes()
            if key not in best or d < best[key][1]:
                best[key] = (state, d)
        walks += 1
        if walks % 500 == 0:
            elapsed = time.perf_counter() - start
            print(f"{walks} walks, {len(best)}/{TARGET} unique, {elapsed:.0f}s", flush=True)

    if len(best) < TARGET:
        print(f"WARNING: stopped at MAX_WALKS with only {len(best)} unique states (target {TARGET})")

    states = np.array([v[0] for v in best.values()])
    distances = np.array([v[1] for v in best.values()])

    # save next to this script, and create the data/ folder if it doesn't exist
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "oll_data.h5")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("states", data=states)
        f.create_dataset("distances", data=distances)

    with h5py.File(out_path, "r") as f:
        print(f["states"].shape, f["distances"].shape)
    print("saved to:", out_path)
    print("distance counts:", dict(zip(*np.unique(distances, return_counts=True))))