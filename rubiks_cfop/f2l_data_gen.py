from notations import Cube
from cube_utils import flatten_cube
import random as rd
import numpy as np
import h5py
import time

# white UP, orange FRONT. Every move below leaves the cross intact.
def algorithm_1(cube: Cube):  cube.R_Prime(); cube.D(); cube.R()
def algorithm_2(cube: Cube):  cube.R_Prime(); cube.D_Prime(); cube.R()
def algorithm_3(cube: Cube):  cube.L_Prime(); cube.D(); cube.L()
def algorithm_4(cube: Cube):  cube.L_Prime(); cube.D_Prime(); cube.L()
def algorithm_5(cube: Cube):  cube.F_Prime(); cube.D(); cube.F()
def algorithm_6(cube: Cube):  cube.F_Prime(); cube.D_Prime(); cube.F()
def algorithm_7(cube: Cube):  cube.B_Prime(); cube.D(); cube.B()
def algorithm_8(cube: Cube):  cube.B_Prime(); cube.D_Prime(); cube.B()
def algorithm_9(cube: Cube):  cube.D()
def algorithm_10(cube: Cube): cube.D_Prime()

ALGS = [algorithm_1, algorithm_2, algorithm_3, algorithm_4, algorithm_5,
        algorithm_6, algorithm_7, algorithm_8, algorithm_9, algorithm_10]
# index of the algorithm that undoes each one
INVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8}


def cross_ok(c: Cube):
    u = c.upper
    return (u[0,1] == u[1,1] and u[1,0] == u[1,1] and u[1,2] == u[1,1] and u[2,1] == u[1,1]
            and c.front[0,1] == c.front[1,1] and c.right[0,1] == c.right[1,1]
            and c.left[0,1]  == c.left[1,1]  and c.back[0,1]  == c.back[1,1])


def f2l_solved(c: Cube):
    if not np.all(c.upper == c.upper[1, 1]):
        return False
    for face in [c.front, c.right, c.back, c.left]:
        if not np.all(face[0:2, :] == face[1, 1]):
            return False
    return True


def random_walk(n):
    cube = Cube()
    samples = []
    last = None
    for step in range(1, n + 1):
        choices = [i for i in range(len(ALGS)) if last is None or i != INVERSE[last]]
        pick = rd.choice(choices)
        ALGS[pick](cube)
        last = pick
        assert cross_ok(cube), f"algorithm_{pick + 1} broke the cross"
        dist = 0 if f2l_solved(cube) else step
        samples.append((flatten_cube(cube), dist))
    return samples


if __name__ == "__main__":
    WALK_LEN = 15
    TARGET = 50000          # unique states wanted
    best = {}               # state bytes -> (state, smallest distance seen)

    solved = flatten_cube(Cube())
    best[solved.tobytes()] = (solved, 0)

    start = time.perf_counter()
    walks = 0
    while len(best) < TARGET:
        for state, d in random_walk(WALK_LEN):
            key = state.tobytes()
            if key not in best or d < best[key][1]:
                best[key] = (state, d)
        walks += 1
        if walks % 500 == 0:
            elapsed = time.perf_counter() - start
            print(f"{walks} walks, {len(best)}/{TARGET} unique, {elapsed:.0f}s", flush=True)

    states = np.array([v[0] for v in best.values()])
    distances = np.array([v[1] for v in best.values()])

    with h5py.File("data/f2l_data.h5", "w") as f:
        f.create_dataset("states", data=states)
        f.create_dataset("distances", data=distances)

    with h5py.File("data/f2l_data.h5", "r") as f:
        print(f["states"].shape, f["distances"].shape)
    print("distance counts:", dict(zip(*np.unique(distances, return_counts=True))))