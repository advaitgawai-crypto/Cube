import os
import sys
import pickle
import random as rd
from collections import deque

import h5py
import numpy as np

from notations import Cube
from cube_utils import flatten_cube
from cross_safe_algorithms import CrossSafe

# ---------------- settings ----------------
N_STARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000   # cross-solved starting cubes
WALK_LEN = 15                                                 # random face turns per start
OUT_PATH = "data/cross_data_v2.h5"
TABLE_PATH = "data/cross_table.pkl"
SEED = 0

# face turns only (quarter-turn metric): every move counts as 1
FACE_MOVES = ["R", "R_Prime", "L", "L_Prime", "U", "U_Prime",
              "D", "D_Prime", "F", "F_Prime", "B", "B_Prime"]
CrossSafeMoves = ["algorithm_1", "algorithm_2", "algorithm_3",
                  "algorithm_4", "algorithm_5", "algorithm_6"]

# The 12 edge slots. Each slot = the two stickers of one edge: (face, row, col).
EDGE_SLOTS = [
    (("upper", 2, 1), ("front", 0, 1)),   # U-F
    (("upper", 1, 2), ("right", 0, 1)),   # U-R
    (("upper", 0, 1), ("back", 0, 1)),    # U-B
    (("upper", 1, 0), ("left", 0, 1)),    # U-L
    (("front", 1, 2), ("right", 1, 0)),   # F-R
    (("front", 1, 0), ("left", 1, 2)),    # F-L
    (("back", 1, 0), ("right", 1, 2)),    # B-R
    (("back", 1, 2), ("left", 1, 0)),     # B-L
    (("lower", 0, 1), ("front", 2, 1)),   # D-F
    (("lower", 1, 2), ("right", 2, 1)),   # D-R
    (("lower", 2, 1), ("back", 2, 1)),    # D-B
    (("lower", 1, 0), ("left", 2, 1)),    # D-L
]


def copy_cube(c):
    n = Cube.__new__(Cube)
    n.upper = c.upper.copy()
    n.front = c.front.copy()
    n.lower = c.lower.copy()
    n.right = c.right.copy()
    n.left = c.left.copy()
    n.back = c.back.copy()
    return n


def cross_key(cube):
    """Describes only the 4 white (color 0) edges: where they are and how they are oriented."""
    key = []
    for (fa, ra, ca), (fb, rb, cb) in EDGE_SLOTS:
        a = int(getattr(cube, fa)[ra, ca])
        b = int(getattr(cube, fb)[rb, cb])
        key.append((a, b) if (a == 0 or b == 0) else (-1, -1))
    return tuple(key)


def build_table():
    """BFS from the solved cube. table[cross_key] = fewest face quarter-turns to solve the cross."""
    if os.path.exists(TABLE_PATH):
        with open(TABLE_PATH, "rb") as f:
            return pickle.load(f)
    solved = Cube()
    table = {cross_key(solved): 0}
    queue = deque([(solved, 0)])
    while queue:
        cube, d = queue.popleft()
        for name in FACE_MOVES:
            nxt = copy_cube(cube)
            getattr(nxt, name)()
            k = cross_key(nxt)
            if k not in table:
                table[k] = d + 1
                queue.append((nxt, d + 1))
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, "wb") as f:
        pickle.dump(table, f)
    return table


def normalize(cube):
    """Undo any whole-cube Y rotation so the centers are back in their standard places."""
    for _ in range(4):
        if cube.upper[1, 1] == 0 and cube.front[1, 1] == 1:
            return True
        cube.Y()
    return False


def make_cross_start(num_algs, solved_key):
    algs = CrossSafe()
    for _ in range(100):
        cube = Cube()
        for _ in range(num_algs):
            getattr(algs, rd.choice(CrossSafeMoves))(cube)
        if normalize(cube) and cross_key(cube) == solved_key:
            return cube
    raise RuntimeError("CrossSafe algorithms did not leave the cross solved")


def walk_from(start, table, n_steps):
    cube = copy_cube(start)
    samples = [(flatten_cube(cube), table[cross_key(cube)])]   # step 0: the start itself
    for _ in range(n_steps):
        getattr(cube, rd.choice(FACE_MOVES))()
        samples.append((flatten_cube(cube), table[cross_key(cube)]))
    return samples


if __name__ == "__main__":
    rd.seed(SEED)
    os.makedirs("data", exist_ok=True)

    print("Building / loading BFS table...")
    table = build_table()
    print(f"table size: {len(table)}, max distance: {max(table.values())}")
    solved_key = cross_key(Cube())

    seen = set()
    states, distances, groups = [], [], []
    dropped = 0
    for g in range(N_STARTS):
        start = make_cross_start(15, solved_key)
        for s, d in walk_from(start, table, WALK_LEN):
            kb = s.tobytes()
            if kb in seen:          # drop duplicate states
                dropped += 1
                continue
            seen.add(kb)
            states.append(s)
            distances.append(d)
            groups.append(g)        # which start (trajectory) this sample came from

    states = np.array(states, dtype=np.int8)
    distances = np.array(distances, dtype=np.int64)
    groups = np.array(groups, dtype=np.int32)

    with h5py.File(OUT_PATH, "w") as f:
        f.create_dataset("states", data=states)
        f.create_dataset("distances", data=distances)
        f.create_dataset("groups", data=groups)

    print(f"saved {len(states)} samples to {OUT_PATH} ({dropped} duplicates dropped)")
    print("distance counts:", dict(enumerate(np.bincount(distances).tolist())))