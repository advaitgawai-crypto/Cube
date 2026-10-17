import os
import random as rd
from collections import deque

import h5py
import numpy as np

from notations import Cube
from cube_utils import flatten_cube
from oll_safe_algorithms import OLLSafe

# ---------------- settings ----------------
N_WALKS = 2000
WALK_LEN = 15
SEED = 0
OUT_PATH = "data/pll_data_v2.h5"

# PLL algorithms only. algorithm_18 / 19 are Y / Y' (whole-cube rotations), so they are excluded.
PLL_ALGS = [f"algorithm_{i}" for i in range(1, 22)]
AUF_MOVES = ["U", "U_Prime", "U2"]
MOVES = PLL_ALGS + AUF_MOVES

ALGS = OLLSafe()


def copy_cube(c):
    n = Cube.__new__(Cube)
    n.upper = c.upper.copy()
    n.front = c.front.copy()
    n.lower = c.lower.copy()
    n.right = c.right.copy()
    n.left = c.left.copy()
    n.back = c.back.copy()
    return n


def normalize(cube):
    """Rotate the whole cube with Y until the front center is red (5).
    PLL frame: upper = yellow (2), front = red (5)."""
    for _ in range(4):
        if cube.front[1, 1] == 5:
            return
        cube.Y()
    raise RuntimeError("could not restore PLL orientation (upper center should be yellow)")


def pll_phase_ok(c):
    """OLL solved (upper face one color) and F2L solved (lower face one color,
    middle and bottom rows of the side faces match their centers)."""
    if not np.all(c.upper == c.upper[1, 1]):
        return False
    if not np.all(c.lower == c.lower[1, 1]):
        return False
    for face in [c.front, c.right, c.back, c.left]:
        if not np.all(face[1:, :] == face[1, 1]):
            return False
    return True


def apply_move(cube, name):
    if name == "U2":
        cube.U()
        cube.U()
    elif name in PLL_ALGS:
        getattr(ALGS, name)(cube)
    else:
        getattr(cube, name)()
    normalize(cube)


def make_solved():
    c = Cube()
    c.X()
    c.X()
    normalize(c)
    assert pll_phase_ok(c), "solved PLL cube failed the phase check"
    return c


def build_table():
    """BFS from the solved cube over the PLL macros and AUF moves.
    table[state bytes] = shortest number of moves to reach that state."""
    start = make_solved()
    table = {flatten_cube(start).tobytes(): 0}
    queue = deque([(start, 0)])
    while queue:
        cube, d = queue.popleft()
        for name in MOVES:
            nxt = copy_cube(cube)
            apply_move(nxt, name)
            assert pll_phase_ok(nxt), f"{name} broke OLL or F2L"
            key = flatten_cube(nxt).tobytes()
            if key not in table:
                table[key] = d + 1
                queue.append((nxt, d + 1))
    return table


if __name__ == "__main__":
    rd.seed(SEED)
    os.makedirs("data", exist_ok=True)

    print("Building BFS table...")
    table = build_table()
    print(f"table size: {len(table)}, max distance: {max(table.values())}")

    solved = make_solved()
    samples = {}   # state bytes -> (state, distance, group)

    for g in range(N_WALKS):
        cube = copy_cube(solved)
        for step in range(WALK_LEN + 1):
            if step > 0:
                apply_move(cube, rd.choice(MOVES))
            state = flatten_cube(cube)
            key = state.tobytes()
            if key not in samples:
                samples[key] = (state, table[key], g)

    vals = list(samples.values())
    states = np.array([v[0] for v in vals], dtype=np.int8)
    distances = np.array([v[1] for v in vals], dtype=np.int64)
    groups = np.array([v[2] for v in vals], dtype=np.int32)

    with h5py.File(OUT_PATH, "w") as f:
        f.create_dataset("states", data=states)
        f.create_dataset("distances", data=distances)
        f.create_dataset("groups", data=groups)

    print(f"saved {len(states)} samples to {OUT_PATH}")
    print("distance counts:", dict(enumerate(np.bincount(distances).tolist())))