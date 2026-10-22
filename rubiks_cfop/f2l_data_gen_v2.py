from notations import Cube
from cube_utils import flatten_cube
import random as rd
import numpy as np
import h5py
import time
import os

# ------------------------------------------------------------------
# F2L data v2: TRUE quarter-turn distances (12 face turns: R L U D F B and primes).
# Goal = F2L solved = top two layers solved (cross + F2L), last layer can be anything.
# White UP, orange FRONT.
#
# How labels work: a BFS from the solved cube builds an exact distance table up to
# TABLE_DEPTH moves. Distance only depends on the 12 pieces of the top two layers, so the
# table is keyed on those pieces only. A sample's label is looked up in the table:
#   found  -> exact true distance (0..TABLE_DEPTH)
#   absent -> CAP = TABLE_DEPTH + 1, meaning "at least TABLE_DEPTH + 1 moves away"
# ------------------------------------------------------------------

TABLE_DEPTH = 7
CAP = TABLE_DEPTH + 1

FACE_MOVES = ["R", "R_Prime", "L", "L_Prime", "U", "U_Prime",
              "D", "D_Prime", "F", "F_Prime", "B", "B_Prime"]      # inverse of i is i ^ 1

# cross-safe macros (cross stays solved). Used so some samples look like real
# "cross solved, F2L messy" cubes. Cost = number of quarter turns in the macro.
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
def algorithm_11(cube: Cube): cube.F_Prime(); cube.R_Prime(); cube.D(); cube.R(); cube.F()
def algorithm_12(cube: Cube): cube.F_Prime(); cube.R_Prime(); cube.D_Prime(); cube.R(); cube.F()
def algorithm_13(cube: Cube): cube.F(); cube.R(); cube.D(); cube.R_Prime(); cube.F_Prime()
def algorithm_14(cube: Cube): cube.F(); cube.R(); cube.D_Prime(); cube.R_Prime(); cube.F_Prime()

MACROS = [algorithm_1, algorithm_2, algorithm_3, algorithm_4, algorithm_5, algorithm_6, algorithm_7,
          algorithm_8, algorithm_9, algorithm_10, algorithm_11, algorithm_12, algorithm_13, algorithm_14]
MACRO_COST = [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 5, 5, 5, 5]
MACRO_INVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10, 12: 13, 13: 12}


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


# ---------------- sticker engine (built from YOUR notations.py) ----------------
FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]   # same order as Cube() colors 0..5

def labelled_cube():
    c = Cube()
    for i, name in enumerate(FACE_ORDER):
        setattr(c, name, np.arange(9).reshape(3, 3) + 9 * i)
    return c

def flat54(c):
    return np.concatenate([getattr(c, n).flatten() for n in FACE_ORDER]).astype(np.int64)

def perm_of(apply_fn):
    """apply_fn(cube) -> permutation F with new_state = old_state[F]"""
    c = labelled_cube()
    apply_fn(c)
    return flat54(c)

FACE_PERMS = [perm_of(lambda c, m=m: getattr(c, m)()) for m in FACE_MOVES]
MACRO_PERMS = [perm_of(m) for m in MACROS]
FACE_INV = [np.argsort(p) for p in FACE_PERMS]
MACRO_INV = [np.argsort(p) for p in MACRO_PERMS]

# stickers of the top two layers: upper face + top two rows of the four side faces
def _sid(face, r, c): return 9 * face + 3 * r + c
TRACKED = [_sid(0, r, c) for r in range(3) for c in range(3)]
for _face in (1, 3, 4, 5):
    TRACKED += [_sid(_face, r, c) for r in range(2) for c in range(3)]
TRACKED = np.array(TRACKED)
_HASH_MUL = np.random.default_rng(12345).integers(1, 2**62, size=len(TRACKED), dtype=np.uint64) | np.uint64(1)


def hash_rows(q):
    """64-bit hash of tracked-sticker positions. q: (n, 33) array."""
    return (q.astype(np.uint64) * _HASH_MUL).sum(axis=1)


def build_table(depth):
    """Exact BFS from the solved cube. Returns (sorted hashes, distances)."""
    q0 = TRACKED.astype(np.uint8)[None, :]
    visited = hash_rows(q0)
    level_hashes = [visited.copy()]
    frontier = q0
    inv = [p.astype(np.uint8) for p in FACE_INV]
    for d in range(1, depth + 1):
        vis_sorted = np.sort(visited)
        rows, hashes = [], []
        for s in range(0, len(frontier), 40000):
            chunk = frontier[s:s + 40000]
            cand = np.concatenate([p[chunk] for p in inv])
            hh = hash_rows(cand)
            pos = np.searchsorted(vis_sorted, hh)
            pos[pos == len(vis_sorted)] = 0
            keep = vis_sorted[pos] != hh
            rows.append(cand[keep]); hashes.append(hh[keep])
        rows = np.concatenate(rows); hashes = np.concatenate(hashes)
        hashes_u, first = np.unique(hashes, return_index=True)
        frontier = rows[first]
        level_hashes.append(hashes_u)
        visited = np.concatenate([visited, hashes_u])
        print(f"  table depth {d}: {len(hashes_u)} new states", flush=True)
    all_h = np.concatenate(level_hashes)
    all_d = np.concatenate([np.full(len(h), d, dtype=np.uint8) for d, h in enumerate(level_hashes)])
    order = np.argsort(all_h)
    return all_h[order], all_d[order]


def load_or_build_table(path, depth):
    if os.path.exists(path):
        z = np.load(path)
        if int(z["depth"]) == depth:
            print(f"Loaded table from {path} ({len(z['hashes'])} states)")
            return z["hashes"], z["dist"]
    print(f"Building exact distance table to depth {depth} ...")
    t = time.perf_counter()
    h, d = build_table(depth)
    np.savez(path, hashes=h, dist=d, depth=depth)
    print(f"Table built: {len(h)} states in {time.perf_counter() - t:.0f}s, saved to {path}")
    return h, d


def lookup(table_h, table_d, q_tracked):
    hq = hash_rows(q_tracked[None, :])[0]
    i = np.searchsorted(table_h, hq)
    if i < len(table_h) and table_h[i] == hq:
        return int(table_d[i])
    return CAP


def state_to_cube(q):
    """q = position of each sticker identity. Returns a real Cube object with colors."""
    s = np.empty(54, dtype=np.int64)
    s[q] = np.arange(54)             # s[pos] = identity at pos
    colors = s // 9
    c = Cube()
    for i, name in enumerate(FACE_ORDER):
        setattr(c, name, colors[9 * i: 9 * i + 9].reshape(3, 3).copy())
    return c


def apply_perm(q, inv):
    return inv[q]


def face_walk(n, table_h, table_d):
    q = np.arange(54)
    out, last = [], None
    for step in range(1, n + 1):
        choices = [i for i in range(12) if last is None or i != (last ^ 1)]
        m = rd.choice(choices)
        q = apply_perm(q, FACE_INV[m]); last = m
        out.append((q.copy(), lookup(table_h, table_d, q[TRACKED]), step, False))
    return out


def macro_walk(n, table_h, table_d):
    q = np.arange(54)
    out, last, cost = [], None, 0
    for _ in range(n):
        choices = [i for i in range(len(MACROS)) if last is None or i != MACRO_INVERSE[last]]
        m = rd.choice(choices)
        q = apply_perm(q, MACRO_INV[m]); last = m; cost += MACRO_COST[m]
        out.append((q.copy(), lookup(table_h, table_d, q[TRACKED]), cost, True))
    return out


def self_check(table_h, table_d):
    """Engine must match the real Cube class, and labels must agree with f2l_solved / cross_ok."""
    for _ in range(300):
        real = Cube(); q = np.arange(54)
        for _ in range(rd.randint(1, 12)):
            if rd.random() < 0.5:
                m = rd.randrange(12); getattr(real, FACE_MOVES[m])(); q = apply_perm(q, FACE_INV[m])
            else:
                m = rd.randrange(len(MACROS)); MACROS[m](real); q = apply_perm(q, MACRO_INV[m])
        assert (flatten_cube(real) == flatten_cube(state_to_cube(q))).all(), "engine != Cube class"
        label = lookup(table_h, table_d, q[TRACKED])
        assert (label == 0) == f2l_solved(real), "label 0 disagrees with f2l_solved"
    print("self-check passed: engine matches Cube class, label 0 <=> f2l_solved")


if __name__ == "__main__":
    TARGET_EXACT = 100_000      # samples with an exact (non-capped) label
    MAX_CAPPED_FRACTION = 0.15  # capped samples (label = CAP) kept at most this share of exact ones
    FACE_WALK_LEN = 10          # face-turn walk length
    MACRO_WALK_LEN = 6          # cross-safe macro walk length
    MAX_WALKS = 400_000         # hard stop so the loop can never run forever

    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "data")
    os.makedirs(out_dir, exist_ok=True)
    table_h, table_d = load_or_build_table(os.path.join(out_dir, f"f2l_table_d{TABLE_DEPTH}.npz"), TABLE_DEPTH)
    self_check(table_h, table_d)

    seen = {}                    # state bytes -> (flat state, label, group)
    exact_count = capped_count = walks = 0
    start = time.perf_counter()
    while exact_count < TARGET_EXACT and walks < MAX_WALKS:
        walks += 1
        samples = face_walk(FACE_WALK_LEN, table_h, table_d) if walks % 2 else macro_walk(MACRO_WALK_LEN, table_h, table_d)
        for q, label, cost, is_macro in samples:
            key = q.tobytes()
            if key in seen:
                continue
            capped = (label == CAP)
            if capped and capped_count >= MAX_CAPPED_FRACTION * max(exact_count, 1):
                continue
            cube = state_to_cube(q)
            assert (label == 0) == f2l_solved(cube), "label 0 must be exactly the solved-F2L state"
            if not capped:
                assert label <= cost, "label can never exceed the moves used to reach the state"
            if is_macro:
                assert cross_ok(cube), "macro broke the cross"
            seen[key] = (flatten_cube(cube), label, walks)
            if capped: capped_count += 1
            else: exact_count += 1
        if walks % 2000 == 0:
            print(f"{walks} walks, exact {exact_count}/{TARGET_EXACT}, capped {capped_count}, "
                  f"{time.perf_counter() - start:.0f}s", flush=True)

    if exact_count < TARGET_EXACT:
        print(f"WARNING: stopped at MAX_WALKS with only {exact_count} exact samples (target {TARGET_EXACT})")

    states = np.array([v[0] for v in seen.values()], dtype=np.int8)
    distances = np.array([v[1] for v in seen.values()], dtype=np.int64)
    groups = np.array([v[2] for v in seen.values()], dtype=np.int32)

    out_path = os.path.join(out_dir, "f2l_data_v2.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("states", data=states)
        f.create_dataset("distances", data=distances)
        f.create_dataset("groups", data=groups)
        f.attrs["metric"] = "quarter-turn face moves (R L U D F B and primes)"
        f.attrs["cap_label"] = CAP
        f.attrs["table_depth"] = TABLE_DEPTH

    with h5py.File(out_path, "r") as f:
        print(f["states"].shape, f["distances"].shape, f["groups"].shape)
    print("saved to:", out_path)
    print(f"label {CAP} means 'at least {CAP} moves away' (table depth {TABLE_DEPTH})")
    print("distance counts:", dict(zip(*np.unique(distances, return_counts=True))))