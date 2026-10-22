import numpy as np
from notations import Cube
from f2l_algorithms import F2L_ALGORITHMS
from f2l_data_gen_v3 import apply_sequence, invert_token

# Whole-F2L solver. Cube orientation is the same as your v3 generator: start from a solved cube,
# apply X2 (cross on the BOTTOM), then your scramble. The solver uses your 41 algorithms (in all
# 4 slot rotations) plus U turns as its moves and fixes the 4 pairs one after another.

FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]


def _labelled():
    c = Cube()
    for i, name in enumerate(FACE_ORDER):
        setattr(c, name, np.arange(9).reshape(3, 3) + 9 * i)
    return c


def perm_of(seq):
    """Permutation F such that new_state = old_state[F] after applying the move string."""
    c = _labelled()
    apply_sequence(c, seq)
    return np.concatenate([getattr(c, n).flatten() for n in FACE_ORDER]).astype(np.int64)


S0 = perm_of("X2")                                  # solved cube after X2
C0 = S0 // 9                                        # colour that belongs on every position
CENTERS = [4, 13, 22, 31, 40, 49]

def _p(face, r, c):
    return 9 * FACE_ORDER.index(face) + 3 * r + c

SLOTS = {                                           # corner + edge stickers of each slot (cross on bottom)
    "front-right": [_p("lower", 0, 2), _p("front", 2, 2), _p("right", 2, 0), _p("front", 1, 2), _p("right", 1, 0)],
    "front-left":  [_p("lower", 0, 0), _p("front", 2, 0), _p("left", 2, 2),  _p("front", 1, 0), _p("left", 1, 2)],
    "back-right":  [_p("lower", 2, 2), _p("back", 2, 0),  _p("right", 2, 2), _p("back", 1, 0),  _p("right", 1, 2)],
    "back-left":   [_p("lower", 2, 0), _p("back", 2, 2),  _p("left", 2, 0),  _p("back", 1, 2),  _p("left", 1, 0)],
}
SLOT_NAMES = list(SLOTS)
CROSS = [_p("lower", 0, 1), _p("front", 2, 1), _p("lower", 1, 2), _p("right", 2, 1),
         _p("lower", 2, 1), _p("back", 2, 1),  _p("lower", 1, 0), _p("left", 2, 1)]


def _ok(states, positions):
    """True for every row of `states` whose stickers at `positions` have the right colours."""
    return (states[:, positions] // 9 == C0[positions]).all(axis=1)


def build_macros():
    y = perm_of("y")
    seen, macros = {}, []
    def add(text, f1):
        f = f1
        k = 0
        while not ((S0[f] // 9)[CENTERS] == C0[CENTERS]).all():
            f = f[y]                                 # compose with one more y rotation
            k += 1
            if k > 3:
                return
        text = text + " y" * k
        key = f.tobytes()
        if key not in seen or len(text) < len(seen[key][0]):
            seen[key] = (text, f)
    for t in ["U", "U'", "U2"]:
        add(t, perm_of(t))
    for r in ["", "y", "y'", "y2"]:
        for name in sorted(F2L_ALGORITHMS):
            parts = [r, F2L_ALGORITHMS[name], invert_token(r) if r else ""]
            text = " ".join(p for p in parts if p)
            add(text, perm_of(text))
    return list(seen.values())

MACROS = build_macros()
MAC_TEXT = [m[0] for m in MACROS]
MAC_PERM = np.array([m[1] for m in MACROS])
MAC_LEN = np.array([len(t.split()) for t in MAC_TEXT])
_HM = np.random.default_rng(5).integers(1, 2**62, size=54, dtype=np.uint64) | np.uint64(1)


def solved_slots(S):
    return [bool(_ok(S[None, :], SLOTS[n])[0]) for n in SLOT_NAMES]

def cross_ok(S):
    return bool(_ok(S[None, :], CROSS)[0])


def _search(S, prev_solved, max_depth):
    """Breadth-first over macros until one more slot is solved without breaking the cross or earlier slots."""
    keep_pos = list(CROSS)
    for n, s in zip(SLOT_NAMES, prev_solved):
        if s:
            keep_pos += SLOTS[n]
    new_slot_pos = [SLOTS[n] for n, s in zip(SLOT_NAMES, prev_solved) if not s]
    level = S[None, :].astype(np.uint8)
    history = []                                      # per level: (parent index, macro index)
    seen = {int((S.astype(np.uint64) * _HM).sum())}
    for depth in range(1, max_depth + 1):
        nxt_states, nxt_par, nxt_mac = [], [], []
        best = None
        for start in range(0, len(level), 4000):
            chunk = level[start:start + 4000]
            cand = chunk[:, MAC_PERM].reshape(-1, 54)               # every macro on every state
            par = np.repeat(np.arange(start, start + len(chunk)), len(MAC_PERM))
            mac = np.tile(np.arange(len(MAC_PERM)), len(chunk))
            good = _ok(cand, keep_pos)
            cand, par, mac = cand[good], par[good], mac[good]
            if len(cand) == 0:
                continue
            for pos in new_slot_pos:                                # did we finish a new slot?
                hit = np.where(_ok(cand, pos))[0]
                for h in hit:
                    cost = MAC_LEN[mac[h]]
                    if best is None or cost < best[0]:
                        best = (cost, par[h], mac[h])
            hh = (cand.astype(np.uint64) * _HM).sum(axis=1)
            u, first = np.unique(hh, return_index=True)
            fresh = np.array([int(x) not in seen for x in u])
            idx = first[fresh]
            seen.update(int(x) for x in u[fresh])
            nxt_states.append(cand[idx]); nxt_par.append(par[idx]); nxt_mac.append(mac[idx])
        if best is not None:
            seq = [best[2]]
            p = best[1]
            for d in range(depth - 1, 0, -1):
                par_arr, mac_arr = history[d - 1]
                seq.append(mac_arr[p]); p = par_arr[p]
            return seq[::-1]
        if not nxt_states:
            return None
        level = np.concatenate(nxt_states)
        history.append((np.concatenate(nxt_par), np.concatenate(nxt_mac)))
    return None


def solve_state(S, max_depth=3):
    """S: sticker state after X2 + scramble. Returns the solution as one move string, or None."""
    S = S.astype(np.uint8)
    if not cross_ok(S):
        raise ValueError("cross is not solved on the bottom, this solver needs it solved")
    out = []
    for _ in range(8):
        solved = solved_slots(S)
        if all(solved):
            return " ".join(out)
        res = _search(S, solved, max_depth)
        if res is None:
            return None
        for m in res:
            S = S[MAC_PERM[m]]
            out.append(MAC_TEXT[m])
    return " ".join(out) if all(solved_slots(S)) else None


def state_from_scramble(scramble):
    return perm_of("X2 " + scramble).astype(np.uint8)


def simplify(solution):
    out, spin = [], 0
    amount = {"y": 1, "y'": 3, "y2": 2}
    def flush():
        nonlocal spin
        spin %= 4
        if spin:
            out.append({1: "y", 2: "y2", 3: "y'"}[spin])
        spin = 0
    for tok in solution.split():
        if tok in amount:
            spin += amount[tok]
        else:
            flush(); out.append(tok)
    flush()
    return " ".join(out)


def solve(scramble, max_depth=3):
    sol = solve_state(state_from_scramble(scramble), max_depth)
    return simplify(sol) if sol is not None else None


def verify(scramble, solution):
    """Re-run with YOUR Cube class: X2 + scramble + solution, then check bottom two layers."""
    c = Cube()
    apply_sequence(c, "X2 " + scramble)
    apply_sequence(c, solution)
    sides = [c.front, c.right, c.back, c.left]
    return bool(np.all(c.lower == c.lower[1, 1]) and all(np.all(f[1:3, :] == f[1, 1]) for f in sides))


if __name__ == "__main__":
    import sys
    # type your scramble here, or pass it on the command line. Moves are separated by spaces.
    SCRAMBLE = "R U2 R' U2 L' U L U"
    if len(sys.argv) > 1:
        SCRAMBLE = " ".join(sys.argv[1:])
    print("scramble:", SCRAMBLE)
    print("slots solved at the start:", dict(zip(SLOT_NAMES, solved_slots(state_from_scramble(SCRAMBLE)))))
    try:
        solution = solve(SCRAMBLE)
    except ValueError as e:
        print("cannot solve:", e)
    else:
        if solution is None:
            print("no solution found with the 41 algorithms (search depth 3)")
        else:
            print("solution:", solution)
            print("moves:", len(solution.split()))
            print("checked with your Cube class:", "solved" if verify(SCRAMBLE, solution) else "NOT solved")