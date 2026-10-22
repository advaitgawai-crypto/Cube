import numpy as np
import sys
from notations import Cube
from f2l_algorithms import F2L_ALGORITHMS
from f2l_data_gen_v3 import apply_sequence, invert_token

# Whole-F2L solver adapted for matrix input and text file output.

FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]

def _labelled():
    c = Cube()
    for i, name in enumerate(FACE_ORDER):
        setattr(c, name, np.arange(9).reshape(3, 3) + 9 * i)
    return c

def perm_of(seq):
    c = _labelled()
    apply_sequence(c, seq)
    return np.concatenate([getattr(c, n).flatten() for n in FACE_ORDER]).astype(np.int64)

S0 = perm_of("X2")                                  # solved cube after X2
C0 = S0 // 9                                        # expected colors with cross on bottom
CENTERS = [4, 13, 22, 31, 40, 49]

def _p(face, r, c):
    return 9 * FACE_ORDER.index(face) + 3 * r + c

SLOTS = {
    "front-right": [_p("lower", 0, 2), _p("front", 2, 2), _p("right", 2, 0), _p("front", 1, 2), _p("right", 1, 0)],
    "front-left":  [_p("lower", 0, 0), _p("front", 2, 0), _p("left", 2, 2),  _p("front", 1, 0), _p("left", 1, 2)],
    "back-right":  [_p("lower", 2, 2), _p("back", 2, 0),  _p("right", 2, 2), _p("back", 1, 0),  _p("right", 1, 2)],
    "back-left":   [_p("lower", 2, 0), _p("back", 2, 2),  _p("left", 2, 0),  _p("back", 1, 2),  _p("left", 1, 0)],
}
SLOT_NAMES = list(SLOTS)
CROSS = [_p("lower", 0, 1), _p("front", 2, 1), _p("lower", 1, 2), _p("right", 2, 1),
         _p("lower", 2, 1), _p("back", 2, 1),  _p("lower", 1, 0), _p("left", 2, 1)]

def _ok(states, positions):
    return (states[:, positions] // 9 == C0[positions]).all(axis=1)

def build_macros():
    y = perm_of("y")
    seen = {}
    def add(text, f1):
        f = f1
        k = 0
        while not ((S0[f] // 9)[CENTERS] == C0[CENTERS]).all():
            f = f[y]
            k += 1
            if k > 3:
                return
        text = text + " y" * k if k else text
        key = f.tobytes()
        if key not in seen or len(text.split()) < len(seen[key][0].split()):
            seen[key] = (text.strip(), f)
            
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
    keep_pos = list(CROSS)
    for n, s in zip(SLOT_NAMES, prev_solved):
        if s: keep_pos += SLOTS[n]
    new_slot_pos = [SLOTS[n] for n, s in zip(SLOT_NAMES, prev_solved) if not s]
    
    level = S[None, :].astype(np.uint8)
    history = []
    seen = {int((S.astype(np.uint64) * _HM).sum())}
    
    for depth in range(1, max_depth + 1):
        nxt_states, nxt_par, nxt_mac = [], [], []
        best = None
        for start in range(0, len(level), 4000):
            chunk = level[start:start + 4000]
            cand = chunk[:, MAC_PERM].reshape(-1, 54)
            par = np.repeat(np.arange(start, start + len(chunk)), len(MAC_PERM))
            mac = np.tile(np.arange(len(MAC_PERM)), len(chunk))
            
            good = _ok(cand, keep_pos)
            cand, par, mac = cand[good], par[good], mac[good]
            if len(cand) == 0: continue
            
            for pos in new_slot_pos:
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
    S = S.astype(np.uint8)
    if not cross_ok(S):
        raise ValueError("Cross is not solved on the bottom layer. Check matrix orientation.")
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

def state_from_matrices(matrices):
    """Convert a list of 6 face matrices into the flat 54-element array F2L needs."""
    return (np.concatenate([m.flatten() for m in matrices]) * 9).astype(np.uint8)

def apply_solution_to_matrices(matrices, solution):
    """Load matrices into a Cube instance, apply the solution, and return new matrices."""
    c = Cube()
    for name, mat in zip(FACE_ORDER, matrices):
        setattr(c, name, mat.copy())
    
    if solution:
        apply_sequence(c, solution)
        
    return [c.upper, c.front, c.lower, c.right, c.left, c.back]

def read_matrices():
    print("Enter the 6 matrices (upper, front, lower, right, left, back), 3 rows per matrix:")
    lines = []
    try:
        for line in sys.stdin:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= 18:  # 6 faces * 3 rows
                break
    except EOFError:
        pass
        
    if len(lines) < 18:
        raise ValueError("Not enough rows provided. Expected 18 rows.")
        
    matrices = []
    for i in range(6):
        face_data = []
        for j in range(3):
            row_str = lines[i*3 + j].replace('[', '').replace(']', '').strip()
            row = [int(x) for x in row_str.split()]
            face_data.append(row)
        matrices.append(np.array(face_data))
    return matrices

def run_f2l_pipeline():
    try:
        matrices = read_matrices()
    except Exception as e:
        print(f"Error reading matrices: {e}")
        return
        
    state = state_from_matrices(matrices)
    print("Slots solved at start:", dict(zip(SLOT_NAMES, solved_slots(state))))
    
    try:
        raw_solution = solve_state(state)
    except ValueError as e:
        print("Cannot solve:", e)
        return
        
    if raw_solution is None:
        print("No solution found with the algorithms (search depth 3)")
        return
        
    solution = simplify(raw_solution)
    print(f"Solution: {solution}")
    print(f"Moves: {len(solution.split())}")
    
    out_matrices = apply_solution_to_matrices(matrices, solution)
    
    print("\n--- OUTPUT MATRICES FOR OLL/PLL STAGE ---")
    for name, mat in zip(FACE_ORDER, out_matrices):
        print(f"{name}:\n{mat}\n")
        
    # Write to f2l_ans.txt
    try:
        with open("f2l_ans.txt", "w") as f:
            for mat in out_matrices:
                for row in mat:
                    f.write(" ".join(map(str, row)) + "\n")
        print("Success: Resulting matrices have been written to 'f2l_ans.txt'.")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    run_f2l_pipeline()