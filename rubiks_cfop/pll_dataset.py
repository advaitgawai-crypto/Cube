import os

import numpy as np

from cube_utils import flatten_cube
from oll_verify import AUF_TOKENS, FRAME_TOKENS
from pll_verify import PLL_COUNT, inverse_case_state, valid_combos


def build_dataset():
    """Return (X, joint_mask, unsolvable).

    X          : (N, 54) flattened cubes, colours 0..5 (OLL already solved)
    joint_mask : (N, 4, 4, 21, 4) bool. joint_mask[i, f, p, a, q] is True when
                 frame f, pre-AUF p, PLL a, post-AUF q together solve state i.
                 Stored as whole combinations, so the heads are trained on
                 valid combinations, not on each head's marginal.
    unsolvable : list of (n, frame, pre, post) with no valid combination

    Covers all 21 cases, 4 frame rotations, 4 pre-AUFs, and 4 post-AUFs,
    deduplicated to cover 100% of distinct PLL states.
    """
    states_dict = {}
    unsolvable = []
    for n in range(1, PLL_COUNT + 1):
        for frame in FRAME_TOKENS:
            for pre in AUF_TOKENS:
                for post in AUF_TOKENS:
                    key = (n, frame, pre, post)
                    cube = inverse_case_state(n, frame, pre, post)
                    state_bytes = flatten_cube(cube).tobytes()
                    if state_bytes in states_dict:
                        continue
                    combos = valid_combos(cube)
                    if not combos:
                        unsolvable.append(key)
                        continue
                    mask = np.zeros((len(FRAME_TOKENS), len(AUF_TOKENS), PLL_COUNT, len(AUF_TOKENS)), dtype=bool)
                    for f, p, a, q in combos:
                        mask[FRAME_TOKENS.index(f), AUF_TOKENS.index(p), a - 1, AUF_TOKENS.index(q)] = True
                    states_dict[state_bytes] = (flatten_cube(cube), mask)

    X = np.array([v[0] for v in states_dict.values()], dtype=np.int64)
    joint = np.array([v[1] for v in states_dict.values()], dtype=bool)
    return X, joint, unsolvable


CACHE_PATH = r"C:\Users\Advait Gawai\OneDrive\Desktop\Cube\rubiks_cfop\models\pll_joint.npz"


def load_or_build(path=CACHE_PATH, rebuild=False):
    """Return (X, joint_mask, unsolvable).

    Reads the saved dataset if it exists, so the slow search runs only once.
    Pass rebuild=True to regenerate it.
    """
    if os.path.exists(path) and not rebuild:
        data = np.load(path)
        return data["X"], data["joint"], []

    X, joint, unsolvable = build_dataset()
    np.savez(path, X=X, joint=joint)
    return X, joint, unsolvable


if __name__ == "__main__":
    X, joint, unsolvable = load_or_build(rebuild=True)
    total = PLL_COUNT * len(FRAME_TOKENS) * len(AUF_TOKENS)
    per_state = joint.reshape(len(joint), -1).sum(axis=1)
    print("Total states:", total)
    print("Kept samples:", len(X))
    print("Input shape:", X.shape)
    print("Average valid whole combinations per state:", round(per_state.mean(), 2))
    print("Min / max valid whole combinations per state:", per_state.min(), per_state.max())
    print("Unsolvable:", len(unsolvable))
    for key in unsolvable[:20]:
        print("   PLL, frame, pre:", key)
