import numpy as np

from notations import Cube
from cube_utils import flatten_cube
from oll_algorithms import OLLAlgorithms
from oll_verify import AUF_STEPS, FRAME_TOKENS, OLL_COUNT, verify

_oll = OLLAlgorithms()

# Whole-cube rotations applied to the case state.
ROTATIONS = {
    0: [],
    1: ["Y"],
    2: ["Y", "Y"],
    3: ["Y_prime"],
}

# Top-layer pre-turns (AUF scrambles) applied to the case state. These are the
# top-layer orientations a real solve can start from. The turn is on the OLL
# layer, so it uses the same steps as the AUF tokens in oll_verify.
AUF_SCRAMBLES = ["none", "U", "U2", "U'"]


def case_state(n, rotation, auf_scramble):
    """Scrambled state for OLL case n, then the AUF scramble, then the rotation."""
    cube = Cube()
    _oll.apply(cube, n, inverse=True)
    for name in AUF_STEPS[auf_scramble]:
        getattr(cube, name)()
    for name in ROTATIONS[rotation]:
        getattr(cube, name)()
    return cube


def build_dataset():
    """Return (X, frame_mask, alg_label, meta, unsolvable).

    X          : (N, 54) flattened cubes, colours 0..5
    frame_mask : (N, 4) bool, True for each frame token that solves the state
                 with AUF = none and the state's own algorithm
    alg_label  : (N,) int, algorithm index 0..56 (OLL number - 1)
    meta       : list of (n, rotation, auf_scramble) for each kept row
    unsolvable : list of (n, rotation, auf_scramble) with no valid frame.
                 These need an AUF head, so they are reported and left out.
    """
    X, masks, labels, meta, unsolvable = [], [], [], [], []
    for n in range(1, OLL_COUNT + 1):
        for rotation in ROTATIONS:
            for auf_scramble in AUF_SCRAMBLES:
                cube = case_state(n, rotation, auf_scramble)
                mask = [verify(cube, f, "none", n)[0] for f in FRAME_TOKENS]
                key = (n, rotation, auf_scramble)
                if not any(mask):
                    unsolvable.append(key)
                    continue
                X.append(flatten_cube(cube))
                masks.append(mask)
                labels.append(n - 1)
                meta.append(key)
    return (
        np.array(X, dtype=np.int64),
        np.array(masks, dtype=bool),
        np.array(labels, dtype=np.int64),
        meta,
        unsolvable,
    )


if __name__ == "__main__":
    X, mask, labels, meta, unsolvable = build_dataset()
    total = OLL_COUNT * len(ROTATIONS) * len(AUF_SCRAMBLES)
    print("Total states:", total)
    print("Kept samples:", len(X))
    print("Input shape:", X.shape)
    counts = mask.sum(axis=1)
    for k in range(1, 5):
        print(f"  kept states with {k} valid frame(s): {int((counts == k).sum())}")
    print("Unsolvable with AUF none (need AUF head):", len(unsolvable))
    for key in unsolvable[:20]:
        print("   OLL, rotation, AUF scramble:", key)