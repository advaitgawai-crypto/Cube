import copy

import numpy as np

from notations import Cube
from oll_algorithms import OLLAlgorithms

# Token sets for the OLL output heads.
FRAME_TOKENS = ("none", "y", "y2", "y'")
AUF_TOKENS = ("none", "U", "U2", "U'")
OLL_COUNT = 57

# The OLL algorithms are written for yellow on top. notations.py has white on
# upper and yellow on lower, so every token sequence is applied in a flipped view
# (z2 applied first, z2 again at the end). The cube passed in and returned stays
# in the notations.py layout, so the model input and the rest of the pipeline do
# not change.
YELLOW = 2
YELLOW_FACE = "upper"          # in the flipped view
OPPOSITE_FACE = "lower"        # in the flipped view
SIDE_FACES = ["front", "right", "back", "left"]

# Side rows that belong to the first two layers, in the flipped view.
# Row 0 is next to upper and row 2 is next to lower.
F2L_ROWS = [1, 2] if YELLOW_FACE == "upper" else [0, 1]

AUF_BASE = "U" if YELLOW_FACE == "upper" else "D"

FRAME_STEPS = {
    "none": [],
    "y": ["Y"],
    "y2": ["Y", "Y"],
    "y'": ["Y_prime"],
}

AUF_STEPS = {
    "none": [],
    "U": [AUF_BASE],
    "U2": [AUF_BASE, AUF_BASE],
    "U'": [AUF_BASE + "_Prime"],
}

_oll = OLLAlgorithms()


def flip(cube):
    """Turn the whole cube upside down (z2) in place. Applying it twice is identity."""
    cube.Z()
    cube.Z()
    return cube


def inverse_case_state(n):
    """Scrambled cube for OLL case n, in the notations.py layout.

    Built in the flipped view, where the algorithm is written, then flipped back.
    """
    cube = Cube()
    flip(cube)
    _oll.apply(cube, n, inverse=True)
    flip(cube)
    return cube


def _flipped_copy(cube):
    trial = copy.deepcopy(cube)
    return flip(trial)


def is_oll_solved(cube):
    """True if the yellow face is uniform and the first two layers are intact.

    Works on a notations.py-layout cube. The check is done in the flipped view.
    """
    view = _flipped_copy(cube)

    if not np.all(getattr(view, YELLOW_FACE) == YELLOW):
        return False

    opposite = getattr(view, OPPOSITE_FACE)
    if not np.all(opposite == opposite[1, 1]):
        return False

    for name in SIDE_FACES:
        face = getattr(view, name)
        for row in F2L_ROWS:
            if not np.all(face[row, :] == face[1, 1]):
                return False
    return True


def _why_not_solved(cube):
    """Human-readable reason, used in failure messages."""
    view = _flipped_copy(cube)
    if not np.all(getattr(view, YELLOW_FACE) == YELLOW):
        return "yellow face not uniform"
    opposite = getattr(view, OPPOSITE_FACE)
    if not np.all(opposite == opposite[1, 1]):
        return f"{OPPOSITE_FACE} face not uniform (first layer broken)"
    for name in SIDE_FACES:
        face = getattr(view, name)
        for row in F2L_ROWS:
            if not np.all(face[row, :] == face[1, 1]):
                return f"{name} row {row} broken (F2L broken)"
    return "ok"


def apply_triple(cube, frame, auf, alg):
    """Apply frame, then AUF, then OLL algorithm `alg` to `cube` in place.

    The tokens run in the flipped view, and the cube is returned in the
    notations.py layout.
    """
    if frame not in FRAME_STEPS:
        raise ValueError(f"unknown frame token: {frame}")
    if auf not in AUF_STEPS:
        raise ValueError(f"unknown AUF token: {auf}")
    if not 1 <= alg <= OLL_COUNT:
        raise ValueError(f"OLL number must be 1..{OLL_COUNT}, got {alg}")

    flip(cube)
    for name in FRAME_STEPS[frame] + AUF_STEPS[auf]:
        getattr(cube, name)()
    _oll.apply(cube, alg)
    flip(cube)
    return cube


def verify(cube, frame, auf, alg):
    """Check a token triple against `cube`. The input cube is not changed.

    Returns (ok, reason).
    """
    trial = copy.deepcopy(cube)
    apply_triple(trial, frame, auf, alg)
    if is_oll_solved(trial):
        return True, "ok"
    return False, _why_not_solved(trial)


def valid_triples(cube):
    """Every (frame, auf, alg) that solves OLL for `cube`.

    Used by the label generator, which needs all valid labels per state.
    """
    valid = []
    for frame in FRAME_TOKENS:
        for auf in AUF_TOKENS:
            for alg in range(1, OLL_COUNT + 1):
                ok, _ = verify(cube, frame, auf, alg)
                if ok:
                    valid.append((frame, auf, alg))
    return valid


if __name__ == "__main__":
    # Positive check: inverse of OLL k, then (none, none, k), must solve.
    # Negative check: a solved cube with OLL k applied must fail, because every
    # OLL case is non-solved. Both checks test the verifier, not the algorithms.
    positive_failures = []
    negative_passes = []
    for n in range(1, OLL_COUNT + 1):
        ok, _ = verify(inverse_case_state(n), "none", "none", n)
        if not ok:
            positive_failures.append(n)

        ok, _ = verify(Cube(), "none", "none", n)
        if ok:
            negative_passes.append(n)

    print("Positive failures:", positive_failures or "none")
    print("Negative passes (should be none):", negative_passes or "none")
