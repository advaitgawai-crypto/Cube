import copy

import numpy as np

from notations import Cube
from oll_algorithms import OLLAlgorithms

# Token sets for the OLL output heads.
FRAME_TOKENS = ("none", "y", "y2", "y'")
AUF_TOKENS = ("none", "U", "U2", "U'")
OLL_COUNT = 57

# notations.py gives yellow centre = 2, and the lower face is yellow.
# The OLL layer is therefore the lower face. If you change the convention
# (e.g. yellow on upper), change YELLOW and YELLOW_FACE together.
YELLOW = 2
YELLOW_FACE = "lower"
OPPOSITE_FACE = "upper"
SIDE_FACES = ["front", "right", "back", "left"]

# Rows on each side face that belong to the first two layers (not the OLL layer).
# Row 0 is next to the upper face and row 2 is next to the lower face.
F2L_ROWS = [0, 1] if YELLOW_FACE == "lower" else [1, 2]

# A U-type AUF is a turn of the OLL layer. Viewed from the yellow face,
# that is the layer's own clockwise turn, which is D when yellow is on lower.
AUF_BASE = "D" if YELLOW_FACE == "lower" else "U"

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


def is_oll_solved(cube):
    """True if the yellow face is uniform and the first two layers are intact.

    F2L check: the opposite face is uniform, and the side rows that belong to
    the first two layers match their own centre colour.
    """
    yellow_face = getattr(cube, YELLOW_FACE)
    if not np.all(yellow_face == YELLOW):
        return False

    opposite = getattr(cube, OPPOSITE_FACE)
    if not np.all(opposite == opposite[1, 1]):
        return False

    for name in SIDE_FACES:
        face = getattr(cube, name)
        for row in F2L_ROWS:
            if not np.all(face[row, :] == face[1, 1]):
                return False
    return True


def _why_not_solved(cube):
    """Human-readable reason, used in failure messages."""
    if not np.all(getattr(cube, YELLOW_FACE) == YELLOW):
        return "yellow face not uniform"
    opposite = getattr(cube, OPPOSITE_FACE)
    if not np.all(opposite == opposite[1, 1]):
        return f"{OPPOSITE_FACE} face not uniform (first layer broken)"
    for name in SIDE_FACES:
        face = getattr(cube, name)
        for row in F2L_ROWS:
            if not np.all(face[row, :] == face[1, 1]):
                return f"{name} row {row} broken (F2L broken)"
    return "ok"


def apply_triple(cube, frame, auf, alg):
    """Apply frame, then AUF, then OLL algorithm `alg` to `cube` in place."""
    if frame not in FRAME_STEPS:
        raise ValueError(f"unknown frame token: {frame}")
    if auf not in AUF_STEPS:
        raise ValueError(f"unknown AUF token: {auf}")
    if not 1 <= alg <= OLL_COUNT:
        raise ValueError(f"OLL number must be 1..{OLL_COUNT}, got {alg}")

    for name in FRAME_STEPS[frame] + AUF_STEPS[auf]:
        getattr(cube, name)()
    _oll.apply(cube, alg)
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
    # Positive check: inverse of OLL k, then the triple (none, none, k), must solve.
    # Negative check: a solved cube with OLL k applied must fail, because every
    # OLL case is non-solved. Both checks test the verifier, not the algorithms.
    positive_failures = []
    negative_passes = []
    for n in range(1, OLL_COUNT + 1):
        scrambled = Cube()
        _oll.apply(scrambled, n, inverse=True)
        ok, _ = verify(scrambled, "none", "none", n)
        if not ok:
            positive_failures.append(n)

        # verify() applies alg n to a solved cube; that must NOT pass.
        ok, _ = verify(Cube(), "none", "none", n)
        if ok:
            negative_passes.append(n)

    print("Positive failures:", positive_failures or "none")
    print("Negative passes (should be none):", negative_passes or "none")