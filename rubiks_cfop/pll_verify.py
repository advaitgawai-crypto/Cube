import copy

import numpy as np

from notations import Cube
from oll_verify import AUF_STEPS, AUF_TOKENS, FRAME_STEPS, FRAME_TOKENS, flip
from pll_algorithms import PLLAlgorithms

PLL_COUNT = 21
FACES = ["upper", "front", "lower", "right", "left", "back"]

# Inverse of each token, used to build scrambles.
AUF_INVERSE = {"none": "none", "U": "U'", "U2": "U2", "U'": "U"}
FRAME_INVERSE = {"none": "none", "y": "y'", "y2": "y2", "y'": "y"}

_pll = PLLAlgorithms()


def is_pll_solved(cube):
    """True if every face is one colour. Any whole-cube orientation counts as solved."""
    return all(np.all(getattr(cube, f) == getattr(cube, f)[0, 0]) for f in FACES)


def apply_triple(cube, frame, pre, alg, post):
    """Apply frame, pre-AUF, PLL algorithm `alg`, post-AUF to `cube` in place.

    The tokens run in the flipped view that the algorithms expect. The cube is
    returned in the notations.py layout.
    """
    if frame not in FRAME_STEPS:
        raise ValueError(f"unknown frame token: {frame}")
    if pre not in AUF_STEPS or post not in AUF_STEPS:
        raise ValueError(f"unknown AUF token: {pre} / {post}")
    if not 1 <= alg <= PLL_COUNT:
        raise ValueError(f"PLL number must be 1..{PLL_COUNT}, got {alg}")

    flip(cube)
    for name in FRAME_STEPS[frame] + AUF_STEPS[pre]:
        getattr(cube, name)()
    _pll.apply(cube, alg)
    for name in AUF_STEPS[post]:
        getattr(cube, name)()
    flip(cube)
    return cube


def verify(cube, frame, pre, alg, post):
    """Check a token combination against `cube`. The input cube is not changed.

    Returns (ok, reason).
    """
    trial = copy.deepcopy(cube)
    apply_triple(trial, frame, pre, alg, post)
    if is_pll_solved(trial):
        return True, "ok"
    return False, "not all faces uniform"


def _clone(cube):
    """Fast copy of a Cube (much cheaper than deepcopy)."""
    n = Cube.__new__(Cube)
    n.upper = cube.upper.copy()
    n.front = cube.front.copy()
    n.lower = cube.lower.copy()
    n.right = cube.right.copy()
    n.left = cube.left.copy()
    n.back = cube.back.copy()
    return n


def valid_combos(cube):
    """Every (frame, pre, alg, post) that solves `cube`.

    Same result as calling verify() on every combination, but shares work: the
    frame and pre-AUF are applied once per branch, and only the post-AUF is tried
    on each algorithm result. The solved check does not depend on orientation,
    so the final flip back is not needed here.
    """
    valid = []
    base = _clone(cube)
    flip(base)
    for frame in FRAME_TOKENS:
        after_frame = _clone(base)
        for name in FRAME_STEPS[frame]:
            getattr(after_frame, name)()
        for pre in AUF_TOKENS:
            after_pre = _clone(after_frame)
            for name in AUF_STEPS[pre]:
                getattr(after_pre, name)()
            for alg in range(1, PLL_COUNT + 1):
                after_alg = _clone(after_pre)
                _pll.apply(after_alg, alg)
                for post in AUF_TOKENS:
                    final = _clone(after_alg)
                    for name in AUF_STEPS[post]:
                        getattr(final, name)()
                    if is_pll_solved(final):
                        valid.append((frame, pre, alg, post))
    return valid


def inverse_case_state(n, frame="none", pre="none", post="none"):
    """Scrambled cube that (frame, pre, n, post) solves, in the notations.py layout.

    Built by applying the inverse of the triple to a solved cube, in reverse order.
    """
    cube = Cube()
    flip(cube)
    for name in AUF_STEPS[AUF_INVERSE[post]]:
        getattr(cube, name)()
    _pll.apply(cube, n, inverse=True)
    for name in AUF_STEPS[AUF_INVERSE[pre]]:
        getattr(cube, name)()
    for name in FRAME_STEPS[FRAME_INVERSE[frame]]:
        getattr(cube, name)()
    flip(cube)
    return cube


if __name__ == "__main__":
    # Positive check: every (frame, pre, post) combination applied to its own
    # inverse state must solve. Negative check: a solved cube with PLL n applied
    # must not be solved, because every PLL case is a permutation of the last layer.
    positive_failures = []
    for n in range(1, PLL_COUNT + 1):
        for frame in FRAME_TOKENS:
            for pre in AUF_TOKENS:
                for post in AUF_TOKENS:
                    ok, _ = verify(inverse_case_state(n, frame, pre, post), frame, pre, n, post)
                    if not ok:
                        positive_failures.append((n, frame, pre, post))

    negative_passes = []
    for n in range(1, PLL_COUNT + 1):
        ok, _ = verify(Cube(), "none", "none", n, "none")
        if ok:
            negative_passes.append(n)

    print("Positive failures:", positive_failures[:20] or "none")
    print("Negative passes (should be none):", negative_passes or "none")
