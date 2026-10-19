from notations import Cube
import numpy as np

# Move-expansion helpers are shared with the OLL file so both use the same notation rules.
from oll_algorithms import _expand, _invert_token

FACES = ["upper", "front", "lower", "right", "left", "back"]


class PLLAlgorithms:
    """The 21 PLL algorithms, numbered 1 to 21 in the order given.

    Each algorithm_N method applies PLL N to a cube in place.
    Pass inverse=True to apply the inverse of the algorithm instead.
    """

    COUNT = 21

    def _run(self, cube, sequence, inverse=False):
        tokens = sequence.split()
        if inverse:
            tokens = [_invert_token(t) for t in reversed(tokens)]
        for token in tokens:
            for method_name in _expand(token):
                getattr(cube, method_name)()
        return cube

    def apply(self, cube, number, inverse=False):
        """Apply PLL `number` (1..21) to `cube` by calling its method."""
        if not 1 <= number <= self.COUNT:
            raise ValueError(f"PLL number must be 1..21, got {number}")
        return getattr(self, f"algorithm_{number}")(cube, inverse)

    def algorithm_1(self, cube, inverse=False):
        return self._run(cube, "x L2 D2 L' U' L D2 L' U L'", inverse)

    def algorithm_2(self, cube, inverse=False):
        return self._run(cube, "x' L2 D2 L U L' D2 L U' L", inverse)

    def algorithm_3(self, cube, inverse=False):
        return self._run(cube, "x' L' U L D' L' U' L D L' U' L D' L' U L D", inverse)

    def algorithm_4(self, cube, inverse=False):
        return self._run(cube, "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R", inverse)

    def algorithm_5(self, cube, inverse=False):
        return self._run(cube, "R2 U R' U R' U' R U' R2 U' D R' U R D'", inverse)

    def algorithm_6(self, cube, inverse=False):
        return self._run(cube, "R' U' R U D' R2 U R' U R U' R U' R2 D", inverse)

    def algorithm_7(self, cube, inverse=False):
        return self._run(cube, "R2 U' R U' R U R' U R2 U D' R U' R' D", inverse)

    def algorithm_8(self, cube, inverse=False):
        return self._run(cube, "R U R' U' D R2 U' R U' R' U R' U R2 D'", inverse)

    def algorithm_9(self, cube, inverse=False):
        return self._run(cube, "M2 U M2 U2 M2 U M2", inverse)

    def algorithm_10(self, cube, inverse=False):
        return self._run(cube, "x R2 F R F' R U2 r' U r U2", inverse)

    def algorithm_11(self, cube, inverse=False):
        return self._run(cube, "R U R' F' R U R' U' R' F R2 U' R'", inverse)

    def algorithm_12(self, cube, inverse=False):
        return self._run(cube, "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'", inverse)

    def algorithm_13(self, cube, inverse=False):
        return self._run(cube, "R' U R U' R' F' U' F R U R' F R' F' R U' R", inverse)

    def algorithm_14(self, cube, inverse=False):
        return self._run(cube, "R U' R' U' R U R D R' U' R D' R' U2 R'", inverse)

    def algorithm_15(self, cube, inverse=False):
        return self._run(cube, "R2 F R U R U' R' F' R U2 R' U2 R", inverse)

    def algorithm_16(self, cube, inverse=False):
        return self._run(cube, "R U R' U' R' F R2 U' R' U' R U R' F'", inverse)

    def algorithm_17(self, cube, inverse=False):
        return self._run(cube, "M2 U M U2 M' U M2", inverse)

    def algorithm_18(self, cube, inverse=False):
        return self._run(cube, "M2 U' M U2 M' U' M2", inverse)

    def algorithm_19(self, cube, inverse=False):
        return self._run(cube, "R' U R' U' y R' F' R2 U' R' U R' F R F", inverse)

    def algorithm_20(self, cube, inverse=False):
        return self._run(cube, "F R U' R' U' R U R' F' R U R' U' R' F R F'", inverse)

    def algorithm_21(self, cube, inverse=False):
        return self._run(cube, "M' U M2 U M2 U M' U2 M2", inverse)


def _is_solved(cube):
    fresh = Cube()
    return all(np.array_equal(getattr(cube, f), getattr(fresh, f)) for f in FACES)


if __name__ == "__main__":
    # Self-check: each algorithm followed by its inverse must restore the solved cube.
    # This checks the notation mapping and the inverse logic. It does NOT check that
    # the algorithm permutes the last layer correctly; the PLL verifier does that.
    pll = PLLAlgorithms()
    failures = []
    for n in range(1, PLLAlgorithms.COUNT + 1):
        cube = Cube()
        pll.apply(cube, n)
        pll.apply(cube, n, inverse=True)
        if not _is_solved(cube):
            failures.append(n)
    if failures:
        print("Inverse check failed for:", failures)
    else:
        print("All 21 PLL algorithms pass the inverse check.")
