from notations import Cube
import numpy as np

# Wide moves, as defined by the user:
#   r = L then x,  r' = x' then L'
#   l = x' then R, l' = R' then x
WIDE_MOVES = {
    "r":  ["L", "X"],
    "r'": ["X_prime", "L_Prime"],
    "l":  ["X_prime", "R"],
    "l'": ["R_Prime", "X"],
}

# Single-turn tokens -> Cube method names in notations.py
MOVE_METHODS = {
    "R": "R",   "R'": "R_Prime",
    "L": "L",   "L'": "L_Prime",
    "U": "U",   "U'": "U_Prime",
    "D": "D",   "D'": "D_Prime",
    "F": "F",   "F'": "F_Prime",
    "B": "B",   "B'": "B_Prime",
    "M": "M",   "M'": "M_prime",
    "x": "X",   "x'": "X_prime",
    "y": "Y",   "y'": "Y_prime",
    "z": "Z",   "z'": "Z_prime",
}


def _expand(token):
    """Turn one token (e.g. 'R', "R'", 'R2', 'r', 'r2') into Cube method names."""
    if token.endswith("2"):
        return _expand(token[:-1]) * 2
    if token in WIDE_MOVES:
        return list(WIDE_MOVES[token])
    if token in MOVE_METHODS:
        return [MOVE_METHODS[token]]
    raise ValueError(f"unknown move token: {token}")


def _invert_token(token):
    if token.endswith("2"):
        return token
    if token.endswith("'"):
        return token[:-1]
    return token + "'"


class OLLAlgorithms:
    """The 57 OLL algorithms. Each algorithm_N method applies OLL N to a cube in place.

    Pass inverse=True to apply the inverse of the algorithm instead.
    """

    COUNT = 57

    def _run(self, cube, sequence, inverse=False):
        tokens = sequence.split()
        if inverse:
            tokens = [_invert_token(t) for t in reversed(tokens)]
        for token in tokens:
            for method_name in _expand(token):
                getattr(cube, method_name)()
        return cube

    def apply(self, cube, number, inverse=False):
        """Apply OLL `number` (1..57) to `cube` by calling its method."""
        if not 1 <= number <= self.COUNT:
            raise ValueError(f"OLL number must be 1..57, got {number}")
        return getattr(self, f"algorithm_{number}")(cube, inverse)

    # 1-10
    def algorithm_1(self, cube, inverse=False):
        return self._run(cube, "R U2 R2 F R F' U2 R' F R F'", inverse)

    def algorithm_2(self, cube, inverse=False):
        return self._run(cube, "r U r' U2 r U2 R' U2 R U' r'", inverse)

    def algorithm_3(self, cube, inverse=False):
        return self._run(cube, "r' R2 U R' U r U2 r' U M'", inverse)

    def algorithm_4(self, cube, inverse=False):
        return self._run(cube, "M U' r U2 r' U' R U' R' M'", inverse)

    def algorithm_5(self, cube, inverse=False):
        return self._run(cube, "l' U2 L U L' U l", inverse)

    def algorithm_6(self, cube, inverse=False):
        return self._run(cube, "r U2 R' U' R U' r'", inverse)

    def algorithm_7(self, cube, inverse=False):
        return self._run(cube, "r U R' U R U2 r'", inverse)

    def algorithm_8(self, cube, inverse=False):
        return self._run(cube, "l' U' L U' L' U2 l", inverse)

    def algorithm_9(self, cube, inverse=False):
        return self._run(cube, "R U R' U' R' F R2 U R' U' F'", inverse)

    def algorithm_10(self, cube, inverse=False):
        return self._run(cube, "R U R' U R' F R F' R U2 R'", inverse)

    # 11-20
    def algorithm_11(self, cube, inverse=False):
        return self._run(cube, "r U R' U R' F R F' R U2 r'", inverse)

    def algorithm_12(self, cube, inverse=False):
        return self._run(cube, "M' R' U' R U' R' U2 R U' R r'", inverse)

    def algorithm_13(self, cube, inverse=False):
        return self._run(cube, "F U R U' R2 F' R U R U' R'", inverse)

    def algorithm_14(self, cube, inverse=False):
        return self._run(cube, "R' F R U R' F' R F U' F'", inverse)

    def algorithm_15(self, cube, inverse=False):
        return self._run(cube, "l' U' l L' U' L U l' U l", inverse)

    def algorithm_16(self, cube, inverse=False):
        return self._run(cube, "r U r' R U R' U' r U' r'", inverse)

    def algorithm_17(self, cube, inverse=False):
        return self._run(cube, "F R' F' R2 r' U R U' R' U' M'", inverse)

    def algorithm_18(self, cube, inverse=False):
        return self._run(cube, "r U R' U R U2 r2 U' R U' R' U2 r", inverse)

    def algorithm_19(self, cube, inverse=False):
        return self._run(cube, "r' R U R U R' U' M' R' F R F'", inverse)

    def algorithm_20(self, cube, inverse=False):
        return self._run(cube, "r U R' U' M2 U R U' R' U' M'", inverse)

    # 21-30
    def algorithm_21(self, cube, inverse=False):
        return self._run(cube, "R U2 R' U' R U R' U' R U' R'", inverse)

    def algorithm_22(self, cube, inverse=False):
        return self._run(cube, "R U2 R2 U' R2 U' R2 U2 R", inverse)

    def algorithm_23(self, cube, inverse=False):
        return self._run(cube, "R2 D' R U2 R' D R U2 R", inverse)

    def algorithm_24(self, cube, inverse=False):
        return self._run(cube, "r U R' U' r' F R F'", inverse)

    def algorithm_25(self, cube, inverse=False):
        return self._run(cube, "F' r U R' U' r' F R", inverse)

    def algorithm_26(self, cube, inverse=False):
        return self._run(cube, "R U2 R' U' R U' R'", inverse)

    def algorithm_27(self, cube, inverse=False):
        return self._run(cube, "R U R' U R U2 R'", inverse)

    def algorithm_28(self, cube, inverse=False):
        return self._run(cube, "r U R' U' r' R U R U' R'", inverse)

    def algorithm_29(self, cube, inverse=False):
        return self._run(cube, "R U R' U' R U' R' F' U' F R U R'", inverse)

    def algorithm_30(self, cube, inverse=False):
        return self._run(cube, "F R' F R2 U' R' U' R U R' F2", inverse)

    # 31-40
    def algorithm_31(self, cube, inverse=False):
        return self._run(cube, "R' U' F U R U' R' F' R", inverse)

    def algorithm_32(self, cube, inverse=False):
        return self._run(cube, "L U F' U' L' U L F L'", inverse)

    def algorithm_33(self, cube, inverse=False):
        return self._run(cube, "R U R' U' R' F R F'", inverse)

    def algorithm_34(self, cube, inverse=False):
        return self._run(cube, "R U R2 U' R' F R U R U' F'", inverse)

    def algorithm_35(self, cube, inverse=False):
        return self._run(cube, "R U2 R2 F R F' R U2 R'", inverse)

    def algorithm_36(self, cube, inverse=False):
        return self._run(cube, "L' U' L U' L' U L U L F' L' F", inverse)

    def algorithm_37(self, cube, inverse=False):
        return self._run(cube, "F R' F' R U R U' R'", inverse)

    def algorithm_38(self, cube, inverse=False):
        return self._run(cube, "R U R' U R U' R' U' R' F R F'", inverse)

    def algorithm_39(self, cube, inverse=False):
        return self._run(cube, "L F' L' U' L U F U' L'", inverse)

    def algorithm_40(self, cube, inverse=False):
        return self._run(cube, "R' F R U R' U' F' U R", inverse)

    # 41-50
    def algorithm_41(self, cube, inverse=False):
        return self._run(cube, "R U R' U R U2 R' F R U R' U' F'", inverse)

    def algorithm_42(self, cube, inverse=False):
        return self._run(cube, "R' U' R U' R' U2 R F R U R' U' F'", inverse)

    def algorithm_43(self, cube, inverse=False):
        return self._run(cube, "F' U' L' U L F", inverse)

    def algorithm_44(self, cube, inverse=False):
        return self._run(cube, "F U R U' R' F'", inverse)

    def algorithm_45(self, cube, inverse=False):
        return self._run(cube, "F R U R' U' F'", inverse)

    def algorithm_46(self, cube, inverse=False):
        return self._run(cube, "R' U' R' F R F' U R", inverse)

    def algorithm_47(self, cube, inverse=False):
        return self._run(cube, "R' U' R' F R F' R' F R F' U R", inverse)

    def algorithm_48(self, cube, inverse=False):
        return self._run(cube, "F R U R' U' R U R' U' F'", inverse)

    def algorithm_49(self, cube, inverse=False):
        return self._run(cube, "r U' r2 U r2 U r2 U' r", inverse)

    def algorithm_50(self, cube, inverse=False):
        return self._run(cube, "r' U r2 U' r2 U' r2 U r'", inverse)

    # 51-57
    def algorithm_51(self, cube, inverse=False):
        return self._run(cube, "F U R U' R' U R U' R' F'", inverse)

    def algorithm_52(self, cube, inverse=False):
        return self._run(cube, "R U R' U R U' B U' B' R'", inverse)

    def algorithm_53(self, cube, inverse=False):
        return self._run(cube, "l' U2 L U L' U' L U L' U l", inverse)

    def algorithm_54(self, cube, inverse=False):
        return self._run(cube, "r U2 R' U' R U R' U' R U' r'", inverse)

    def algorithm_55(self, cube, inverse=False):
        return self._run(cube, "R' F R U R U' R2 F' R2 U' R' U R U R'", inverse)

    def algorithm_56(self, cube, inverse=False):
        return self._run(cube, "r' U' r U' R' U R U' R' U R r' U r", inverse)

    def algorithm_57(self, cube, inverse=False):
        return self._run(cube, "R U R' U' M' U R U' r'", inverse)


FACES = ["upper", "front", "lower", "right", "left", "back"]


def _is_solved(cube):
    fresh = Cube()
    return all(np.array_equal(getattr(cube, f), getattr(fresh, f)) for f in FACES)


if __name__ == "__main__":
    # Self-check: each algorithm followed by its inverse must restore the solved cube.
    # This checks the notation mapping and inverse logic. It does NOT check that the
    # algorithm orients the top layer; the label verifier does that.
    oll = OLLAlgorithms()
    failures = []
    for n in range(1, OLLAlgorithms.COUNT + 1):
        cube = Cube()
        oll.apply(cube, n)
        oll.apply(cube, n, inverse=True)
        if not _is_solved(cube):
            failures.append(n)
    if failures:
        print("Inverse check failed for:", failures)
    else:
        print("All 57 algorithms pass the inverse check.")