"""PLL Test Tool.

Takes the scrambled cube as six 3x3 matrices (colours 0-5, the same layout as the
Cube class in notations.py), validates that OLL is preserved, and outputs the exact
sequence of moves to solve PLL.
"""

import os
import random
import re
import sys

import numpy as np

from notations import Cube
from oll_algorithms import _expand
from oll_verify import AUF_TOKENS, FRAME_TOKENS, is_oll_solved
from pll_algorithms import PLLAlgorithms
from pll_solver import PLLSolver
from pll_verify import PLL_COUNT, inverse_case_state, is_pll_solved

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "pll_model.pt"
)

# Same order that Cube.display() prints the faces.
FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]


class _Recorder(PLLAlgorithms):
    """PLL algorithm runner that records the move sequence."""

    def _run(self, cube, sequence, inverse=False):
        self.last = sequence
        return super()._run(cube, sequence, inverse)


def algorithm_text(number):
    """Return the written moves of PLL `number`."""
    recorder = _Recorder()
    recorder.apply(Cube(), number)
    return recorder.last


def to_moves(path):
    """Convert solver tokens (e.g. ['y', 'U', 'PLL_11', "U'"]) into executable moves.

    PLL algorithms are written for yellow on top, but notations.py has yellow
    on the bottom, so moves start and end with z2 to match oll_verify.flip().
    """
    moves = ["z2"]
    for token in path:
        if token.startswith("PLL_"):
            moves.extend(algorithm_text(int(token[4:])).split())
        else:
            moves.append(token)
    moves.append("z2")
    return moves


def read_matrices():
    """Read the six matrices from the keyboard, or from a text file given as an argument.

    From the keyboard, reading stops once 54 numbers have been entered.
    A blank first line means "no cube" (the caller makes a random test cube).
    """
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as file:
            return file.read()

    lines, found = [], 0
    while found < 54:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            if found == 0:
                break
            continue
        lines.append(line)
        found += len(re.findall(r"\d", line))
    return "\n".join(lines)


def cube_from_matrices(text):
    """Build a cube from six 3x3 matrices of colours 0-5.

    Faces come in the order upper, front, lower, right, left, back, each stored as in
    notations.py (the same as cube.upper, cube.front, ...). Any layout works: plain
    rows, nested lists, or a numpy printout, because only the 54 digits are read.
    Raises ValueError if the cube cannot be a valid white-on-top cube.
    """
    digits = [int(d) for d in re.findall(r"\d", text)]
    if len(digits) != 54:
        raise ValueError(f"expected 54 numbers (six 3x3 matrices), got {len(digits)}")
    if max(digits) > 5:
        raise ValueError("colours must be numbers from 0 to 5")

    cube = Cube()
    for i, name in enumerate(FACE_ORDER):
        setattr(cube, name, np.array(digits[9 * i:9 * i + 9]).reshape(3, 3))

    for colour in range(6):
        if digits.count(colour) != 9:
            raise ValueError(f"colour {colour} appears {digits.count(colour)} times, expected 9")

    # Centres: white (0) on upper, yellow (2) on lower, sides in the cube's own order
    # (a y turn of the solved cube is allowed).
    if cube.upper[1, 1] != 0 or cube.lower[1, 1] != 2:
        raise ValueError("upper centre must be 0 (white) and lower centre must be 2 (yellow)")
    sides = tuple(int(getattr(cube, f)[1, 1]) for f in ("front", "right", "back", "left"))
    turned, allowed = Cube(), []
    for _ in range(4):
        allowed.append(tuple(int(getattr(turned, f)[1, 1]) for f in ("front", "right", "back", "left")))
        turned.Y()
    if sides not in allowed:
        raise ValueError(
            f"side centres (front, right, back, left) = {sides} do not match the cube's "
            "colour order; check that the faces are in the order upper, front, lower, right, left, back"
        )
    return cube


def solve_pll(cube, solver=None):
    """Find and print every move needed to solve PLL on `cube`."""
    solver = solver or PLLSolver(MODEL_PATH)

    if is_pll_solved(cube):
        print("PLL is already solved on this cube! No moves needed.")
        return []

    path = solver.solve(cube)
    if path is None:
        print("No solution found by the model.")
        return None

    moves = to_moves(path)
    case_match = next((t for t in path if t.startswith("PLL_")), None)

    if case_match:
        case_num = case_match[4:]
        print(f"\n--- PLL Solution (Case {case_num}) ---")
    else:
        print("\n--- PLL Solution (PLL Skip - AUF Alignment Only) ---")
    replay = solver.copy_cube(cube)
    for i, move in enumerate(moves, 1):
        for name in _expand(move):
            getattr(replay, name)()
        print(f"  {i:2d}. {move}")

    move_str = " ".join(moves)
    print(f"\nSolved in {len(moves)} moves:\n  {move_str}")

    if is_pll_solved(replay):
        print("\n[SUCCESS] Replaying moves verified: Cube is completely solved!")
    else:
        print("\n[WARNING] Replay verification failed.")

    return moves


def main():
    print("Enter the scrambled cube as six 3x3 matrices, one row per line.")
    print("Order: upper, front, lower, right, left, back (as Cube.display() prints).")
    print("Colours: 0 white, 1 orange, 2 yellow, 3 green, 4 blue, 5 red")
    print("  - A text file with the six matrices:  python pll_test.py cube.txt")
    print("  - Blank: generate a random test cube")

    try:
        text = read_matrices().strip()
    except OSError as error:
        print(f"\n[ERROR] Could not read the file: {error}")
        sys.exit(1)

    if text:
        try:
            cube = cube_from_matrices(text)
        except ValueError as error:
            print(f"\n[ERROR] Invalid cube: {error}")
            sys.exit(1)

        if not is_oll_solved(cube):
            print(
                "\n[ERROR] OLL is not solved on this cube (the yellow face or the F2L is incomplete).\n"
                "PLL requires OLL to be solved (all yellow facing the bottom and F2L intact)."
            )
            sys.exit(1)

        print("\nCube successfully read from your matrices.")
    else:
        number = random.randint(1, PLL_COUNT)
        frame = random.choice(FRAME_TOKENS)
        pre_turn = random.choice(AUF_TOKENS)
        cube = inverse_case_state(number, frame, pre_turn, "none")
        print(f"\nRandom test cube generated:")
        print(f"  PLL Case: {number} | Viewing Frame: {frame} | Top-layer turn: {pre_turn}")

    solve_pll(cube)


if __name__ == "__main__":
    main()