"""OLL Test Tool.

Takes the scrambled cube as six 3x3 matrices (colours 0-5, the same layout as the
Cube class in notations.py), validates that F2L is preserved, and outputs the exact
sequence of moves to solve OLL.
"""

import copy
import os
import random
import re
import sys

import numpy as np

from notations import Cube
from oll_algorithms import OLLAlgorithms, _expand
from oll_dataset import AUF_SCRAMBLES, ROTATIONS, case_state
from oll_solver import OLLSolver
from oll_verify import F2L_ROWS, OLL_COUNT, OPPOSITE_FACE, SIDE_FACES, flip, is_oll_solved

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "oll_model.pt"
)

# Same order that Cube.display() prints the faces.
FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]


class _Recorder(OLLAlgorithms):
    """OLL algorithm runner that records the move sequence."""

    def _run(self, cube, sequence, inverse=False):
        self.last = sequence
        return super()._run(cube, sequence, inverse)


def algorithm_text(number):
    """Return the written moves of OLL `number`."""
    recorder = _Recorder()
    recorder.apply(Cube(), number)
    return recorder.last


def to_moves(path):
    """Convert solver tokens (e.g. ['y', 'OLL_27']) into executable moves.

    OLL algorithms are written for yellow on top, but notations.py has yellow
    on the bottom, so moves start and end with z2 (flip over, then back).
    """
    moves = ["z2"]
    for token in path:
        if token.startswith("OLL_"):
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


def f2l_solved(cube):
    """Check if the white face and the first two layers are intact."""
    view = copy.deepcopy(cube)
    flip(view)
    opposite = getattr(view, OPPOSITE_FACE)
    if not np.all(opposite == opposite[1, 1]):
        return False
    for name in SIDE_FACES:
        face = getattr(view, name)
        for row in F2L_ROWS:
            if not np.all(face[row, :] == face[1, 1]):
                return False
    return True


def solve_oll(cube, solver=None):
    """Find and print every move needed to solve OLL on `cube`."""
    solver = solver or OLLSolver(MODEL_PATH)

    if is_oll_solved(cube):
        print("OLL is already solved on this cube! No moves needed.")
        return []

    path = solver.solve(cube)
    if path is None:
        print("No solution found by the model.")
        return None

    moves = to_moves(path)
    case_num = next(t for t in path if t.startswith("OLL_"))[4:]

    print(f"\n--- OLL Solution (Case {case_num}) ---")
    replay = solver.copy_cube(cube)
    for i, move in enumerate(moves, 1):
        for name in _expand(move):
            getattr(replay, name)()
        print(f"  {i:2d}. {move}")

    move_str = " ".join(moves)
    print(f"\nSolved in {len(moves)} moves:\n  {move_str}")

    if is_oll_solved(replay):
        print("\n[SUCCESS] Replaying moves verified: OLL is successfully solved!")
    else:
        print("\n[WARNING] Replay verification failed.")

    return moves


def main():
    print("=" * 60)
    print("                RUBIK'S CUBE OLL SOLVER")
    print("=" * 60)
    print("Enter the scrambled cube as six 3x3 matrices, one row per line.")
    print("Order: upper, front, lower, right, left, back (as Cube.display() prints).")
    print("Colours: 0 white, 1 orange, 2 yellow, 3 green, 4 blue, 5 red")
    print("Examples:")
    print("  - A row:  0 0 0   (brackets and commas are fine too)")
    print("  - A text file with the six matrices:  python oll_test.py cube.txt")
    print("  - Blank: generate a random test cube")
    print("-" * 60)

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

        if not f2l_solved(cube):
            print(
                "\n[ERROR] This cube does not have the cross and F2L solved.\n"
                "OLL requires the first two layers (cross + F2L) to be solved."
            )
            sys.exit(1)

        print("\nCube successfully read from your matrices.")
    else:
        number = random.randint(1, OLL_COUNT)
        rotation = random.choice(list(ROTATIONS))
        pre_turn = random.choice(AUF_SCRAMBLES)
        cube = case_state(number, rotation, pre_turn)
        print(f"\nRandom test cube generated:")
        print(f"  OLL Case: {number} | Rotation: {rotation} | Top-layer turn: {pre_turn}")

    solve_oll(cube)


if __name__ == "__main__":
    main()
