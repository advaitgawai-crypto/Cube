import os
import random
import re
import sys

import numpy as np

from cross_solver import CrossSolver
from notations import Cube

# models folder sits next to this file: rubiks_cfop\models
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "cross_net_dropout03.pt"
)

# Same order that Cube.display() prints the faces.
FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]


def to_standard(move):
    """'R_Prime' -> "R'" for display."""
    return move.replace("_Prime", "'")


def read_matrices():
    """Read the six matrices from the keyboard, or from a text file given as an argument.

    From the keyboard, reading stops once 54 numbers have been entered.
    A blank first line means "no cube" (the caller makes a random scramble).
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


def solve_cross(cube, solver=None):
    """Print every move needed to solve the cross on `cube`, and return the move list.

    The input cube is not changed. Returns None if the solver finds no solution.
    """
    solver = solver or CrossSolver(MODEL_PATH)

    moves = solver.solve(cube)
    if moves is None:
        print("No solution found.")
        return None
    if not moves:
        print("Cross is already solved.")
        return []

    # Replay the moves on a copy, so the printout matches what really happens.
    replay = solver.copy_cube(cube)
    for i, move in enumerate(moves, 1):
        getattr(replay, move)()
        print(f"{i}. {to_standard(move)}")

    print(f"Cross solved in {len(moves)} moves: {' '.join(to_standard(m) for m in moves)}")
    if not solver.solved(replay):
        print("Warning: replaying the moves did not solve the cross.")
    return moves


if __name__ == "__main__":
    # Usage:
    #   python c.py              type the six 3x3 matrices (blank = random 8-move scramble)
    #   python c.py cube.txt     read the six matrices from a text file
    #
    # Order: upper, front, lower, right, left, back (as Cube.display() prints).
    # Colours: 0 white, 1 orange, 2 yellow, 3 green, 4 blue, 5 red
    print("Enter the scrambled cube as six 3x3 matrices, one row per line.")
    print("Order: upper, front, lower, right, left, back (as Cube.display() prints).")
    print("Colours: 0 white, 1 orange, 2 yellow, 3 green, 4 blue, 5 red")
    print("Blank: random 8-move scramble")

    try:
        text = read_matrices().strip()
    except OSError as error:
        print(f"Could not read the file: {error}")
        sys.exit(1)

    if text:
        try:
            cube = cube_from_matrices(text)
        except ValueError as error:
            print(f"Invalid cube: {error}")
            sys.exit(1)
        print("Cube read from your matrices.")
    else:
        cube = Cube()
        scramble = [random.choice(CrossSolver.FACE_MOVES) for _ in range(8)]
        for move in scramble:
            getattr(cube, move)()
        print("Scramble:", " ".join(to_standard(m) for m in scramble))

    solve_cross(cube)
