import os
import random
import sys

from cross_solver import CrossSolver
from notations import Cube

# models folder sits next to this file: rubiks_cfop\models
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "cross_net_dropout03.pt"
)


def to_standard(move):
    """'R_Prime' -> "R'" for display."""
    return move.replace("_Prime", "'")


def parse_scramble(text):
    """Standard notation ("R U2 F'") -> list of Cube method names."""
    moves = []
    for token in text.split():
        face = token[0].upper()
        if face not in "RLUDFB" or token[1:] not in ("", "'", "2"):
            raise ValueError(f"bad move: {token}")
        if token.endswith("'"):
            moves.append(face + "_Prime")
        elif token.endswith("2"):
            moves.extend([face, face])
        else:
            moves.append(face)
    return moves


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
    #   python c.py                  random 8-move scramble
    #   python c.py R U2 F' L        your own scramble, standard notation
    cube = Cube()
    if len(sys.argv) > 1:
        scramble = parse_scramble(" ".join(sys.argv[1:]))
    else:
        scramble = [random.choice(CrossSolver.FACE_MOVES) for _ in range(8)]

    for move in scramble:
        getattr(cube, move)()
    print("Scramble:", " ".join(to_standard(m) for m in scramble))

    solve_cross(cube)
