import copy
import os
import random
import sys

import numpy as np

from notations import Cube
from oll_algorithms import OLLAlgorithms, _expand, _invert_token
from oll_dataset import AUF_SCRAMBLES, ROTATIONS, case_state
from oll_solver import OLLSolver
from oll_verify import F2L_ROWS, OLL_COUNT, OPPOSITE_FACE, SIDE_FACES, flip

# models folder sits next to this file: rubiks_cfop\models
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "oll_model.pt"
)


class _Recorder(OLLAlgorithms):
    """Same algorithms, but remembers the move string of the last one applied."""

    def _run(self, cube, sequence, inverse=False):
        self.last = sequence
        return super()._run(cube, sequence, inverse)


def algorithm_text(number):
    """The written moves of OLL `number`, e.g. "R U R' U R U2 R'"."""
    recorder = _Recorder()
    recorder.apply(Cube(), number)
    return recorder.last


def to_moves(path):
    """Turn the solver's token path (['y', 'OLL_27']) into the actual moves.

    The OLL algorithms are written for yellow on top, but notations.py has yellow
    on the bottom, so the moves start and end with z2 (flip the cube over, then back).
    """
    moves = ["z2"]
    for token in path:
        if token.startswith("OLL_"):
            moves.extend(algorithm_text(int(token[4:])).split())
        else:
            moves.append(token)
    moves.append("z2")
    return moves


def cube_from_moves(text):
    """Build a scrambled cube by applying the *inverse* of your moves.

    When you type an OLL algorithm, you want the cube state that *needs* that
    algorithm to be solved.  Applying the inverse of the moves to a solved cube
    produces exactly that state.

    Supports R L U D F B, wide moves r l, slices M, rotations x y z, primes and 2.
    Raises ValueError if a move is not understood.
    """
    tokens = text.split()
    for token in tokens:
        _expand(token)  # raises ValueError on a bad token

    # Invert: reverse order, invert each token
    inv_tokens = [_invert_token(t) for t in reversed(tokens)]

    cube = Cube()
    flip(cube)
    for token in inv_tokens:
        for name in _expand(token):
            getattr(cube, name)()
    flip(cube)
    return cube


def f2l_solved(cube):
    """True if the white layer and the first two layers are intact."""
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
    """Print every move needed to solve OLL on `cube`, and return the move list.

    The input cube must have the cross and F2L solved. It is not changed.
    Returns None if the solver finds no solution.
    """
    solver = solver or OLLSolver(MODEL_PATH)

    path = solver.solve(cube)
    if path is None:
        print("No solution found.")
        return None
    if not path:
        print("OLL is already solved.")
        return []

    moves = to_moves(path)
    case = next(t for t in path if t.startswith("OLL_"))[4:]

    # Replay the printed moves on a copy, so the printout matches what really happens.
    replay = solver.copy_cube(cube)
    for i, move in enumerate(moves, 1):
        for name in _expand(move):
            getattr(replay, name)()
        print(f"{i}. {move}")

    print(f"OLL case {case} solved in {len(moves)} moves: {' '.join(moves)}")
    if not solver.solved(replay):
        print("Warning: replaying the moves did not solve OLL.")
    return moves


if __name__ == "__main__":
    # Usage:
    #   python o.py                          asks you to type moves (blank = random cube)
    #   python o.py "R U R' U' R' F R F'"    moves in quotes (PowerShell needs the quotes)
    #
    # Your moves are applied to a solved cube with yellow on top, so they must keep
    # the cross and F2L solved. An OLL algorithm (or its inverse) plus U turns works.
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter your moves (blank = random test cube): ")
    text = text.strip()

    if text:
        try:
            cube = cube_from_moves(text)
        except ValueError as error:
            print(f"Could not read the moves: {error}")
            sys.exit(1)
        if not f2l_solved(cube):
            print(
                "These moves break the cross or F2L, so OLL can't be solved from this cube.\n"
                "Use moves that keep the first two layers solved, such as an OLL "
                "algorithm or its inverse, with U turns."
            )
            sys.exit(1)
        print("Cube built from your moves (yellow on top).")
    else:
        number = random.randint(1, OLL_COUNT)
        rotation = random.choice(list(ROTATIONS))
        pre_turn = random.choice(AUF_SCRAMBLES)
        cube = case_state(number, rotation, pre_turn)
        print(f"Random test cube: OLL case {number}, rotation {rotation}, top-layer turn {pre_turn}")

    solve_oll(cube)
