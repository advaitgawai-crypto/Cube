"""OLL Test Tool.

Takes scramble moves from the user (applied to a solved cube with yellow on top),
validates that F2L is preserved, and outputs the exact sequence of moves to solve OLL.
"""

import copy
import os
import random
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


def cube_from_moves(text):
    """Apply scramble moves to a solved cube with yellow on top.

    Supports R L U D F B, wide moves r l, slice M, rotations x y z, primes (') and 2.
    """
    tokens = text.split()
    for token in tokens:
        _expand(token)  # validates each token

    cube = Cube()
    flip(cube)
    for token in tokens:
        for name in _expand(token):
            getattr(cube, name)()
    flip(cube)
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
    print("Enter moves applied to a solved cube (yellow on top).")
    print("Examples:")
    print("  - Single algorithm: F R U R' U' F'")
    print("  - With AUF / rotation: U R U R' U R U2 R'")
    print("  - Blank: generate a random test scramble")
    print("-" * 60)

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter your moves (blank = random test cube): ")
    text = text.strip()

    if text:
        try:
            cube = cube_from_moves(text)
        except ValueError as error:
            print(f"\n[ERROR] Invalid move syntax: {error}")
            sys.exit(1)

        if not f2l_solved(cube):
            print(
                "\n[ERROR] These moves break the cross or F2L.\n"
                "OLL requires the first two layers (cross + F2L) to be solved.\n"
                "Use top-layer turns (U, U', U2) or OLL/F2L-safe algorithms."
            )
            sys.exit(1)

        print("\nCube successfully set up from your moves (yellow on top).")
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
