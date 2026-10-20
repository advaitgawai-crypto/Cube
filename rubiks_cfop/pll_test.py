"""PLL Test Tool.

Takes scramble moves from the user (applied to a solved cube with yellow on top),
validates that OLL is preserved, and outputs the exact sequence of moves to solve PLL.
"""

import os
import random
import sys

from notations import Cube
from oll_algorithms import _expand
from oll_verify import AUF_TOKENS, FRAME_TOKENS, flip, is_oll_solved
from pll_algorithms import PLLAlgorithms
from pll_solver import PLLSolver
from pll_verify import PLL_COUNT, inverse_case_state, is_pll_solved

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "pll_model.pt"
)


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
    on the bottom, so moves start and end with z2 (flip over, then back).
    """
    moves = ["z2"]
    for token in path:
        if token.startswith("PLL_"):
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
    print("=" * 60)
    print("                RUBIK'S CUBE PLL SOLVER")
    print("=" * 60)
    print("Enter moves applied to a solved cube (yellow on top).")
    print("Examples:")
    print("  - Single algorithm: R U R' F' R U R' U' R' F R2 U' R'")
    print("  - Chained algorithms: M2 U M2 U2 M2 U M2 M2 U M U2 M' U M2")
    print("  - With AUFs / rotations: U R U R' U' R' F R2 U' R' U' R U R' F' U'")
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

        if not is_oll_solved(cube):
            print(
                "\n[ERROR] These moves break OLL (the yellow top face or the F2L).\n"
                "PLL requires OLL to be solved (all yellow facing up and F2L intact).\n"
                "Use PLL algorithms, whole-cube rotations (y), and top-layer turns (U, U', U2)."
            )
            sys.exit(1)

        print("\nCube successfully set up from your moves (yellow on top).")
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
