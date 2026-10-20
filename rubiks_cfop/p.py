import os
import random
import sys

from notations import Cube
from oll_algorithms import _expand, _invert_token
from oll_verify import AUF_TOKENS, FRAME_TOKENS, flip, is_oll_solved
from pll_algorithms import PLLAlgorithms
from pll_solver import PLLSolver
from pll_verify import PLL_COUNT, inverse_case_state

# models folder sits next to this file: rubiks_cfop\models
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "pll_model.pt"
)


class _Recorder(PLLAlgorithms):
    """Same algorithms, but remembers the move string of the last one applied."""

    def _run(self, cube, sequence, inverse=False):
        self.last = sequence
        return super()._run(cube, sequence, inverse)


def algorithm_text(number):
    """The written moves of PLL `number`, e.g. "R U R' U' R' F R2 U' R' U' R U R' F'"."""
    recorder = _Recorder()
    recorder.apply(Cube(), number)
    return recorder.last


def to_moves(path):
    """Turn the solver's token path (['y', 'U', 'PLL_4', "U'"]) into the actual moves.

    The PLL algorithms are written for yellow on top, but notations.py has yellow
    on the bottom, so the moves start and end with z2 (flip the cube over, then back).
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
    """Build a scrambled cube by applying the *inverse* of your moves.

    When you type a PLL algorithm, you want the cube state that *needs* that
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


def solve_pll(cube, solver=None):
    """Print every move needed to solve PLL on `cube`, and return the move list.

    The input cube must have OLL solved. It is not changed.
    Returns None if the solver finds no solution.
    """
    solver = solver or PLLSolver(MODEL_PATH)

    path = solver.solve(cube)
    if path is None:
        print("No solution found.")
        return None
    if not path:
        print("PLL is already solved.")
        return []

    moves = to_moves(path)
    case_match = next((t for t in path if t.startswith("PLL_")), None)
    if case_match:
        case = case_match[4:]
        print(f"PLL case {case} solved in {len(moves)} moves: {' '.join(moves)}")
    else:
        print(f"PLL skip solved in {len(moves)} moves: {' '.join(moves)}")
    if not solver.solved(replay):
        print("Warning: replaying the moves did not solve PLL.")
    return moves


if __name__ == "__main__":
    # Usage:
    #   python p.py                          asks you to type moves (blank = random cube)
    #   python p.py "R U R' U R U2 R'"       moves in quotes (PowerShell needs the quotes)
    #
    # Your moves are applied to a solved cube with yellow on top, so they must keep
    # OLL solved. A PLL algorithm (or its inverse) plus U turns works.
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
        if not is_oll_solved(cube):
            print(
                "These moves break OLL (the yellow face, cross or F2L), so PLL can't be "
                "solved from this cube.\n"
                "Use moves that keep OLL solved, such as a PLL algorithm or its inverse, "
                "with U turns."
            )
            sys.exit(1)
        print("Cube built from your moves (yellow on top).")
    else:
        number = random.randint(1, PLL_COUNT)
        frame = random.choice(FRAME_TOKENS)
        pre_turn = random.choice(AUF_TOKENS)
        cube = inverse_case_state(number, frame, pre_turn, "none")
        print(f"Random test cube: PLL case {number}, rotation {frame}, top-layer turn {pre_turn}")

    solve_pll(cube)
