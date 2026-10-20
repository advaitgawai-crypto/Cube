# test_real_scramble.py
import random
from notations import Cube
from cross_solver import CrossSolver


def to_notation(move):
    """Convert names like 'R_Prime' to standard notation like \"R'\"."""
    return move.replace("_Prime", "'")


def format_moves(moves):
    return " ".join(to_notation(m) for m in moves) if moves else "(no moves needed)"


if __name__ == "__main__":
    solver = CrossSolver("models/cross_net_dropout03.pt")

    # 1. Scramble a fresh cube with random moves
    depth = 8
    scramble = [random.choice(solver.FACE_MOVES) for _ in range(depth)]

    cube = Cube()
    for m in scramble:
        getattr(cube, m)()

    print("Scramble:", format_moves(scramble))
    print()
    print("Cube state after scramble:")
    cube.display()

    # 2. Ask the solver for the moves that solve the cross
    solution = solver.solve(cube)

    if solution is None:
        print("No solution found")
    else:
        print("Solution:", format_moves(solution), f"({len(solution)} moves)")

        # 3. Check the solution on a copy, so the scrambled cube is untouched
        check = solver.copy_cube(cube)
        for m in solution:
            getattr(check, m)()
        print("Cross solved after applying solution:", solver.solved(check))