# test_pll_from_oll.py
import time
from notations import Cube
from oll_safe_algorithms import OLLSafe
from pll_solver import PLLSolver


def pll_scramble(alg_number):
    """Fresh (solved) cube with one PLL algorithm applied: cross, F2L, OLL solved."""
    cube = Cube()
    getattr(OLLSafe(), f"algorithm_{alg_number}")(cube)
    return cube


if __name__ == "__main__":
    solver = PLLSolver("models/pll_net_dropout0.pt")

    alg_numbers = sorted(
        int(name.split("_")[1])
        for name in dir(OLLSafe)
        if name.startswith("algorithm_")
    )

    max_iterations = 20000
    print(f"{'alg':>4} {'time(s)':>8} {'moves':>6}  result")
    print("-" * 40)

    for n in alg_numbers:
        cube = pll_scramble(n)
        start = time.perf_counter()
        solution = solver.solve(cube, max_iterations=max_iterations)
        elapsed = time.perf_counter() - start

        if solution is None:
            print(f"{n:>4} {elapsed:>8.2f} {'-':>6}  NO SOLUTION (budget hit)")
        else:
            ok = solver.solved(solver.copy_cube(cube))  # sanity check of starting state
            print(f"{n:>4} {elapsed:>8.2f} {len(solution):>6}  solved={ok}")