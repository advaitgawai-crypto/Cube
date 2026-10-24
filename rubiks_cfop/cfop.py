import os
import sys
import time
import numpy as np

# Resolve output paths relative to this script file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import functions from existing modules
from notations import Cube
from c import cube_from_matrices, read_matrices, solve_cross, to_standard, FACE_ORDER
from f import state_from_matrices, solve_state, simplify, apply_solution_to_matrices
from o import solve_oll
from p import solve_pll
from oll_algorithms import _expand
from pll_verify import is_pll_solved

def format_matrices(cube):
    """Format the six faces of the cube as text matrices in standard FACE_ORDER."""
    lines = []
    for face_name in FACE_ORDER:
        face_arr = getattr(cube, face_name)
        for row in face_arr:
            lines.append(" ".join(str(val) for val in row))
        lines.append("")
    return "\n".join(lines).strip()

def print_stage_matrices(cube, stage_title):
    print(f"\n--- {stage_title} Matrices ---")
    for name in FACE_ORDER:
        print(f"{name}:")
        print(getattr(cube, name))
    print("-" * 40)

def main():
    print("=" * 60)
    print("             FULL CFOP RUBIK'S CUBE SOLVER")
    print("=" * 60)

    try:
        text = read_matrices().strip()
    except OSError as error:
        print(f"Could not read the input: {error}")
        sys.exit(1)

    if not text:
        print("No input matrices provided. Exiting.")
        sys.exit(1)

    try:
        cube = cube_from_matrices(text)
    except ValueError as error:
        print(f"Invalid cube input: {error}")
        sys.exit(1)

    print_stage_matrices(cube, "Initial Scrambled Cube")

    total_moves = []
    start_total_time = time.time()

    # Align Y-rotation if necessary so front center is 1 (orange)
    align_y_moves = []
    while cube.front[1, 1] != 1 and len(align_y_moves) < 4:
        cube.Y()
        align_y_moves.append('y')
    total_moves.extend(align_y_moves)

    # -------------------------------------------------------------------------
    # STAGE 1: CROSS
    # -------------------------------------------------------------------------
    print("\n--- 1. CROSS STAGE ---")
    t0 = time.time()
    cross_moves = solve_cross(cube)
    cross_time = time.time() - t0

    if cross_moves is None:
        print("Failed to solve Cross stage.")
        sys.exit(1)

    for m in cross_moves:
        getattr(cube, m)()

    # Apply X2 to flip cross from top (white on upper) to bottom (white on lower) for F2L
    cube.X()
    cube.X()

    cross_std = [to_standard(m) for m in cross_moves] + ["x2"]
    total_moves.extend([to_standard(m) for m in cross_moves] + ["x2"])
    print(f"Cross Moves ({len(cross_std)}): {' '.join(cross_std)}")
    print(f"Cross Stage Time: {cross_time:.4f} seconds")
    print_stage_matrices(cube, "Cross Solved & Oriented for F2L")

    # -------------------------------------------------------------------------
    # STAGE 2: F2L
    # -------------------------------------------------------------------------
    print("\n--- 2. F2L STAGE ---")
    f2l_matrices = [getattr(cube, name) for name in FACE_ORDER]
    f2l_state = state_from_matrices(f2l_matrices)

    t0 = time.time()
    try:
        raw_f2l_sol = solve_state(f2l_state)
    except ValueError as e:
        print(f"F2L Solver error: {e}")
        sys.exit(1)
    f2l_time = time.time() - t0

    if raw_f2l_sol is None:
        print("No F2L solution found.")
        sys.exit(1)

    f2l_solution = simplify(raw_f2l_sol)
    f2l_moves = f2l_solution.split() if f2l_solution else []
    
    # Apply F2L solution to the cube object
    updated_f2l_mats = apply_solution_to_matrices(f2l_matrices, f2l_solution)
    for name, mat in zip(FACE_ORDER, updated_f2l_mats):
        setattr(cube, name, mat)

    # Apply X2 to flip the cube back (white on top, yellow on bottom) for OLL/PLL
    cube.X()
    cube.X()
    f2l_moves.append("x2")

    total_moves.extend(f2l_moves)
    print(f"F2L Moves ({len(f2l_moves)}): {f2l_solution + ' x2' if f2l_solution else 'x2'}")
    print(f"F2L Stage Time: {f2l_time:.4f} seconds")
    print_stage_matrices(cube, "F2L Solved & Oriented for OLL/PLL")

    # -------------------------------------------------------------------------
    # STAGE 3: OLL
    # -------------------------------------------------------------------------
    print("\n--- 3. OLL STAGE ---")
    t0 = time.time()
    oll_moves = solve_oll(cube)
    oll_time = time.time() - t0

    if oll_moves is None:
        print("Failed to solve OLL stage.")
        sys.exit(1)

    for m in oll_moves:
        for name in _expand(m):
            getattr(cube, name)()

    total_moves.extend(oll_moves)
    print(f"OLL Stage Time: {oll_time:.4f} seconds")
    print_stage_matrices(cube, "OLL Solved")

    # -------------------------------------------------------------------------
    # STAGE 4: PLL
    # -------------------------------------------------------------------------
    print("\n--- 4. PLL STAGE ---")
    t0 = time.time()
    pll_moves = solve_pll(cube)
    pll_time = time.time() - t0

    if pll_moves is None:
        print("Failed to solve PLL stage.")
        sys.exit(1)

    for m in pll_moves:
        for name in _expand(m):
            getattr(cube, name)()

    # Safety check to guarantee cube is fully solved after PLL
    if not is_pll_solved(cube):
        print("PLL failed verification")
        sys.exit(1)

    total_moves.extend(pll_moves)
    print(f"PLL Stage Time: {pll_time:.4f} seconds")
    print_stage_matrices(cube, "PLL Solved (Fully Solved Cube)")

    total_elapsed_time = time.time() - start_total_time

    # Save final solved matrices
    formatted_output = format_matrices(cube) + "\n"
    output_filename = os.path.join(_SCRIPT_DIR, "full_solved_matrix.txt")
    with open(output_filename, "w") as f:
        f.write(formatted_output)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("                     FULL SOLUTION SUMMARY")
    print("=" * 60)
    print(f"Cross ({len(cross_std)} moves, {cross_time:.4f}s): {' '.join(cross_std)}")
    print(f"F2L   ({len(f2l_moves)} moves, {f2l_time:.4f}s): {f2l_solution + ' x2' if f2l_solution else 'x2'}")
    print(f"OLL   ({len(oll_moves)} moves, {oll_time:.4f}s): {' '.join(oll_moves)}")
    print(f"PLL   ({len(pll_moves)} moves, {pll_time:.4f}s): {' '.join(pll_moves)}")
    print("-" * 60)
    print(f"TOTAL MOVE COUNT: {len(total_moves)}")
    print(f"TOTAL TIME TAKEN: {total_elapsed_time:.4f} seconds")
    print(f"FULL SOLUTION: {' '.join(total_moves)}")
    print(f"Final solved matrices written to: {output_filename}")
    print("=" * 60)

if __name__ == "__main__":
    main()