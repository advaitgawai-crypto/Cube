import sys

from notations import Cube
from p import FACE_ORDER, cube_from_matrices, read_matrices, solve_pll, to_moves
from oll_algorithms import _expand

def format_matrices(cube):
    """Format the six faces of the cube as text matrices in standard FACE_ORDER."""
    lines = []
    for face_name in FACE_ORDER:
        face_arr = getattr(cube, face_name)
        for row in face_arr:
            lines.append(" ".join(str(val) for val in row))
        lines.append("")
    return "\n".join(lines).strip()

def main():
    try:
        text = read_matrices().strip()
    except OSError as error:
        print(f"\n[ERROR] Could not read the file: {error}")
        sys.exit(1)

    if not text:
        print("\n[ERROR] No input matrices provided. Exiting.")
        sys.exit(1)

    try:
        cube = cube_from_matrices(text)
    except ValueError as error:
        print(f"\n[ERROR] Invalid cube: {error}")
        sys.exit(1)

    moves = solve_pll(cube)

    if moves is None:
        print("Could not solve PLL.")
        sys.exit(1)

    # Apply the moves returned by solve_pll
    for move in moves:
        for name in _expand(move):
            getattr(cube, name)()

    # Save solved 6 matrices to p_solved_matrix.txt
    formatted_output = format_matrices(cube) + "\n"
    output_filename = "p_solved_matrix.txt"
    with open(output_filename, "w") as f:
        f.write(formatted_output)

    print(f"\nApplied PLL moves and saved updated matrices to {output_filename}.")

if __name__ == "__main__":
    main()