import os
import sys
import numpy as np

from c import read_matrices, cube_from_matrices, solve_cross, FACE_ORDER

def format_matrices(cube):
    """Format the six faces of the cube as text matrices in standard FACE_ORDER."""
    lines = []
    for face_name in FACE_ORDER:
        face_arr = getattr(cube, face_name)
        for row in face_arr:
            lines.append(" ".join(str(val) for val in row))
        lines.append("") # blank line between faces/sections optional or clean spacing
    return "\n".join(lines).strip()

def main():
    try:
        text = read_matrices().strip()
    except OSError as error:
        print(f"Could not read the file: {error}")
        sys.exit(1)

    if not text:
        print("No input matrices provided. Exiting.")
        sys.exit(1)

    try:
        cube = cube_from_matrices(text)
    except ValueError as error:
        print(f"Invalid cube: {error}")
        sys.exit(1)

    moves = solve_cross(cube)

    if moves is None:
        print("Could not solve cross.")
        sys.exit(1)

    # Apply the moves suggested/found by c.py (solve_cross)
    for move in moves:
        getattr(cube, move)()

    # Save solved 6 matrices to c_solved_matrix.txt
    formatted_output = format_matrices(cube) + "\n"
    output_filename = "c_solved_matrix.txt"
    with open(output_filename, "w") as f:
        f.write(formatted_output)

    print(f"Applied {len(moves)} moves to solve cross and saved updated matrices to {output_filename}.")

if __name__ == "__main__":
    main()
