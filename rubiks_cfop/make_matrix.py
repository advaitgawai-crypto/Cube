"""Turn moves into six 3x3 matrices and save them to matrix.txt.

Moves are applied to a solved cube with yellow on top, the same way oll_test.py and
pll_test.py read moves. The saved matrices use the notations.py layout, so they can
be fed straight back into c.py, oll_test.py or pll_test.py.

Usage:
    python make_matrix.py                      asks for moves
    python make_matrix.py "R U R' U' R' F R F'"  moves in quotes (PowerShell needs them)
"""

import os
import sys

from notations import Cube
from oll_algorithms import _expand
from oll_verify import flip

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrix.txt")

# Same order that Cube.display() prints the faces, and that the test tools read.
FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]


def cube_from_moves(text):
    """Apply moves to a solved cube with yellow on top. Raises ValueError on a bad move."""
    tokens = text.split()
    for token in tokens:
        _expand(token)  # raises ValueError on an unknown move

    cube = Cube()
    flip(cube)
    for token in tokens:
        for name in _expand(token):
            getattr(cube, name)()
    flip(cube)
    return cube


def matrices_text(cube):
    """Six 3x3 matrices, one row per line, faces separated by a blank line."""
    faces = []
    for name in FACE_ORDER:
        matrix = getattr(cube, name)
        faces.append("\n".join(" ".join(str(int(v)) for v in row) for row in matrix))
    return "\n\n".join(faces) + "\n"


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter your moves (blank = solved cube): ")
    text = text.strip()

    try:
        cube = cube_from_moves(text)
    except ValueError as error:
        print(f"Could not read the moves: {error}")
        sys.exit(1)

    with open(OUTPUT_PATH, "w") as file:
        file.write(matrices_text(cube))

    print(f"Saved the six matrices to {OUTPUT_PATH}")
    print("Use it with: python oll_test.py matrix.txt")


if __name__ == "__main__":
    main()
