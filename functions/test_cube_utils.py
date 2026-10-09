# test_cube_utils.py
import numpy as np
from notations import Cube
from cube_utils import flatten_cube, unflatten_cube

FACES = ["upper", "front", "lower", "back", "left", "right"]

def cubes_equal(c1, c2):
    return all(np.array_equal(getattr(c1, f), getattr(c2, f)) for f in FACES)

def test_flatten_length():
    flat = flatten_cube(Cube())
    assert flat.shape == (54,)

def test_solved_cube_values():
    flat = flatten_cube(Cube())
    # order: upper, front, lower, back, left, right -> colors 0, 1, 2, 5, 4, 3
    expected = np.repeat([0, 1, 2, 5, 4, 3], 9)
    assert np.array_equal(flat, expected)

def test_round_trip_solved():
    cube = Cube()
    assert cubes_equal(unflatten_cube(flatten_cube(cube)), cube)

def test_round_trip_after_move():
    cube = Cube()
    cube.R()
    assert cubes_equal(unflatten_cube(flatten_cube(cube)), cube)

def test_unflatten_is_independent():
    flat = flatten_cube(Cube())
    new = unflatten_cube(flat)
    flat[:] = 9  # change the source array
    assert np.all(new.upper == 0)  # the cube should not change

def test_move_changes_expected_stickers():
    cube = Cube()
    cube.R()
    flat = flatten_cube(cube)

    expected = np.repeat([0, 1, 2, 5, 4, 3], 9)
    # R moves the right column (col 2) of upper/front/lower and col 0 of back
    expected[[2, 5, 8]] = 1       # upper col 2 <- front's color
    expected[[11, 14, 17]] = 2    # front col 2 <- lower's color
    expected[[20, 23, 26]] = 5    # lower col 2 <- back's color
    expected[[27, 30, 33]] = 0    # back col 0 <- upper's color
    # left (36-44) and right (45-53) stay all 4s and all 3s

    assert np.array_equal(flat, expected)

def test_four_R_moves_return_to_solved():
    cube = Cube()
    for _ in range(4):
        cube.S()
    assert np.array_equal(flatten_cube(cube), flatten_cube(Cube()))