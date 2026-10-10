import time
import copy
import h5py
import numpy as np
import random as rd
from collections import deque

from notations import Cube
from cube_utils import flatten_cube

MOVES = [
    "R", "R_Prime", "L", "L_Prime",
    "U", "U_Prime", "D", "D_Prime",
    "F", "F_Prime", "B", "B_Prime"
]

def f2l_solved(cube: Cube) -> bool:
# 1it i s as per the upper face, change altogether if needed during debugging
    # 1. Check if the Upper face (cross + corners) is completely solved
    if not np.all(cube.upper == cube.upper[1, 1]):
        return False

# 2. Check top two rows of all four surrounding faces
    faces = [cube.front, cube.right, cube.back, cube.left]
    for face in faces:
        center = face[1, 1]
        if not np.all(face[0:2, :] == center):
            return False

    return True


def random_f2l_walk(walk_len=8):
    """
    Starts from a fully solved cube (which means F2L is already solved),
    takes sequential random turns, and records (flattened_state, distance).
    """
    cube = Cube()
    samples = []
    
    for step in range(1, walk_len + 1):
        move = rd.choice(MOVES)
        getattr(cube, move)()
        samples.append((flatten_cube(cube), step))
        
    return samples


def main():
    all_samples = []
    total_walks = 6250   # 6,250 * 8 moves = 50,000 samples
    walk_len = 8
    
    print("Generating F2L training data...")
    start_time = time.perf_counter()

    for i in range(1, total_walks + 1):
        all_samples.extend(random_f2l_walk(walk_len))
        
        if i % 500 == 0:
            elapsed = time.perf_counter() - start_time
            time_left = (elapsed / i) * (total_walks - i)
            print(f"{i}/{total_walks} walks done, ~{time_left:.1f}s remaining", flush=True)

    states_list, distances_list = zip(*all_samples)
    states = np.array(states_list)
    distances = np.array(distances_list)

    # Save to HDF5
    output_path = "data/f2l_data.h5"
    with h5py.File(output_path, "w") as f:
        f.create_dataset("states", data=states)
        f.create_dataset("distances", data=distances)

    print(f"\nDone! Saved to {output_path}")
    print(f"States shape: {states.shape}, Distances shape: {distances.shape}")


if __name__ == "__main__":
    main()