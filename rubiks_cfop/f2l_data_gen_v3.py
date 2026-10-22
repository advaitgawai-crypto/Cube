import os
import random
import h5py
import numpy as np
from notations import Cube
from f2l_algorithms import F2L_ALGORITHMS

# 1. Map notation strings to Cube methods
MOVE_MAP = {
    "R": lambda c: c.R(),
    "R'": lambda c: c.R_Prime(),
    "L": lambda c: c.L(),
    "L'": lambda c: c.L_Prime(),
    "U": lambda c: c.U(),
    "U'": lambda c: c.U_Prime(),
    "D": lambda c: c.D(),
    "D'": lambda c: c.D_Prime(),
    "F": lambda c: c.F(),
    "F'": lambda c: c.F_Prime(),
    "B": lambda c: c.B(),
    "B'": lambda c: c.B_Prime(),
    "X": lambda c: c.X(),
    "X'": lambda c: c.X_prime(),
    "Y": lambda c: c.Y(),
    "y": lambda c: c.Y(),
    "Y'": lambda c: c.Y_prime(),
    "y'": lambda c: c.Y_prime(),
    "Z": lambda c: c.Z(),
    "Z'": lambda c: c.Z_prime(),
    # 'd' (Dw) translates mechanically to D then Y
    "d": lambda c: (c.D(), c.Y()),
    "d'": lambda c: (c.D_Prime(), c.Y_prime()),
}

def apply_move(cube, move_token):
    """Executes a single move notation, handling double turns (e.g. X2, U2)."""
    if move_token.endswith("2"):
        base = move_token[:-1]
        for _ in range(2):
            MOVE_MAP[base](cube)
    else:
        MOVE_MAP[move_token](cube)

def apply_sequence(cube, seq_str):
    """Executes a space-separated sequence of moves."""
    if not seq_str.strip():
        return
    for token in seq_str.strip().split():
        apply_move(cube, token)

def invert_token(token):
    """Inverts an individual move notation."""
    if token.endswith("2"):
        return token
    if token.endswith("'"):
        return token[:-1]
    return token + "'"

def invert_alg(alg_str):
    """Reverses and inverts an algorithm to create the scramble setup."""
    tokens = alg_str.strip().split()
    return " ".join([invert_token(t) for t in reversed(tokens)])

def cube_to_feature_vector(cube):
    """
    Extracts all 54 stickers and converts them into a 1D one-hot vector (54 * 6 = 324 features).
    """
    faces = [cube.upper, cube.front, cube.lower, cube.right, cube.left, cube.back]
    stickers = np.concatenate([f.flatten() for f in faces])  # shape: (54,)
    
    # One-hot encode the 6 color indices (0 through 5)
    one_hot = np.eye(6, dtype=np.float32)[stickers].flatten()  # shape: (324,)
    return one_hot

def generate_f2l_dataset(num_samples=50000, output_dir=None, output_filename="f2l_dataset.h5"):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    # Ensure the target folder exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    X = []
    y = []

    alg_keys = sorted(list(F2L_ALGORITHMS.keys()))
    num_classes = len(alg_keys)
    
    auf_options = ["", "U", "U'", "U2"]
    slot_rotations = ["", "Y", "Y'", "Y2"]

    print(f"Generating {num_samples} F2L samples across {num_classes} cases...")

    for i in range(num_samples):
        # 1. Initialize cube with solved cross on top face
        cube = Cube()
        
        # 2. Rotate X2 to place cross on bottom face before F2L
        apply_sequence(cube, "X2")

        # 3. Select target algorithm class
        label_idx = random.randint(0, num_classes - 1)
        base_alg = F2L_ALGORITHMS[alg_keys[label_idx]]

        # 4. Pick random AUF and slot orientation
        pre_u = random.choice(auf_options)
        slot_y = random.choice(slot_rotations)

        # 5. Build setup scramble (inverted algorithm + inverted AUF)
        setup_moves = invert_alg(base_alg)
        if pre_u:
            setup_moves += " " + invert_token(pre_u)

        # 6. Apply slot rotation and scramble moves
        if slot_y:
            apply_sequence(cube, slot_y)
        apply_sequence(cube, setup_moves)

        # 7. Convert state to feature vector
        state_vec = cube_to_feature_vector(cube)
        X.append(state_vec)
        y.append(label_idx)

        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Save datasets into .h5 format with gzip compression
    print(f"Saving data to {output_path}...")
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("X", data=X, compression="gzip")
        hf.create_dataset("y", data=y, compression="gzip")

    print(f"Done! Successfully saved to {output_path} | X: {X.shape}, y: {y.shape}")

if __name__ == "__main__":
    generate_f2l_dataset(num_samples=50000)