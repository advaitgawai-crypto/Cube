import torch

from cube_utils import flatten_cube
from oll_dataset import ROTATIONS, AUF_SCRAMBLES, case_state
from oll_train import OLLNet
from oll_verify import FRAME_TOKENS, OLL_COUNT, apply_triple, is_oll_solved
from notations import Cube

CHECKPOINT = r"C:\Users\Advait Gawai\OneDrive\Desktop\Cube\rubiks_cfop\models\oll_model.pt"

def load_model():
    model = OLLNet()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    return model


def solve_oll(model, cube):
    """Solve OLL in place with one model call: predict frame and algorithm, apply both.

    The AUF is always 'none', because all 912 training states are solvable with
    frame + algorithm alone. Returns (cube, solved_bool).
    """
    if is_oll_solved(cube):
        return cube, True

    x = torch.tensor(flatten_cube(cube), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        frame_logits, alg_logits = model(x)
    frame = FRAME_TOKENS[frame_logits.argmax(dim=1).item()]
    alg = alg_logits.argmax(dim=1).item() + 1

    apply_triple(cube, frame, "none", alg)
    return cube, is_oll_solved(cube)


def main():
    model = load_model()

    total = 0
    failures = []

    # Every input the model was trained on: all cases x pre-turns x rotations.
    for n in range(1, OLL_COUNT + 1):
        for rotation in ROTATIONS:
            for auf_scramble in AUF_SCRAMBLES:
                cube = case_state(n, rotation, auf_scramble)
                _, solved = solve_oll(model, cube)
                total += 1
                if not solved:
                    failures.append((n, rotation, auf_scramble))

    # The already-solved OLL state. The solver should return it unchanged.
    _, solved = solve_oll(model, Cube())
    total += 1
    if not solved:
        failures.append(("solved", None, None))

    print(f"Solved {total - len(failures)} of {total} inputs.")
    if failures:
        print("Failures (OLL case, rotation, pre-turn):")
        for f in failures[:20]:
            print("   ", f)


if __name__ == "__main__":
    main()
