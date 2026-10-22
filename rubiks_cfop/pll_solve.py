import torch

from cube_utils import flatten_cube
from notations import Cube
from oll_verify import AUF_TOKENS, FRAME_TOKENS
from pll_train import CHECKPOINT, PLLNet, joint_probs
from pll_verify import PLL_COUNT, apply_triple, inverse_case_state, is_pll_solved


def load_model():
    model = PLLNet()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    return model


def solve_pll(model, cube):
    """Solve PLL in place with one model call.

    The solver picks the single most probable whole combination of
    (frame, pre-AUF, PLL algorithm, post-AUF), then applies it.
    Returns (cube, solved_bool). The cube stays in the notations.py layout.
    """
    if is_pll_solved(cube):
        return cube, True

    x = torch.tensor(flatten_cube(cube), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        frame_l, pre_l, alg_l, post_l = model(x)
        probs = joint_probs(frame_l, pre_l, alg_l, post_l)[0]

    f, p, a, q = [int(i) for i in torch.unravel_index(probs.argmax(), probs.shape)]
    frame = FRAME_TOKENS[f]
    pre = AUF_TOKENS[p]
    alg = a + 1
    post = AUF_TOKENS[q]

    apply_triple(cube, frame, pre, alg, post)
    return cube, is_pll_solved(cube)


def main():
    model = load_model()

    total = 0
    failures = []

    # Every PLL input: 21 cases x 4 frames x 4 pre-AUFs.
    for n in range(1, PLL_COUNT + 1):
        for frame in FRAME_TOKENS:
            for pre in AUF_TOKENS:
                cube = inverse_case_state(n, frame, pre, "none")
                _, solved = solve_pll(model, cube)
                total += 1
                if not solved:
                    failures.append((n, frame, pre))

    # The already-solved state. The solver should return it unchanged.
    _, solved = solve_pll(model, Cube())
    total += 1
    if not solved:
        failures.append(("solved", None, None))

    print(f"Solved {total - len(failures)} of {total} inputs.")
    if failures:
        print("Failures (PLL case, frame, pre-AUF):")
        for f in failures[:20]:
            print("   ", f)


if __name__ == "__main__":
    main()
