import torch

from cube_utils import flatten_cube
from notations import Cube
from oll_verify import AUF_STEPS, AUF_TOKENS, FRAME_STEPS, FRAME_TOKENS, flip
from pll_train import PLLNet, joint_probs
from pll_verify import apply_triple, is_pll_solved


class PLLSolver:
    """Solves PLL with one model call: predicts a frame, pre-AUF, PLL algorithm
    and post-AUF as one whole combination.

    Like OLLSolver, there is no search. The model covers every PLL input
    (all cases, frames and pre-AUFs), so one prediction is enough.
    """

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PLLNet().to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()  # Turn off dropout

    def copy_cube(self, c):
        n = Cube.__new__(Cube)
        n.upper = c.upper.copy()
        n.front = c.front.copy()
        n.lower = c.lower.copy()
        n.right = c.right.copy()
        n.left = c.left.copy()
        n.back = c.back.copy()
        return n

    def solved(self, cube: Cube):
        return is_pll_solved(cube)

    def predict(self, cube: Cube):
        """Return (frame, pre, PLL number, post) for the most probable combination."""
        state_tensor = torch.tensor(flatten_cube(cube), dtype=torch.int64).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        with torch.no_grad():
            frame_l, pre_l, alg_l, post_l = self.net(state_tensor)
            probs = joint_probs(frame_l, pre_l, alg_l, post_l)[0]
        f, p, a, q = [int(i) for i in torch.unravel_index(probs.argmax(), probs.shape)]
        return FRAME_TOKENS[f], AUF_TOKENS[p], a + 1, AUF_TOKENS[q]

    def solve(self, scrambled_cube):
        """Return the path of tokens that solves PLL, or None.

        The path lists each non-'none' token in order: frame, pre-AUF,
        PLL algorithm, post-AUF. The input cube is not changed.
        """
        cube = self.copy_cube(scrambled_cube)
        if self.solved(cube):
            return []

        # Check if the cube is already in a PLL skip (just needs AUF: U, U', U2):
        for frame in FRAME_TOKENS:
            for auf in AUF_TOKENS:
                if frame == "none" and auf == "none":
                    continue
                trial = self.copy_cube(cube)
                flip(trial)
                for name in FRAME_STEPS[frame] + AUF_STEPS[auf]:
                    getattr(trial, name)()
                flip(trial)
                if self.solved(trial):
                    path = []
                    if frame != "none":
                        path.append(frame)
                    if auf != "none":
                        path.append(auf)
                    return path

        frame, pre, alg, post = self.predict(cube)
        apply_triple(cube, frame, pre, alg, post)

        if not self.solved(cube):
            return None

        path = []
        if frame != "none":
            path.append(frame)
        if pre != "none":
            path.append(pre)
        path.append(f"PLL_{alg}")
        if post != "none":
            path.append(post)
        return path
