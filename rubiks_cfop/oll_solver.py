import torch

from cube_utils import flatten_cube
from notations import Cube
from oll_train import OLLNet
from oll_verify import FRAME_TOKENS, apply_triple, is_oll_solved


class OLLSolver:
    """Solves OLL with one model call: predicts a frame and an OLL algorithm.

    Unlike CrossSolver, there is no A* search. The model covers every OLL input
    (all cases, pre-turns and rotations), so one prediction is enough.
    """

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = OLLNet().to(self.device)
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
        return is_oll_solved(cube)

    def predict(self, cube: Cube):
        """Return (frame token, OLL number) for this cube."""
        state_tensor = torch.tensor(flatten_cube(cube), dtype=torch.int64).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        with torch.no_grad():
            frame_logits, alg_logits = self.net(state_tensor)
        frame = FRAME_TOKENS[frame_logits.argmax(dim=1).item()]
        alg = alg_logits.argmax(dim=1).item() + 1
        return frame, alg

    def solve(self, scrambled_cube):
        """Return the path of tokens that solves OLL, or None.

        The path is the frame token (if not 'none') followed by the OLL token.
        The input cube is not changed.
        """
        cube = self.copy_cube(scrambled_cube)
        if self.solved(cube):
            return []

        frame, alg = self.predict(cube)
        apply_triple(cube, frame, "none", alg)

        if not self.solved(cube):
            return None

        path = []
        if frame != "none":
            path.append(frame)
        path.append(f"OLL_{alg}")
        return path
