import heapq
import torch
import torch.nn as nn
from cube_utils import flatten_cube
from notations import Cube


class PLLNet(nn.Module):
    """Matches the saved checkpoint: 324 -> 512 -> 512 -> 256 -> 1."""

    def __init__(self, num_colors=6):
        super().__init__()
        self.num_colors = num_colors
        self.net = nn.Sequential(
            nn.Linear(54 * num_colors, 512),  # net.0
            nn.ReLU(),                        # net.1
            nn.Linear(512, 512),              # net.2
            nn.ReLU(),                        # net.3
            nn.Linear(512, 256),              # net.4
            nn.ReLU(),                        # net.5
            nn.Linear(256, 1),                # net.6
        )

    def forward(self, states):
        onehot = nn.functional.one_hot(states, num_classes=self.num_colors).float()
        onehot = onehot.view(onehot.size(0), -1)
        return self.net(onehot)


class PLLSolver:

    FACE_MOVES = ["R", "R_Prime", "L", "L_Prime", "U", "U_Prime",
                  "D", "D_Prime", "F", "F_Prime", "B", "B_Prime"]

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

    def heuristic(self, cube):
        state_vector = flatten_cube(cube)
        state_tensor = torch.tensor(state_vector, dtype=torch.int64).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        with torch.no_grad():
            prediction = self.net(state_tensor)
        return prediction.item()

    def solved(self, cube: Cube):
        faces = [cube.upper, cube.front, cube.lower, cube.right, cube.left, cube.back]
        return bool(all((face == face[1, 1]).all() for face in faces))

    def get_neighbors(self, cube):
        neighbors = []
        for move in self.FACE_MOVES:
            new_cube = self.copy_cube(cube)
            getattr(new_cube, move)()
            neighbors.append((new_cube, move))
        return neighbors

    def solve(self, scrambled_cube, max_iterations=100000):
        open_set = []
        counter = 0

        start_f = self.heuristic(scrambled_cube)
        heapq.heappush(open_set, (start_f, counter, scrambled_cube, []))
        counter += 1

        closed_set = set()
        g_scores = {}
        start_bytes = flatten_cube(scrambled_cube).tobytes()
        g_scores[start_bytes] = 0

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            f, _, current_cube, path = heapq.heappop(open_set)
            current_bytes = flatten_cube(current_cube).tobytes()

            if current_bytes in closed_set:
                continue
            closed_set.add(current_bytes)

            if self.solved(current_cube):
                return path

            for neighbor, move in self.get_neighbors(current_cube):
                neighbor_bytes = flatten_cube(neighbor).tobytes()
                if neighbor_bytes in closed_set:
                    continue
                tentative_g = g_scores[current_bytes] + 1
                if neighbor_bytes not in g_scores or tentative_g < g_scores[neighbor_bytes]:
                    g_scores[neighbor_bytes] = tentative_g
                    h = self.heuristic(neighbor)
                    heapq.heappush(open_set, (tentative_g + h, counter, neighbor, path + [move]))
                    counter += 1

        return None