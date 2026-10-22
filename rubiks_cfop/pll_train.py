import torch
import torch.nn as nn
import torch.nn.functional as F

from oll_verify import AUF_TOKENS, FRAME_TOKENS
from pll_dataset import load_or_build
from pll_verify import PLL_COUNT

EPOCHS = 2000
LEARNING_RATE = 1e-3
SEED = 0
CHECKPOINT = r"C:\Users\Advait Gawai\OneDrive\Desktop\Cube\rubiks_cfop\models\pll_model.pt"


class PLLNet(nn.Module):
    """One-hot cube -> shared trunk -> frame, pre-AUF, PLL and post-AUF heads."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(54 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.frame_head = nn.Linear(256, len(FRAME_TOKENS))
        self.pre_head = nn.Linear(256, len(AUF_TOKENS))
        self.alg_head = nn.Linear(256, PLL_COUNT)
        self.post_head = nn.Linear(256, len(AUF_TOKENS))

    def forward(self, stickers):
        x = F.one_hot(stickers, num_classes=6).float().flatten(1)
        h = self.trunk(x)
        return (
            self.frame_head(h),
            self.pre_head(h),
            self.alg_head(h),
            self.post_head(h),
        )


def joint_probs(frame_l, pre_l, alg_l, post_l):
    """Probability of every whole combination, shape (N, 4, 4, 21, 4)."""
    pf = F.softmax(frame_l, dim=1)
    pp = F.softmax(pre_l, dim=1)
    pa = F.softmax(alg_l, dim=1)
    pq = F.softmax(post_l, dim=1)
    return (
        pf[:, :, None, None, None]
        * pp[:, None, :, None, None]
        * pa[:, None, None, :, None]
        * pq[:, None, None, None, :]
    )


def joint_loss(frame_l, pre_l, alg_l, post_l, joint_mask):
    """Maximise the total probability on the valid whole combinations."""
    probs = joint_probs(frame_l, pre_l, alg_l, post_l)
    valid_mass = (probs * joint_mask).sum(dim=(1, 2, 3, 4)).clamp_min(1e-9)
    return -torch.log(valid_mass).mean()


def main():
    torch.manual_seed(SEED)

    X_np, joint_np, unsolvable = load_or_build()
    if unsolvable:
        print(f"Warning: {len(unsolvable)} states dropped (no valid combination)")

    X = torch.tensor(X_np)
    joint_m = torch.tensor(joint_np)

    model = PLLNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        frame_l, pre_l, alg_l, post_l = model(X)
        loss = joint_loss(frame_l, pre_l, alg_l, post_l, joint_m)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"epoch {epoch:5d}  loss {loss.item():.4f}")

    model.eval()
    rows = torch.arange(len(X))
    with torch.no_grad():
        frame_l, pre_l, alg_l, post_l = model(X)

        # Each head's own choice, checked as a whole combination.
        f_idx = frame_l.argmax(1)
        p_idx = pre_l.argmax(1)
        a_idx = alg_l.argmax(1)
        q_idx = post_l.argmax(1)
        argmax_ok = joint_m[rows, f_idx, p_idx, a_idx, q_idx]

        # The solver's rule: the single most probable whole combination.
        probs = joint_probs(frame_l, pre_l, alg_l, post_l).reshape(len(X), -1)
        best = probs.argmax(1)
        best_ok = joint_m.reshape(len(X), -1)[rows, best]

    print(f"Per-head argmax is a valid whole combination: {argmax_ok.float().mean().item():.3f}")
    print(f"Most probable whole combination is valid: {best_ok.float().mean().item():.3f}")

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"Saved {CHECKPOINT}")


if __name__ == "__main__":
    main()
