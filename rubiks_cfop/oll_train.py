import torch
import torch.nn as nn
import torch.nn.functional as F

from oll_dataset import build_dataset
from oll_verify import FRAME_TOKENS, OLL_COUNT

EPOCHS = 2000
LEARNING_RATE = 1e-3
SEED = 0
CHECKPOINT = r"C:\Users\Advait Gawai\OneDrive\Desktop\Cube\rubiks_cfop\models\oll_model.pt"

class OLLNet(nn.Module):
    """One-hot cube -> shared trunk -> frame head (4) and algorithm head (57)."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(54 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.frame_head = nn.Linear(256, len(FRAME_TOKENS))
        self.alg_head = nn.Linear(256, OLL_COUNT)

    def forward(self, stickers):
        x = F.one_hot(stickers, num_classes=6).float().flatten(1)
        h = self.trunk(x)
        return self.frame_head(h), self.alg_head(h)


def frame_loss(frame_logits, frame_mask):
    """Train against the set of valid frames: maximise the total probability on them."""
    probs = F.softmax(frame_logits, dim=1)
    valid_mass = (probs * frame_mask).sum(dim=1).clamp_min(1e-9)
    return -torch.log(valid_mass).mean()


def main():
    torch.manual_seed(SEED)

    X_np, mask_np, labels_np, _meta, unsolvable = build_dataset()
    if unsolvable:
        print(f"Warning: {len(unsolvable)} states dropped (no valid frame)")
    X = torch.tensor(X_np)
    mask = torch.tensor(mask_np)
    labels = torch.tensor(labels_np)

    model = OLLNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        frame_logits, alg_logits = model(X)
        loss = frame_loss(frame_logits, mask) + F.cross_entropy(alg_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"epoch {epoch:5d}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        frame_logits, alg_logits = model(X)
        frame_pred = frame_logits.argmax(dim=1)
        frame_ok = mask[torch.arange(len(X)), frame_pred]
        alg_ok = alg_logits.argmax(dim=1) == labels

    print(f"Frame head accuracy (training data): {frame_ok.float().mean().item():.3f}")
    print(f"Algorithm head accuracy (training data): {alg_ok.float().mean().item():.3f}")

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"Saved {CHECKPOINT}")


if __name__ == "__main__":
    main()