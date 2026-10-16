import torch
import torch.nn as nn
import torch.nn.functional as F

class CubeNet(nn.Module):
    def __init__(self, hidden_sizes=(512, 512, 256), dropout_p=0.1):
        super().__init__()
        # 54 stickers, each one of 6 colors -> 324 inputs
        self.num_colors = 6
        input_size = 324

        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())           # the activation
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))  # Regularization
            prev_size = size

        layers.append(nn.Linear(prev_size, 1))   # output: one distance per state
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        # states: (batch, 54) integers 0-5
        onehot = F.one_hot(states, num_classes=self.num_colors)   # the states tensor
        x = onehot.reshape(states.shape[0], -1)    # keep batch size, let -1 handle the rest
        x = x.float()
        return self.net(x)       # shape: (batch, 1)