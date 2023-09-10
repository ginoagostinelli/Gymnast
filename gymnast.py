import os
import torch
from torch import nn


class Gymnast(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()

        self.dqn = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.dqn(x)

    def from_pretrained(self, model_path: str) -> None:
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'model not found in "{model_path}"')

        self.load_state_dict(torch.load(model_path))
