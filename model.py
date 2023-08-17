from torch import nn


class DQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()

        self.dqn = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.dqn(x)
