import os
from model import DQN
import numpy as np
import gymnasium as gym
import torch

output_dir = os.path.join(os.path.dirname(__file__), "output")

env = gym.make("LunarLander-v2", render_mode="human")
state, info = env.reset(seed=42)

state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

dqn = DQN(state_size, num_actions)
model_path = os.path.join(output_dir, "model.pt")
dqn.from_pretrained(model_path)

while True:
    state = torch.tensor(np.expand_dims(state, axis=0))
    q_values = dqn(state)
    action = np.argmax(q_values.detach().numpy()[0])

    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
