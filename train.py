import numpy as np
import time
import torch
import torch.nn.functional as F
from collections import deque, namedtuple
import gymnasium as gym

from model import DQN
import utils

# Hyperparameters
batch_size = 64
discount_factor = 0.995
learning_rate = 1e-3
update_interval = 4
memory_capacity = 100_000  # size of memory buffer
num_episodes = 2500
max_num_timesteps = 1000
num_points_for_average = 100

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminated"])


def compute_loss(experiences, discount_factor, main_dqn, target_dqn):
    states, actions, rewards, next_states, terminated = experiences

    with torch.no_grad():
        max_qsa, _ = torch.max(target_dqn(next_states), dim=-1)
        y_targets = rewards + (discount_factor * max_qsa * (1 - terminated))

    q_values = main_dqn(states)

    actions = actions.to(torch.int64)
    selected_q_values = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze()

    loss = F.mse_loss(selected_q_values, y_targets)

    return loss


def learn_agent(experiences, discount_factor):
    loss = compute_loss(experiences, discount_factor, main_dqn, target_dqn)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    utils.soft_update_target_network(main_dqn, target_dqn)


def train_agent():
    epsilon = 1.0  # initial ε value for ε-greedy policy
    total_point_history = []

    memory_buffer = deque(maxlen=memory_capacity)
    target_dqn.load_state_dict(main_dqn.state_dict())

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        for timestep in range(max_num_timesteps):
            state_tensor = torch.tensor(np.expand_dims(state, axis=0))
            q_values = main_dqn(state_tensor)
            action = utils.choose_action(q_values, epsilon)

            next_state, reward, terminated, _, _ = env.step(action)

            memory_buffer.append(experience(state, action, reward, next_state, terminated))

            update = utils.check_update_conditions(timestep, update_interval, memory_buffer)

            if update:
                experiences = utils.sample_batch_from_memory(memory_buffer)
                learn_agent(experiences, discount_factor=discount_factor)

            state = next_state.copy()
            total_reward += reward

            if terminated:
                break

        total_point_history.append(total_reward)
        average_latest_points = np.mean(total_point_history[-num_points_for_average:])
        epsilon = utils.calculate_new_epsilon(epsilon)

        print(
            f"\rEpisode {episode+1:3d} | "
            f"Avg. points of last {num_points_for_average} episodes: {average_latest_points:.2f} | "
            f"Epsilon: {epsilon:.4f}",
            end="",
        )

        if (episode + 1) % num_points_for_average == 0:
            print(
                f"\rEpisode {episode+1:3d} | "
                f"Avg. points of last {num_points_for_average} episodes: {average_latest_points:.2f} | "
                f"Epsilon: {epsilon:.4f}",
            )

        if average_latest_points >= 250.0:
            print(f"\n\nEnvironment solved in {episode+1} episodes")

            break

    env.close()


env = gym.make("LunarLander-v2")
state, info = env.reset(seed=42)

state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

main_dqn = DQN(state_size, num_actions)
target_dqn = DQN(state_size, num_actions)
optimizer = torch.optim.AdamW(main_dqn.parameters(), lr=learning_rate)

start = time.time()
train_agent()

total_runtime = time.time() - start
print(f"\nTotal Runtime: {total_runtime:.2f} s ({(total_runtime/60):.2f} min)")