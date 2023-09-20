import numpy as np
import random
import torch


def soft_update_target_network(main_gymnast, target_gymnast, soft_update_tau):
    """Update the target network's weights using soft update"""
    for target_weights, q_net_weights in zip(target_gymnast.parameters(), main_gymnast.parameters()):
        target_weights.data.copy_(
            soft_update_tau * q_net_weights.data + (1.0 - soft_update_tau) * target_weights.data
        )


def sample_batch_from_memory(batch_size, memory_buffer):
    """Retrieve a batch of experiences from the memory buffer"""
    batch = random.sample(memory_buffer, k=batch_size)  # Experiences

    # Collect data into separate lists
    states_list, actions_list, rewards_list, next_states_list, terminated_list = [], [], [], [], []

    for e in batch:
        if e is not None:
            states_list.append(e.state)
            actions_list.append(e.action)
            rewards_list.append(e.reward)
            next_states_list.append(e.next_state)
            terminated_list.append(e.terminated)

    # Convert the lists to numpy arrays
    states = torch.tensor(np.array(states_list), dtype=torch.float32)
    actions = torch.tensor(np.array(actions_list), dtype=torch.float32)
    rewards = torch.tensor(np.array(rewards_list), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states_list), dtype=torch.float32)
    terminated = torch.tensor(np.array(terminated_list), dtype=torch.float32)

    return states, actions, rewards, next_states, terminated


def calculate_new_epsilon(epsilon, min_epsilon, epsilon_decay_rate):
    """Calculate the new epsilon value for epsilon-greedy exploration"""
    return max(min_epsilon, epsilon_decay_rate * epsilon)


def choose_action(q_values, epsilon=0.0, num_actions=None):
    """Choose an action using epsilon-greedy exploration"""
    if random.random() > epsilon:
        return np.argmax(q_values.detach().numpy()[0])
    else:
        return random.choice(np.arange(num_actions))
