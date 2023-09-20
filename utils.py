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
    states = torch.tensor([e.state for e in batch if e is not None], dtype=torch.float32)
    actions = torch.tensor([e.action for e in batch if e is not None], dtype=torch.float32)
    rewards = torch.tensor([e.reward for e in batch if e is not None], dtype=torch.float32)
    next_states = torch.tensor([e.next_state for e in batch if e is not None], dtype=torch.float32)
    terminated = torch.tensor([e.terminated for e in batch if e is not None], dtype=torch.float32)
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
