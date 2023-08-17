import numpy as np
import random
import torch

# Hyperparameter
MINIBATCH_SIZE = 64
SOFT_UPDATE_TAU = 1e-3
EPSILON_DECAY_RATE = 0.995
MIN_EPSILON = 0.01


def soft_update_target_network(main_dqn, target_dqn):
    """Update the target network's weights using soft update"""
    for target_weights, q_net_weights in zip(target_dqn.parameters(), main_dqn.parameters()):
        target_weights.data.copy_(
            SOFT_UPDATE_TAU * q_net_weights.data + (1.0 - SOFT_UPDATE_TAU) * target_weights.data
        )


def check_update_conditions(timestep, update_interval, memory_buffer):
    """Check if conditions are met for updating the network"""
    return (timestep + 1) % update_interval == 0 and len(memory_buffer) > MINIBATCH_SIZE


def sample_batch_from_memory(memory_buffer):
    """Retrieve a batch of experiences from the memory buffer"""
    batch = random.sample(memory_buffer, k=MINIBATCH_SIZE)  # Experiences
    states = torch.tensor([e.state for e in batch if e is not None], dtype=torch.float32)
    actions = torch.tensor([e.action for e in batch if e is not None], dtype=torch.float32)
    rewards = torch.tensor([e.reward for e in batch if e is not None], dtype=torch.float32)
    next_states = torch.tensor([e.next_state for e in batch if e is not None], dtype=torch.float32)
    terminated = torch.tensor([e.terminated for e in batch if e is not None], dtype=torch.float32)
    return states, actions, rewards, next_states, terminated


def calculate_new_epsilon(epsilon):
    """Calculate the new epsilon value for epsilon-greedy exploration"""
    return max(MIN_EPSILON, EPSILON_DECAY_RATE * epsilon)


def choose_action(q_values, epsilon=0.0):
    """Choose an action using epsilon-greedy exploration"""
    if random.random() > epsilon:
        return np.argmax(q_values.detach().numpy()[0])
    else:
        return random.choice(np.arange(4))
