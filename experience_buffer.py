from collections import deque, namedtuple
import random
import numpy as np
import torch
from operator import itemgetter

class PrioritizedExperienceBuffer:
    def __init__(self, maxlen, minibatch_size, device, alpha=0.6):
        self.max_buffer_capacity = maxlen  # max number of experiences
        self.minibatch_size = minibatch_size  # number of experiences to sample
        self.buffer = deque(maxlen=maxlen)  # buffer to store experiences
        self.priorities = deque(maxlen=maxlen)  # buffer to store priorities
        self.device = device  # device to store experiences
        self.alpha = alpha  # how much prioritization is used (0 - no prioritization, 1 - full prioritization)

    def is_full(self):
        # Check if buffer is full
        return len(self.buffer) == self.max_buffer_capacity

    def push(self, experience_tuple):  # (s, a, r, s', is_terminal)
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(experience_tuple)
        self.priorities.append(max_priority)

    def sample_experiences(self, beta=0.4):
        """
        Select batch from buffer.
        """
        if len(self.buffer) < self.minibatch_size:
            return None

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), self.minibatch_size, p=probabilities)
        sampled_tuples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, terminals = [], [], [], [], []

        for (state, action, reward, next_state, terminal) in sampled_tuples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(terminal)

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        return [states, actions, rewards, next_states, terminals, indices, weights]

    def update_priorities(self, indices, td_errors, epsilon=1e-5):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + epsilon
