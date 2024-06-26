import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in indices])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def size(self):
        return len(self.buffer)
