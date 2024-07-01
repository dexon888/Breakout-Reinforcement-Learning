import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from dqn_model import DQN


class DQNAgent:
    def __init__(self, env, state_shape, action_size, device):
        self.env = env
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.model = DQN(
            self.device, in_channels=state_shape[2], num_actions=action_size).to(self.device)
        self.target_model = DQN(
            self.device, in_channels=state_shape[2], num_actions=action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                next_state = torch.FloatTensor(
                    next_state).unsqueeze(0).to(self.device)
                target = reward + self.gamma * \
                    torch.max(self.target_model(next_state)[0]).item()
            target_f = self.model(state).detach()
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)

        states = torch.cat(states)
        targets = torch.cat(targets).detach()

        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(states)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def run_episode(self, batch_size):
        state, info = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:
            action = self.act(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            self.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done or truncated:
                self.update_target_model()
                break
        if len(self.memory) > batch_size:
            self.replay(batch_size)
        return total_reward

    def train(self, episodes, batch_size):
        for e in range(episodes):
            reward = self.run_episode(batch_size)
            if e % 10 == 0:
                print(f"Episode {e}/{episodes}, Reward: {reward}")


if __name__ == "__main__":
    import gymnasium as gym
    from atari_wrapper import wrap_deepmind

    env = gym.make('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env)
    state_shape = env.observation_space.shape
    print(f"State shape: {state_shape}")
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env, state_shape, action_size, device)
    episodes = 1000
    batch_size = 32

    agent.train(episodes, batch_size)
