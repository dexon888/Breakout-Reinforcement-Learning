import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
import os
from tqdm import tqdm

class DQNAgent:
    def __init__(self, env, state_shape, action_size, device, args):
        self.env = env
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=args.buffer_size)
        self.gamma = args.gamma  # discount rate
        self.epsilon = args.epsilon_start  # exploration rate
        self.epsilon_min = args.epsilon_end
        self.epsilon_decay = (args.epsilon_start - args.epsilon_end) / args.decay_end
        self.learning_rate = args.learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.model_save_path = args.path_to_trained_model
        self.log_interval = args.log_interval if hasattr(args, 'log_interval') else 100  # Default logging interval

    def _build_model(self):
        model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        ).to(self.device)
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0]).item()
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
            self.epsilon -= self.epsilon_decay

        return loss.item()  # Return the loss value for logging

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, name))

    def run_episode(self, batch_size):
        state, _ = self.env.reset()
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
            loss = self.replay(batch_size)
            return total_reward, loss
        return total_reward, None

    def train(self, episodes, batch_size, save_interval=100):
        rewards = []
        losses = []
        for e in tqdm(range(episodes), desc="Training"):
            reward, loss = self.run_episode(batch_size)
            rewards.append(reward)
            if loss is not None:
                losses.append(loss)
            if e % self.log_interval == 0:
                avg_reward = np.mean(rewards[-self.log_interval:])
                avg_loss = np.mean(losses[-self.log_interval:]) if losses else 0
                tqdm.write(f"Episode {e}/{episodes}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
            if e % save_interval == 0:
                self.save(f"model_episode_{e}.pth")