import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from tqdm import tqdm
from experience_buffer import PrioritizedExperienceBuffer  # Import PrioritizedExperienceBuffer

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_input_dim = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc_value = nn.Linear(512, 1)
        self.fc_adv = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = F.relu(self.fc1(x))
        value = self.fc_value(x)
        adv = self.fc_adv(x)
        return value + (adv - adv.mean())

class DQNAgent:
    def __init__(self, env, state_shape, action_size, device, args):
        self.env = env
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.memory = PrioritizedExperienceBuffer(args.buffer_size, args.batch_size, self.device)
        self.gamma = args.gamma
        self.epsilon = args.epsilon_start
        self.epsilon_min = args.epsilon_end
        self.epsilon_decay = (args.epsilon_start - args.epsilon_end) / args.decay_end
        self.learning_rate = args.learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')
        self.model_save_path = args.path_to_trained_model
        self.log_interval = args.log_interval if hasattr(args, 'log_interval') else 100
        self.rewards = []
        self.losses = []

    def _build_model(self):
        return DuelingDQN(self.state_shape, self.action_size).to(self.device)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        experiences = self.memory.sample_experiences()
        if experiences is None:
            return None

        states, actions, rewards, next_states, terminals, indices, weights = experiences

        # Correcting the permutation of the states tensor
        states = torch.FloatTensor(np.array(states)).to(self.device).permute(0, 3, 1, 2)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device).permute(0, 3, 1, 2)
        terminals = torch.FloatTensor(terminals).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - terminals))

        loss = self.criterion(q_values, target_q_values.detach())
        weighted_loss = (weights * loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        self.memory.update_priorities(indices, loss.detach().cpu().numpy())

        return weighted_loss.item()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.model_save_path, name))

    def run_episode(self, batch_size):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        loss = 0
        rally_length = 0

        while not done:
            action = self.act(state)
            step_result = self.env.step(action)
            
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                next_state, reward, done, info = step_result
                truncated = False

            if reward > 0:
                reward += 1

            rally_length += 1
            if reward > 0:
                reward += rally_length * 0.01

            if done and not truncated:
                reward -= 2
                if rally_length > 10:
                    reward += rally_length * 0.05

            self.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(self.memory.buffer) > batch_size:
                loss = self.replay(batch_size)
                if loss is not None:
                    self.losses.append(loss)

        self.rewards.append(total_reward)
        return total_reward, loss

    def train(self, episodes, batch_size, save_interval=100):
        for e in tqdm(range(episodes), desc="Training"):
            reward, loss = self.run_episode(batch_size)
            if e % self.log_interval == 0:
                avg_reward = np.mean(self.rewards[-self.log_interval:])
                avg_loss = np.mean(self.losses[-self.log_interval:]) if self.losses else 0
                tqdm.write(f"Episode {e}/{episodes}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
            if e % save_interval == 0:
                self.save(f"model_episode_{e}.pth")
