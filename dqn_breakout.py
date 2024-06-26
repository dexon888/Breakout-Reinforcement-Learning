import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt

# Define wrappers
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        truncated = None
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, truncated, _ = self.env.step(1)
        if done or truncated:
            self.env.reset(**kwargs)
        obs, _, done, truncated, _ = self.env.step(2)
        if done or truncated:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.override_num_noops if self.override_num_noops is not None else np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, truncated, _ = self.env.step(self.noop_action)
            if done or truncated:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

def make_atari(env_id):
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_deepmind(env, clip_rewards=True, frame_stack=False, scale=False):
    env = FireResetEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env

# Detect GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU detected. Using GPU.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected. Using CPU.")

# Hyperparameters
learning_rate = 0.00025
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 1000000
batch_size = 32
memory_size = 1000000
update_target_freq = 10000
num_episodes = 5000
max_episode_steps = 2000  # Default maximum number of steps per episode
eval_interval = 100  # Evaluate every 100 episodes
eval_episodes = 10  # Number of episodes to run during evaluation

# Preprocess frame using NumPy
def preprocess_frame(frame):
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = frame[34:194]  # Crop the image
        frame = frame.mean(2)  # Convert to grayscale
    else:
        frame = frame[34:194]  # Crop the image if it is already grayscale
    frame = frame[::2, ::2]  # Downsample by factor of 2
    return frame.astype(np.float32).reshape(80, 80, 1) / 255.0  # Normalize pixel values

# Stack frames
def stack_frames(frames):
    return np.stack(frames, axis=2)

# Define the DQN
def build_dqn(input_shape, num_actions):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Initialize environment, replay memory, and networks
env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env)
state = preprocess_frame(env.reset())
state_stack = deque([state] * 4, maxlen=4)

memory = deque(maxlen=memory_size)
primary_network = build_dqn((80, 80, 4), env.action_space.n)
target_network = build_dqn((80, 80, 4), env.action_space.n)
target_network.set_weights(primary_network.get_weights())

# Training loop
epsilon = epsilon_start
total_rewards = []
avg_eval_rewards = []
episodes = []

for episode in range(num_episodes):
    state = preprocess_frame(env.reset())
    state_stack = deque([state] * 4, maxlen=4)
    total_reward = 0
    
    for t in range(max_episode_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = primary_network.predict(np.expand_dims(stack_frames(state_stack), axis=0))
            action = np.argmax(q_values)
        
        next_state, reward, done, truncated, info = env.step(action)  # Updated to unpack info
        next_state = preprocess_frame(next_state)
        state_stack.append(next_state)
        memory.append((stack_frames(state_stack.copy()), action, reward, stack_frames(state_stack), done))
        
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = np.array(states)
            next_states = np.array(next_states)
            q_values_next = target_network.predict(next_states)
            targets = rewards + gamma * np.max(q_values_next, axis=1) * (1 - np.array(dones))
            
            q_values = primary_network.predict(states)
            for i, action in enumerate(actions):
                q_values[i][action] = targets[i]
            
            primary_network.fit(states, q_values, epochs=1, verbose=0)
        
        if t % update_target_freq == 0:
            target_network.set_weights(primary_network.get_weights())
        
        state = next_state
        total_reward += reward
        
        if done or truncated:
            break
    
    epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
    total_rewards.append(total_reward)
    
    # Print training progress and evaluate
    if episode % eval_interval == 0:
        avg_reward = np.mean(total_rewards[-eval_interval:])
        print(f"Episode: {episode}, Average Reward (last {eval_interval} episodes): {avg_reward}, Epsilon: {epsilon}")
        
        # Evaluation
        eval_rewards = []
        for eval_episode in range(eval_episodes):
            state = preprocess_frame(env.reset())
            state_stack = deque([state] * 4, maxlen=4)
            eval_total_reward = 0
            
            while True:
                q_values = primary_network.predict(np.expand_dims(stack_frames(state_stack), axis=0))
                action = np.argmax(q_values)
                
                next_state, reward, done, truncated, info = env.step(action)  # Updated to unpack info
                next_state = preprocess_frame(next_state)
                state_stack.append(next_state)
                
                eval_total_reward += reward
                state = next_state
                
                if done or truncated:
                    break
            
            eval_rewards.append(eval_total_reward)
        
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_rewards.append(avg_eval_reward)
        episodes.append(episode)
        print(f"Evaluation: Average Reward over {eval_episodes} episodes: {avg_eval_reward}")
    
    if episode % 100 == 0:
        primary_network.save(f"dqn_breakout_{episode}.h5")

env.close()

# Plotting the performance
plt.figure(figsize=(12, 6))
plt.plot(total_rewards, label='Total Reward per Episode')
plt.plot(episodes, avg_eval_rewards, label='Average Evaluation Reward', marker='o')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Performance of DQN on BreakoutNoFrameskip-v4')
plt.legend()
plt.show()
