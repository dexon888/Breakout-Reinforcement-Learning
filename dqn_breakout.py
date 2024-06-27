import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, done, truncated, _ = self.env.step(1)  # FIRE action to start the game
        if done or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, done, truncated, _ = self.env.step(2)  # Another FIRE action to ensure the game starts
        if done or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

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
        obs, info = self.env.reset(**kwargs)
        noops = self.override_num_noops if self.override_num_noops is not None else np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            obs, _, done, truncated, _ = self.env.step(self.noop_action)
            if done or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

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

# Create directory to save models
model_save_dir = 'trained_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

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
max_episode_steps = 100000  
eval_interval = 1000  
eval_episodes = 10
checkpoint_interval = 100 

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
state, info = env.reset()
state = preprocess_frame(state)
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

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Total Reward per Episode')
line2, = ax.plot([], [], label='Average Evaluation Reward', marker='o')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Performance of DQN on BreakoutNoFrameskip-v4')
plt.legend()

for episode in range(num_episodes):
    print(f"Starting episode {episode}")
    state, info = env.reset()
    state = preprocess_frame(state)
    state_stack = deque([state] * 4, maxlen=4)
    total_reward = 0

    progress_bar = tqdm(total=max_episode_steps, desc=f"Episode {episode}", unit="step")
    for t in range(max_episode_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = primary_network.predict(np.expand_dims(stack_frames(state_stack), axis=0), verbose=0)
            action = np.argmax(q_values)

        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_frame(next_state)
        state_stack.append(next_state)
        memory.append((stack_frames(list(state_stack)), action, reward, stack_frames(list(state_stack)), done))

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = np.array(states)
            next_states = np.array(next_states)
            q_values_next = target_network.predict(next_states, verbose=0)
            targets = rewards + gamma * np.max(q_values_next, axis=1) * (1 - np.array(dones))

            q_values = primary_network.predict(states, verbose=0)
            for i, action in enumerate(actions):
                q_values[i][action] = targets[i]

            primary_network.fit(states, q_values, epochs=1, verbose=0)

        if t % update_target_freq == 0:
            target_network.set_weights(primary_network.get_weights())
            print(f"Updated target network at step {t}")

        state = next_state
        total_reward += reward

        if done or truncated:
            tqdm.write(f"Episode: {episode}, Step: {t}, Finished with Total Reward: {total_reward}")
            break

        if t % 10 == 0:
            tqdm.write(f"Episode: {episode}, Step: {t}, Total Reward: {total_reward}, Epsilon: {epsilon}")

        progress_bar.update(1)

    progress_bar.close()
    epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)
    total_rewards.append(total_reward)
    print(f"Episode {episode} finished with total reward: {total_reward}")


    if episode % checkpoint_interval == 0:
        model_path = os.path.join(model_save_dir, f"dqn_breakout_{episode}.keras")
        tqdm.write(f"Saving model checkpoint at episode {episode} to {model_path}")
        primary_network.save(model_path)
        tqdm.write(f"Model checkpoint at episode {episode} saved successfully")

    # Update the plot
    line1.set_xdata(np.arange(len(total_rewards)))
    line1.set_ydata(total_rewards)
    line2.set_xdata(episodes)
    line2.set_ydata(avg_eval_rewards)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

print("Finished all episodes")
env.close()

# Final plot update
plt.ioff()
plt.show()
