import os
import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
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
        obs, _, done, truncated, _ = self.env.step(1)  # FIRE action to start the game
        if done or truncated:
            self.env.reset(**kwargs)
        obs, _, done, truncated, _ = self.env.step(2)  # Another FIRE action to ensure the game starts
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
    env = gym.make(env_id, render_mode='human')  # Set render mode here
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_deepmind(env, clip_rewards=True, frame_stack=False, scale=False):
    env = FireResetEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env

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

# Load the trained model with custom objects
model_dir = 'trained_models'  # Directory where models are saved
model_filename = 'dqn_breakout_0.h5'  # Name of the model file
model_path = os.path.join(model_dir, model_filename)  # Full path to the model

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
trained_model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)

# Initialize environment
env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env)

# Play the game using the trained model
num_episodes = 5  # Number of episodes to play
for episode in range(num_episodes):
    state = preprocess_frame(env.reset())
    state_stack = deque([state] * 4, maxlen=4)
    total_reward = 0
    
    while True:
        env.render()  # Render the environment to see the agent in action
        q_values = trained_model.predict(np.expand_dims(stack_frames(state_stack), axis=0))
        action = np.argmax(q_values)
        
        # Log actions and rewards
        print(f"Action taken: {action}")
        
        next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_frame(next_state)
        state_stack.append(next_state)
        total_reward += reward
        
        # Log rewards and state
        print(f"Reward received: {reward}, Done: {done}, Truncated: {truncated}")
        
        if done or truncated:
            break
    
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
