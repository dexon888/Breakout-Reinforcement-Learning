import gymnasium as gym
import numpy as np
from collections import deque
from atari_wrapper import wrap_deepmind
import cv2

class Environment:
    def __init__(self, env_name, args, atari_wrapper=True, test=False, render_mode=None):
        if render_mode:
            self.env = gym.make(env_name, render_mode=render_mode)
        else:
            self.env = gym.make(env_name)
        self.render_mode = render_mode
        self.test = test
        if atari_wrapper:
            self.env = wrap_deepmind(self.env, clip_rewards=test)
        
        self.frames = deque(maxlen=4)

    def seed(self, seed):
        self.env.unwrapped.seed(seed)

    def reset(self):
        state, _ = self.env.reset()
        state = self.preprocess(state)
        for _ in range(4):
            self.frames.append(state)
        stacked_frames = np.stack(self.frames, axis=0)
        stacked_frames = np.transpose(stacked_frames, (1, 2, 0))  # Stack along channels
        return stacked_frames, None

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            next_state, reward, done, truncated, info = result
            done = done or truncated
        else:
            next_state, reward, done, info = result

        next_state = self.preprocess(next_state)
        self.frames.append(next_state)
        stacked_frames = np.stack(self.frames, axis=0)
        stacked_frames = np.transpose(stacked_frames, (1, 2, 0))  # Stack along channels
        return stacked_frames, reward, done, info

    def preprocess(self, frame):
        frame = np.array(frame)  # Ensure frame is a numpy array
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        frame = frame.astype(np.float32) / 255.0
        return frame

    def render(self):
        if self.render_mode == 'human':
            self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32)

    @property
    def action_space(self):
        return self.env.action_space
