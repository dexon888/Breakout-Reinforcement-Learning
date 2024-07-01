# environment.py
import gymnasium as gym
from atari_wrapper import wrap_deepmind  # Import your wrapper if applicable

class Environment:
    def __init__(self, env_name, args, atari_wrapper=True, test=False, render_mode=None):
        if render_mode:
            self.env = gym.make(env_name, render_mode=render_mode)
        else:
            self.env = gym.make(env_name)
        self.render_mode = render_mode
        if atari_wrapper:
            self.env = wrap_deepmind(self.env, clip_rewards=test)
    
    def seed(self, seed):
        self.env.unwrapped.seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        if self.render_mode == 'human':
            self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
