import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from collections import deque

# Define the Breakout environment
env = gym.make('Breakout-v4')

# Print the action space and observation space to verify
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")


