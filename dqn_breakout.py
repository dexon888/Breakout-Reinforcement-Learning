import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from collections import deque

# Define the Breakout environment
env = gym.make('ALE/Breakout-v5')

# Print the action space and observation space to verify
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Preprocess the frame
def preprocess_frame(frame):
    frame = frame[34:194]  # Crop the image
    frame = frame.mean(2)  # Convert to grayscale
    frame = frame[::2, ::2]  # Downsample by factor of 2
    return frame.astype(np.float32) / 255.0  # Normalize pixel values

# Test preprocessing
import matplotlib.pyplot as plt

state = env.reset()
preprocessed_state = preprocess_frame(state)
plt.imshow(preprocessed_state, cmap='gray')
plt.title('Preprocessed Frame')
plt.show()

