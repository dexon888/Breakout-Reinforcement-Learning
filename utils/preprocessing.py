import numpy as np

# Preprocess the frame
def preprocess_frame(frame):
    frame = frame[34:194]  # Crop the image to focus on the playing area
    frame = frame.mean(2)  # Convert to grayscale by averaging the color channels
    frame = frame[::2, ::2]  # Downsample by a factor of 2
    return frame.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]