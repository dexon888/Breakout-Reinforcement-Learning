import tensorflow as tf
from tensorflow import keras

def build_model(input_shape, action_space):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(action_space, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00025), loss='mse')
    return model
