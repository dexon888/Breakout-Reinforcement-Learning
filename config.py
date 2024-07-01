HYPERPARAMETERS = {
    "batch_size": 32,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.99995,  # Slower decay rate
    "learning_rate": 0.00025,
    "replay_buffer_size": 100000,
    "target_update_freq": 1000,
    "num_episodes": 1000,
    "input_shape": (84, 84, 4),
    "evaluation_freq": 50,  # Evaluate every 50 episodes
    "checkpoint_freq": 5,  # Save model every 50 episodes
    "eval_episodes": 10     # Number of episodes to run for evaluation
}

ENV_NAME = "BreakoutNoFrameskip-v4"
