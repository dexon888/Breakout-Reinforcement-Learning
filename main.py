import argparse
import numpy as np
from environment import Environment
import time
import torch
from test import test
seed = 11037

def parse():
    """
    Parse command line arguments and return an `argparse.Namespace` object.
    """
    parser = argparse.ArgumentParser(description="Breakout")
    parser.add_argument('--test_dqn', action='store_true', help='whether to test DQN')
    parser.add_argument('--render_mode', type=str, choices=['human', 'rgb_array'], default='human', help='render mode')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except ImportError:
        pass
    args = parser.parse_args()
    return args

def run(args):
    """
    Execute the main program.

    Args:
        render_mode: - None    --> no render is computed. (good when testing on many episodes)
                     - 'human' --> The environment is continuously rendered (human consumption)
    """
    env = Environment('BreakoutNoFrameskip-v4', args,
                      atari_wrapper=True, test=True, render_mode=args.render_mode)
    from agent_dqn import DQNAgent  # Ensure this matches your class name
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env, state_shape, action_size, device, args)
    if args.train_dqn:
        agent.train(args.episodes, args.batch_size, save_interval=args.save_interval)
    if args.test_dqn:
        if args.model_path:
            agent.load(args.model_path)
        test(agent, env, total_episodes=100)

if __name__ == '__main__':
    args = parse()
    run(args)
