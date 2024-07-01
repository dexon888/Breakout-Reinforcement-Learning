import argparse
from test import test
from environment import Environment
import time
import torch


def parse():
    """
    Parse command line arguments and return an `argparse.Namespace` object.
    """
    parser = argparse.ArgumentParser(description="Breakout")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true',
                        help='whether to train DQN')
    parser.add_argument('--test_dqn', action='store_true',
                        help='whether to test DQN')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='number of episodes to train')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except ImportError:
        pass
    args = parser.parse_args()
    return args


def run(args):
    """
    Execute the main program that can either train or test an agent.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True, test=False)
        from agent_dqn import DQNAgent
        state_shape = env.observation_space.shape
        action_size = env.action_space.n
        agent = DQNAgent(env, state_shape, action_size, device)
        agent.train(args.episodes, args.batch_size)

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args,
                          atari_wrapper=True, test=True)
        from agent_dqn import DQNAgent
        state_shape = env.observation_space.shape
        action_size = env.action_space.n
        agent = DQNAgent(env, state_shape, action_size, device)
        test(agent, env, total_episodes=100, record_video=False)

    print('running time:', time.time() - start_time)


if __name__ == '__main__':
    args = parse()
    run(args)
