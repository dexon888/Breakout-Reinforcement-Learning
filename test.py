import argparse
import numpy as np
from environment import Environment
import time
from tqdm import tqdm
import torch

seed = 11037

def test(agent, env, total_episodes=30, record_video=False):
    """
    Test an agent in the given environment.

    Args:
        agent: An agent to test.
        env: An environment to test the agent in.
        total_episodes: The number of episodes to test the agent for.
        record_video: (bool) whether you need to record video
    """
    rewards = []
    env.seed(seed)
    start_time = time.time()
    for i in tqdm(range(total_episodes), desc="Testing"):
        state, _ = env.reset()
        if hasattr(agent, 'init_game_setting'):
            agent.init_game_setting()
        episode_reward = 0.0

        terminated, truncated = False, False
        while not terminated and not truncated:
            if record_video:
                env.render()  # This will render the environment to the screen
            action = agent.act(state)  # Ensure this method matches your agent's action method
            step_result = env.step(action)
            
            # Handle both 4 and 5 return values from step_result
            if len(step_result) == 5:
                state, reward, terminated, truncated, _ = step_result
            else:
                state, reward, terminated, _ = step_result
                truncated = False

            episode_reward += reward
            if terminated or truncated:
                if truncated:
                    print("Truncated: ", truncated)
                print(f"Episode {i + 1} reward: {episode_reward}")
                break

        rewards.append(episode_reward)

    print('Run %d episodes' % (total_episodes))
    print('Mean:', np.mean(rewards))
    print('rewards', rewards)
    print('running time', time.time() - start_time)

    env.close()

def parse():
    """
    Parse command line arguments and return an `argparse.Namespace` object.
    """
    parser = argparse.ArgumentParser(description="Breakout")
    parser.add_argument('--test_dqn', action='store_true', help='whether to test DQN')
    parser.add_argument('--render_mode', type=str, choices=['human', 'rgb_array'], default='human', help='render mode')
    parser.add_argument('--model_path', type=str, help='path to the trained model')
    parser.add_argument('--episodes', type=int, default=30, help='number of episodes to test')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except ImportError:
        pass
    args = parser.parse_args()
    return args

def run(args, render_mode=None, record_video=False):
    """
    Execute the main program.

    Args:
        render_mode: - None    --> no render is computed. (good when testing on many episodes)
                     - 'human' --> The environment is continuously rendered (human consumption)
        record_video: (bool) whether you need to record video
    """
    env = Environment('BreakoutNoFrameskip-v4', args,
                      atari_wrapper=True, test=True, render_mode=render_mode)
    from agent_dqn import DQNAgent  # Ensure this matches your class name
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env, state_shape, action_size, device, args)
    if args.model_path:
        agent.load(args.model_path)
    test(agent, env, total_episodes=args.episodes, record_video=record_video)


if __name__ == '__main__':
    args = parse()
    run(args)
