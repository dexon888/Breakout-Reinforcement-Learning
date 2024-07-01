import argparse
import numpy as np
from environment import Environment
import time
from gym.wrappers.monitoring import video_recorder
from tqdm import tqdm
import torch

seed = 11037

def test(agent, env, total_episodes=30):
    """
    Test an agent in the given environment.

    Args:
        agent: An agent to test.
        env: An environment to test the agent in.
        total_episodes: The number of episodes to test the agent for.
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
            env.render()  # This will render the environment to the screen
            action = agent.act(state)  # Ensure this method matches your agent's action method
            state, reward, terminated, truncated, _ = env.step(action)
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
    test(agent, env, total_episodes=100, record_video=record_video)


if __name__ == '__main__':
    args = parse()
    run(args)
