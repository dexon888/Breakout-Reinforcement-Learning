import gymnasium as gym
import cv2
import keyboard

# Define key mappings to actions
key_mapping = {
    'a': 3,  # Move paddle left
    'd': 2,  # Move paddle right
    's': 0,  # No action
    'w': 1   # Fire (optional in Breakout)
}

def play_atari(env_name):
    env = gym.make(env_name, render_mode='human')
    state, info = env.reset()
    done = False

    while not done:
        if keyboard.is_pressed('q'):
            break
        action = 0  # Default to no action
        for key in key_mapping:
            if keyboard.is_pressed(key):
                action = key_mapping[key]
                break

        state, reward, done, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            cv2.imshow("Breakout", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_atari('BreakoutNoFrameskip-v4')
