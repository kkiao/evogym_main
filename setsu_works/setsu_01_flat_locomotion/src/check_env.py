import gymnasium as gym
import evogym.envs

from src.body import make_walker_body


def main():
    body = make_walker_body()
    env = gym.make("Walker-v0", body=body, render_mode=None)
    obs, _ = env.reset(seed=7)
    env.action_space.seed(7)
    print("観測の形:", obs.shape)
    print("行動空間:", env.action_space)
    print("行動の最小値:", env.action_space.low)
    print("行動の最大値:", env.action_space.high)

    try:
        for number in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            print(number, "報酬:", reward)
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
