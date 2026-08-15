import gymnasium as gym
import evogym.envs

from src.body import make_fixed_body

body = make_fixed_body()
env = gym.make("Walker-v0", body=body, render_mode=None)

obs, info = env.reset(seed=7)
print("観測の形:", obs.shape)
print("行動空間:", env.action_space)
print("行動の最小値:", env.action_space.low)
print("行動の最大値:", env.action_space.high)

for number in range(10):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(number, "報酬:", reward)
    obs = next_obs
    if terminated or truncated:
        break

env.close()