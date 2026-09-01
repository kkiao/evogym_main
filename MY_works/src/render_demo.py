from src.reinforce import make_mean_action
import imageio.v2 as imageio
import gymnasium as gym
import evogym.envs
import numpy as np
import torch

from src.policy import Policy


TASK_NAME = "Walker-v0"#climer->walker
BODY_FILE = "results/parallel_fixed_body.npy"
POLICY_FILE = "results/parallel_fixed_policy.pt"
OUTPUT_FILE = "results/parallel_fixed.gif"
MAX_STEPS = 300



def make_gif():
    """指定した体とPolicyをClimberで動かし、フレームをGIFとして保存する。"""
    body = np.load(BODY_FILE)
    env = gym.make(TASK_NAME, body=body, render_mode="rgb_array")
    obs, _ = env.reset(seed=100)

    policy = Policy(len(obs), env.action_space.shape[0])
    if POLICY_FILE is not None:
        policy.load_state_dict(torch.load(POLICY_FILE, map_location="cpu"))

    frames = []
    for step_number in range(MAX_STEPS):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = make_mean_action(
            policy,
            obs,
            env.action_space.low,
            env.action_space.high,
        )
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs = next_obs

        #if step_number < 10:
            #print("step:", step_number, "action:", action)

        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(OUTPUT_FILE, frames, fps=30)
    print("保存しました:", OUTPUT_FILE)


if __name__ == "__main__":
    make_gif()