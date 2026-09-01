"""実験ディレクトリから方策を読み込み、決定論的WalkerのデモGIFを生成する。"""

import argparse
import json
from pathlib import Path

import evogym.envs  # noqa: F401 - EvoGym環境を登録するために読み込む
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
import torch

from src.policy import Policy
from src.reinforce import choose_action, make_mean_action
from src.wrappers import NormalizeActionSpace


TASK_NAME = "Walker-v0"
PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="生成训练策略的 EvoGym 演示 GIF。")
    parser.add_argument("--run-name", required=True, help="runs 下的实验目录名。")
    parser.add_argument(
        "--checkpoint",
        choices=["best", "latest", "initial"],
        default="best",
        help="渲染最佳、最新或训练前策略。",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help="仅 PPO：渲染 checkpoints/model_<step>_steps.zip 阶段模型。",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="最多执行的仿真步数。")
    parser.add_argument("--seed", type=int, default=1000, help="环境随机种子。")
    parser.add_argument("--frame-skip", type=int, default=2, help="每隔多少仿真步保存一帧。")
    parser.add_argument("--fps", type=int, default=30, help="GIF 每秒帧数。")
    parser.add_argument(
        "--stochastic-std",
        type=float,
        help="仅 REINFORCE：按此标准差采样；默认使用确定性均值动作。",
    )
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="仅 PPO：采样随机策略；默认使用确定性动作。",
    )
    parser.add_argument("--output", help="输出路径；默认保存在实验目录。")
    return parser.parse_args()


def load_policy(run_dir, checkpoint_name, observation_size, action_size):
    policy = Policy(observation_size, action_size)
    if checkpoint_name == "latest":
        checkpoint = torch.load(
            run_dir / "latest_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = torch.load(
            run_dir / f"{checkpoint_name}_policy.pt",
            map_location="cpu",
            weights_only=True,
        )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def load_ppo(run_dir, checkpoint_name, checkpoint_step, env):
    if checkpoint_step is None:
        model_path = run_dir / f"{checkpoint_name}_model.zip"
    else:
        model_path = run_dir / "checkpoints" / f"model_{checkpoint_step}_steps.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 PPO 模型：{model_path}")
    return PPO.load(model_path, env=env, device="cpu")


def main():
    args = parse_args()
    if args.max_steps <= 0 or args.frame_skip <= 0 or args.fps <= 0:
        raise ValueError("--max-steps、--frame-skip 和 --fps 必须大于 0。")
    if args.stochastic_std is not None and args.stochastic_std <= 0:
        raise ValueError("--stochastic-std 必须大于 0。")
    if args.checkpoint_step is not None and args.checkpoint_step <= 0:
        raise ValueError("--checkpoint-step 必须大于 0。")

    run_dir = PROJECT_DIR / "runs" / args.run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"找不到实验目录：{run_dir}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    algorithm = config["algorithm"]
    body = np.load(run_dir / "body.npy")
    stage_name = (
        f"step_{args.checkpoint_step}"
        if args.checkpoint_step is not None
        else args.checkpoint
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / f"{stage_name}_demo.gif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(TASK_NAME, body=body, render_mode="rgb_array")
    if algorithm == "PPO":
        env = NormalizeActionSpace(env)
    obs, _ = env.reset(seed=args.seed)
    if algorithm == "PPO":
        if args.stochastic_std is not None:
            raise ValueError("PPO 请使用 --stochastic-policy，不使用 --stochastic-std。")
        ppo_model = load_ppo(
            run_dir,
            args.checkpoint,
            args.checkpoint_step,
            env,
        )
        policy = None
    else:
        if args.checkpoint_step is not None:
            raise ValueError("--checkpoint-step 目前只支持 PPO 实验。")
        if args.stochastic_policy:
            raise ValueError("REINFORCE 请使用 --stochastic-std。")
        ppo_model = None
        policy = load_policy(
            run_dir,
            args.checkpoint,
            len(obs),
            env.action_space.shape[0],
        )
    torch.manual_seed(args.seed)

    frames = []
    total_return = 0.0
    steps = 0
    try:
        for step_number in range(args.max_steps):
            if step_number % args.frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if ppo_model is not None:
                action, _ = ppo_model.predict(
                    obs,
                    deterministic=not args.stochastic_policy,
                )
            elif args.stochastic_std is None:
                action = make_mean_action(
                    policy,
                    obs,
                    env.action_space.low,
                    env.action_space.high,
                )
            else:
                with torch.no_grad():
                    action, _ = choose_action(
                        policy,
                        obs,
                        env.action_space.low,
                        env.action_space.high,
                        args.stochastic_std,
                    )
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            steps += 1
            if terminated or truncated:
                break

        final_frame = env.render()
        if final_frame is not None:
            frames.append(final_frame)
    finally:
        env.close()

    if not frames:
        raise RuntimeError("环境没有返回可保存的画面。")
    imageio.mimsave(output_path, frames, fps=args.fps, loop=0)
    print(f"GIF 已保存：{output_path}")
    print(
        f"checkpoint={stage_name}, return={total_return:.6f}, "
        f"steps={steps}, mean_speed={total_return / max(steps, 1):.8f}, "
        f"algorithm={algorithm}"
    )


if __name__ == "__main__":
    main()
