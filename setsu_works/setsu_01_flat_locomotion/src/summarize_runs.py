"""同一設定で実行した複数のPPO乱数シード実験を集計する。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SUMMARY_FIELDS = [
    "timesteps",
    "run_count",
    "mean_return",
    "std_return",
    "min_return",
    "max_return",
    "mean_speed",
    "std_speed",
]


def parse_args():
    parser = argparse.ArgumentParser(description="汇总多个 PPO 随机种子实验。")
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default="ppo_multiseed_summary",
        help="runs 下的汇总目录名。",
    )
    return parser.parse_args()


def read_evaluations(path):
    with path.open(newline="", encoding="utf-8") as file:
        return {
            int(row["timesteps"]): {
                "mean_return": float(row["mean_return"]),
                "mean_speed": float(row["mean_speed"]),
            }
            for row in csv.DictReader(file)
        }


def main():
    args = parse_args()
    if len(args.run_names) < 2:
        raise ValueError("多种子汇总至少需要两个实验。")
    if Path(args.output_dir).name != args.output_dir:
        raise ValueError("--output-dir 不能包含路径。")

    run_data = []
    reference_body = None
    reference_settings = None
    settings_names = (
        "algorithm",
        "task_name",
        "max_steps",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "vf_coef",
        "ent_coef",
        "clip_range",
        "max_grad_norm",
    )

    for run_name in args.run_names:
        run_dir = PROJECT_DIR / "runs" / run_name
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        settings = {name: config[name] for name in settings_names}
        body = np.load(run_dir / "body.npy")
        if reference_settings is None:
            reference_settings = settings
            reference_body = body
        elif settings != reference_settings or not np.array_equal(body, reference_body):
            raise ValueError(f"实验 {run_name} 的配置或身体与第一组不同。")
        baselines = json.loads((run_dir / "baselines.json").read_text(encoding="utf-8"))
        run_data.append(
            {
                "name": run_name,
                "seed": config["seed"],
                "evaluations": read_evaluations(run_dir / "ppo_evaluation.csv"),
                "random_return": baselines["random_actions"]["mean_return"],
                "random_speed": baselines["random_actions"]["mean_speed"],
            }
        )

    common_timesteps = sorted(
        set.intersection(*(set(run["evaluations"]) for run in run_data))
    )
    if not common_timesteps:
        raise ValueError("这些实验没有共同的评估时间点。")

    rows = []
    for timesteps in common_timesteps:
        returns = np.asarray(
            [run["evaluations"][timesteps]["mean_return"] for run in run_data]
        )
        speeds = np.asarray(
            [run["evaluations"][timesteps]["mean_speed"] for run in run_data]
        )
        rows.append(
            {
                "timesteps": timesteps,
                "run_count": len(run_data),
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std()),
                "min_return": float(returns.min()),
                "max_return": float(returns.max()),
                "mean_speed": float(speeds.mean()),
                "std_speed": float(speeds.std()),
            }
        )

    output_dir = PROJECT_DIR / "runs" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "multiseed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    final_step = common_timesteps[-1]
    final_returns = [
        run["evaluations"][final_step]["mean_return"] for run in run_data
    ]
    final_speeds = [
        run["evaluations"][final_step]["mean_speed"] for run in run_data
    ]
    initial_returns = [run["evaluations"][0]["mean_return"] for run in run_data]
    random_returns = [run["random_return"] for run in run_data]
    summary = {
        "algorithm": "PPO",
        "run_names": [run["name"] for run in run_data],
        "seeds": [run["seed"] for run in run_data],
        "common_final_timesteps": final_step,
        "initial_return_mean": float(np.mean(initial_returns)),
        "random_action_return_mean": float(np.mean(random_returns)),
        "final_return_mean": float(np.mean(final_returns)),
        "final_return_std": float(np.std(final_returns)),
        "final_return_min": float(np.min(final_returns)),
        "final_return_max": float(np.max(final_returns)),
        "final_speed_mean": float(np.mean(final_speeds)),
        "final_speed_std": float(np.std(final_speeds)),
        "per_run_final_return": {
            run["name"]: run["evaluations"][final_step]["mean_return"]
            for run in run_data
        },
        "shared_settings": reference_settings,
    }
    (output_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    x = np.asarray(common_timesteps)
    mean_returns = np.asarray([row["mean_return"] for row in rows])
    std_returns = np.asarray([row["std_return"] for row in rows])
    mean_speeds = np.asarray([row["mean_speed"] for row in rows])
    std_speeds = np.asarray([row["std_speed"] for row in rows])

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    figure.suptitle("EvoGym Walker PPO - 3-seed reproducibility", fontsize=14)
    for run in run_data:
        axes[0].plot(
            x,
            [run["evaluations"][step]["mean_return"] for step in common_timesteps],
            color="gray",
            alpha=0.35,
            linewidth=1,
        )
        axes[1].plot(
            x,
            [run["evaluations"][step]["mean_speed"] for step in common_timesteps],
            color="gray",
            alpha=0.35,
            linewidth=1,
        )
    axes[0].plot(x, mean_returns, marker="o", color="#d1495b", linewidth=2.5, label="Mean")
    axes[0].fill_between(x, mean_returns - std_returns, mean_returns + std_returns, color="#d1495b", alpha=0.2, label="± 1 std")
    axes[0].axhline(np.mean(random_returns), color="#7b2cbf", linestyle=":", label="Random-action baseline")
    axes[0].set_ylabel("Deterministic return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, mean_speeds, marker="o", color="#2a9d8f", linewidth=2.5, label="Mean")
    axes[1].fill_between(x, mean_speeds - std_speeds, mean_speeds + std_speeds, color="#2a9d8f", alpha=0.2, label="± 1 std")
    axes[1].axhline(np.mean([run["random_speed"] for run in run_data]), color="#7b2cbf", linestyle=":", label="Random-action baseline")
    axes[1].set_xlabel("Training timesteps")
    axes[1].set_ylabel("Mean displacement / step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.text(
        0.5,
        0.01,
        f"Final return: {summary['final_return_mean']:.4f} ± "
        f"{summary['final_return_std']:.4f} across {len(run_data)} seeds",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(output_dir / "ppo_multiseed_curves.png", dpi=180)
    plt.close(figure)

    print(f"多种子汇总已保存：{output_dir}")
    print(
        f"final return={summary['final_return_mean']:.6f} ± "
        f"{summary['final_return_std']:.6f}; "
        f"speed={summary['final_speed_mean']:.8f} ± "
        f"{summary['final_speed_std']:.8f}"
    )


if __name__ == "__main__":
    main()
