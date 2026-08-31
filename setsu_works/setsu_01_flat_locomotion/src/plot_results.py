"""新しい実験ディレクトリの学習・評価CSVから提出用の曲線を作成する。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 EvoGym 强化学习实验曲线。")
    parser.add_argument("--run-name", required=True, help="runs 下的实验目录名。")
    parser.add_argument("--window", type=int, default=20, help="训练奖励移动平均窗口。")
    parser.add_argument("--output", help="输出 PNG 路径；默认保存在实验目录。")
    return parser.parse_args()


def read_numeric_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def read_monitor_csv(path):
    """先頭行がJSONコメントであるStable-Baselines3 Monitor形式を読み込む。"""
    with path.open(newline="", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line.startswith("#"):
            file.seek(0)
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def moving_average(values, window):
    averages = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        averages.append(running_sum / min(index + 1, window))
    return averages


def main():
    args = parse_args()
    if args.window <= 0:
        raise ValueError("--window 必须大于 0。")

    run_dir = PROJECT_DIR / "runs" / args.run_name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    algorithm = config["algorithm"]
    if algorithm == "PPO":
        training_rows = read_monitor_csv(run_dir / "training.monitor.csv")
        evaluation_rows = read_numeric_csv(run_dir / "ppo_evaluation.csv")
        training_x = []
        cumulative_steps = 0.0
        for row in training_rows:
            cumulative_steps += row["l"]
            training_x.append(cumulative_steps)
        returns = [row["r"] for row in training_rows]
        eval_x = [row["timesteps"] for row in evaluation_rows]
        x_label = "Training timesteps"
    else:
        training_rows = read_numeric_csv(run_dir / "training.csv")
        evaluation_rows = read_numeric_csv(run_dir / "evaluation.csv")
        training_x = [row["episode"] for row in training_rows]
        returns = [row["episode_return"] for row in training_rows]
        eval_x = [row["completed_episode"] for row in evaluation_rows]
        x_label = "Completed training episodes"
    if not training_rows:
        raise ValueError("训练日志还没有数据。")
    if not evaluation_rows:
        raise ValueError("评估日志还没有数据。")

    average_returns = moving_average(returns, args.window)

    eval_returns = [row["mean_return"] for row in evaluation_rows]
    eval_stds = [row["std_return"] for row in evaluation_rows]
    eval_speeds = [row["mean_speed"] for row in evaluation_rows]

    baselines_path = run_dir / "baselines.json"
    random_return = None
    random_speed = None
    if baselines_path.exists():
        baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
        random_return = baselines["random_actions"]["mean_return"]
        random_speed = baselines["random_actions"]["mean_speed"]

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)
    figure.suptitle(f"EvoGym Walker {algorithm} - {args.run_name}", fontsize=14)

    axes[0].plot(training_x, returns, color="lightgray", linewidth=1, label="Episode return")
    axes[0].plot(
        training_x,
        average_returns,
        color="#1769aa",
        linewidth=2.2,
        label=f"Moving average ({args.window})",
    )
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Training return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    lower = [mean - std for mean, std in zip(eval_returns, eval_stds)]
    upper = [mean + std for mean, std in zip(eval_returns, eval_stds)]
    axes[1].plot(eval_x, eval_returns, marker="o", color="#d1495b")
    axes[1].fill_between(eval_x, lower, upper, color="#d1495b", alpha=0.18)
    axes[1].axhline(eval_returns[0], color="black", linestyle="--", linewidth=1, label="Untrained baseline")
    if random_return is not None:
        axes[1].axhline(random_return, color="#7b2cbf", linestyle=":", linewidth=1.5, label="Random-action baseline")
    axes[1].set_ylabel("Deterministic eval return")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(eval_x, eval_speeds, marker="o", color="#2a9d8f")
    axes[2].axhline(eval_speeds[0], color="black", linestyle="--", linewidth=1, label="Untrained baseline")
    if random_speed is not None:
        axes[2].axhline(random_speed, color="#7b2cbf", linestyle=":", linewidth=1.5, label="Random-action baseline")
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Mean displacement / step")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    initial_eval = eval_returns[0]
    best_eval = max(eval_returns)
    figure.text(
        0.5,
        0.01,
        f"Initial deterministic return: {initial_eval:.6f}   "
        f"Best recorded return: {best_eval:.6f}   "
        f"Absolute change: {best_eval - initial_eval:+.6f}",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.98))

    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "learning_curves.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"曲线已保存：{output_path}")


if __name__ == "__main__":
    main()
