"""エピソード別ログを一定ステップごとに集約し、決定論的評価点を重ねる。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="绘制两个 PPO 实验的高分辨率趋势图。")
    parser.add_argument("--original-run", required=True)
    parser.add_argument("--alternative-run", required=True)
    parser.add_argument("--original-label", default="original_walker")
    parser.add_argument("--alternative-label", default="layered_walker")
    parser.add_argument("--bin-steps", type=int, default=5_000)
    parser.add_argument("--max-timesteps", type=int, default=300_000)
    parser.add_argument("--output-dir", default="body_trends_5k_seed7")
    return parser.parse_args()


def read_monitor(path):
    with path.open(newline="", encoding="utf-8") as file:
        first_line = file.readline()
        if not first_line.startswith("#"):
            file.seek(0)
        return [
            {"return": float(row["r"]), "length": int(float(row["l"]))}
            for row in csv.DictReader(file)
        ]


def read_evaluations(path, max_timesteps):
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {
                "timesteps": int(row["timesteps"]),
                "return": float(row["mean_return"]),
                "speed": float(row["mean_speed"]),
            }
            for row in csv.DictReader(file)
            if int(row["timesteps"]) <= max_timesteps
        ]


def aggregate_training(rows, bin_steps, max_timesteps):
    bins = {}
    cumulative_steps = 0
    for row in rows:
        cumulative_steps += row["length"]
        if cumulative_steps > max_timesteps:
            break
        boundary = ((cumulative_steps - 1) // bin_steps + 1) * bin_steps
        bucket = bins.setdefault(boundary, {"returns": [], "speeds": []})
        bucket["returns"].append(row["return"])
        bucket["speeds"].append(row["return"] / row["length"])

    return [
        {
            "timesteps": boundary,
            "mean_training_return": sum(values["returns"]) / len(values["returns"]),
            "mean_training_speed": sum(values["speeds"]) / len(values["speeds"]),
            "episodes": len(values["returns"]),
        }
        for boundary, values in sorted(bins.items())
    ]


def load_run(run_name, bin_steps, max_timesteps):
    run_dir = PROJECT_DIR / "runs" / run_name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return {
        "name": run_name,
        "label": config.get("body_name", run_name),
        "training": aggregate_training(
            read_monitor(run_dir / "training.monitor.csv"),
            bin_steps,
            max_timesteps,
        ),
        "evaluations": read_evaluations(
            run_dir / "ppo_evaluation.csv",
            max_timesteps,
        ),
    }


def write_training_csv(path, original, alternative):
    original_by_step = {row["timesteps"]: row for row in original["training"]}
    alternative_by_step = {
        row["timesteps"]: row for row in alternative["training"]
    }
    common_steps = sorted(set(original_by_step) & set(alternative_by_step))
    fieldnames = [
        "timesteps",
        "original_mean_training_return",
        "alternative_mean_training_return",
        "original_mean_training_speed",
        "alternative_mean_training_speed",
        "original_episodes",
        "alternative_episodes",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for step in common_steps:
            first = original_by_step[step]
            second = alternative_by_step[step]
            writer.writerow(
                {
                    "timesteps": step,
                    "original_mean_training_return": first["mean_training_return"],
                    "alternative_mean_training_return": second["mean_training_return"],
                    "original_mean_training_speed": first["mean_training_speed"],
                    "alternative_mean_training_speed": second["mean_training_speed"],
                    "original_episodes": first["episodes"],
                    "alternative_episodes": second["episodes"],
                }
            )


def write_evaluation_csv(path, original, alternative):
    original_by_step = {row["timesteps"]: row for row in original["evaluations"]}
    alternative_by_step = {
        row["timesteps"]: row for row in alternative["evaluations"]
    }
    common_steps = sorted(set(original_by_step) & set(alternative_by_step))
    fieldnames = [
        "timesteps",
        "original_deterministic_return",
        "alternative_deterministic_return",
        "original_deterministic_speed",
        "alternative_deterministic_speed",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for step in common_steps:
            first = original_by_step[step]
            second = alternative_by_step[step]
            writer.writerow(
                {
                    "timesteps": step,
                    "original_deterministic_return": first["return"],
                    "alternative_deterministic_return": second["return"],
                    "original_deterministic_speed": first["speed"],
                    "alternative_deterministic_speed": second["speed"],
                }
            )


def series(rows, key):
    return [row["timesteps"] for row in rows], [row[key] for row in rows]


def moving_average(values, window):
    output = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        current = values[start : index + 1]
        output.append(sum(current) / len(current))
    return output


def plot_main_training_trend(output_dir, original, alternative, bin_steps):
    figure, axis = plt.subplots(figsize=(11, 6.5))
    colors = ["#1f77b4", "#ff7f0e"]
    for run, color in zip((original, alternative), colors):
        x, returns = series(run["training"], "mean_training_return")
        axis.plot(x, returns, color=color, linewidth=1.2, alpha=0.32)
        axis.plot(
            x,
            moving_average(returns, 3),
            color=color,
            linewidth=3,
            label=f"{run['label']} (15k moving average)",
        )

    axis.set_title("PPO training trend: two robot bodies", fontsize=15)
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean episode return")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.text(
        0.5,
        0.015,
        f"Thin lines: observed {bin_steps // 1000}k-step means. "
        "Thick lines: trailing 15k moving averages. Higher return means more forward displacement.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    figure.savefig(output_dir / "main_training_trend_5k.png", dpi=180)
    plt.close(figure)


def plot_deterministic_diagnostic(output_dir, original, alternative):
    figure, axis = plt.subplots(figsize=(11, 6.5))
    colors = ["#1f77b4", "#ff7f0e"]
    plotted = {}
    for run, color in zip((original, alternative), colors):
        # 25k刻みの点だけを残し、rollout調整で生じる100352などの終端点は除外する。
        regular = [
            row
            for row in run["evaluations"]
            if row["timesteps"] == 0 or row["timesteps"] % 25_000 == 0
        ]
        x, returns = series(regular, "return")
        plotted[run["label"]] = dict(zip(x, returns))
        axis.plot(
            x,
            returns,
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.4,
            label=run["label"],
        )

    axis.axvspan(170_000, 180_000, color="#d62728", alpha=0.10)
    original_points = plotted[original["label"]]
    if 175_000 in original_points:
        axis.annotate(
            f"Deterministic collapse\nreturn={original_points[175_000]:.3f}",
            xy=(175_000, original_points[175_000]),
            xytext=(135_000, 1.0),
            arrowprops={"arrowstyle": "->", "color": "#b22222"},
            color="#b22222",
            fontsize=10,
        )
    if 200_000 in original_points:
        axis.annotate(
            f"Recovered\nreturn={original_points[200_000]:.3f}",
            xy=(200_000, original_points[200_000]),
            xytext=(205_000, 2.0),
            arrowprops={"arrowstyle": "->", "color": "#2e7d32"},
            color="#2e7d32",
            fontsize=10,
        )

    axis.set_title("Deterministic checkpoint diagnostic", fontsize=15)
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Deterministic return")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.text(
        0.5,
        0.015,
        "Saved 25k checkpoints. This figure explains policy instability; it is not the main dense trend chart.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    figure.savefig(output_dir / "deterministic_checkpoint_diagnostic.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.bin_steps <= 0 or args.max_timesteps <= 0:
        raise ValueError("--bin-steps 和 --max-timesteps 必须大于 0。")
    if Path(args.output_dir).name != args.output_dir:
        raise ValueError("--output-dir 不能包含路径。")

    original = load_run(args.original_run, args.bin_steps, args.max_timesteps)
    alternative = load_run(
        args.alternative_run,
        args.bin_steps,
        args.max_timesteps,
    )
    original["label"] = args.original_label
    alternative["label"] = args.alternative_label
    output_dir = PROJECT_DIR / "runs" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_training_csv(output_dir / "training_trends_5k.csv", original, alternative)
    write_evaluation_csv(
        output_dir / "deterministic_evaluations.csv",
        original,
        alternative,
    )
    plot_main_training_trend(
        output_dir,
        original,
        alternative,
        args.bin_steps,
    )
    plot_deterministic_diagnostic(output_dir, original, alternative)

    colors = ["#1f77b4", "#ff7f0e"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    figure.suptitle(
        "PPO body comparison: dense training trend and deterministic evaluation",
        fontsize=15,
    )
    for run, color in zip((original, alternative), colors):
        x, y = series(run["training"], "mean_training_return")
        axes[0, 0].plot(x, y, color=color, linewidth=2, label=run["label"])
        x, y = series(run["training"], "mean_training_speed")
        axes[1, 0].plot(x, y, color=color, linewidth=2, label=run["label"])
        x, y = series(run["evaluations"], "return")
        axes[0, 1].plot(x, y, color=color, marker="o", linewidth=2, label=run["label"])
        x, y = series(run["evaluations"], "speed")
        axes[1, 1].plot(x, y, color=color, marker="o", linewidth=2, label=run["label"])

    axes[0, 0].set_title(f"Stochastic training return ({args.bin_steps:,}-step bins)")
    axes[1, 0].set_title(f"Stochastic training speed ({args.bin_steps:,}-step bins)")
    axes[0, 1].set_title("Deterministic evaluation return (saved points)")
    axes[1, 1].set_title("Deterministic evaluation speed (saved points)")
    axes[0, 0].set_ylabel("Mean episode return")
    axes[1, 0].set_ylabel("Mean displacement / step")
    axes[0, 1].set_ylabel("Return")
    axes[1, 1].set_ylabel("Mean displacement / step")
    for axis in axes.flat:
        axis.axvline(175_000, color="#888888", linestyle="--", linewidth=1, alpha=0.7)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[1, 0].set_xlabel("Training timesteps")
    axes[1, 1].set_xlabel("Training timesteps")
    figure.text(
        0.5,
        0.01,
        "Left: 5k means from sampled training episodes. Right: deterministic evaluations; "
        "historical points were saved every 25k. Dashed line marks 175k.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output_dir / "body_trends_5k.png", dpi=180)
    plt.close(figure)
    print(f"高分辨率趋势已保存：{output_dir}")


if __name__ == "__main__":
    main()
