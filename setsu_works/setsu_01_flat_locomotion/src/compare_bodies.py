"""同一のPPO設定で二種類のロボット形状を比較する。"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="比较两个 PPO 身体实验。")
    parser.add_argument("--original-run", required=True)
    parser.add_argument("--alternative-run", required=True)
    parser.add_argument("--output-dir", default="body_comparison")
    return parser.parse_args()


def read_evaluations(path):
    with path.open(newline="", encoding="utf-8") as file:
        return {
            int(row["timesteps"]): {
                "return": float(row["mean_return"]),
                "speed": float(row["mean_speed"]),
            }
            for row in csv.DictReader(file)
        }


def load_run(run_name):
    run_dir = PROJECT_DIR / "runs" / run_name
    return {
        "name": run_name,
        "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
        "body": np.load(run_dir / "body.npy"),
        "evaluations": read_evaluations(run_dir / "ppo_evaluation.csv"),
    }


def material_counts(body):
    counts = Counter(int(value) for value in body.flatten())
    return {str(material): counts.get(material, 0) for material in range(5)}


def main():
    args = parse_args()
    if Path(args.output_dir).name != args.output_dir:
        raise ValueError("--output-dir 不能包含路径。")

    original = load_run(args.original_run)
    alternative = load_run(args.alternative_run)
    comparable_names = (
        "algorithm",
        "task_name",
        "seed",
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
    original_settings = {
        name: original["config"][name] for name in comparable_names
    }
    alternative_settings = {
        name: alternative["config"][name] for name in comparable_names
    }
    if original_settings != alternative_settings:
        raise ValueError("两个实验的算法、种子或训练参数不同，不能作为单变量身体对照。")

    common_steps = sorted(
        set(original["evaluations"]) & set(alternative["evaluations"])
    )
    if not common_steps:
        raise ValueError("两个实验没有共同评估节点。")

    rows = []
    for step in common_steps:
        rows.append(
            {
                "timesteps": step,
                "original_return": original["evaluations"][step]["return"],
                "alternative_return": alternative["evaluations"][step]["return"],
                "original_speed": original["evaluations"][step]["speed"],
                "alternative_speed": alternative["evaluations"][step]["speed"],
            }
        )

    output_dir = PROJECT_DIR / "runs" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "body_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    final = rows[-1]
    speed_ratio = final["alternative_speed"] / final["original_speed"]
    original_label = original["config"].get("body_name", "walker")
    alternative_label = alternative["config"].get("body_name", "alternative")
    summary = {
        "original_run": original["name"],
        "alternative_run": alternative["name"],
        "controlled_settings": original_settings,
        "original_body": original["body"].tolist(),
        "alternative_body": alternative["body"].tolist(),
        "original_material_counts": material_counts(original["body"]),
        "alternative_material_counts": material_counts(alternative["body"]),
        "final_timesteps": final["timesteps"],
        "original_final_return": final["original_return"],
        "alternative_final_return": final["alternative_return"],
        "original_final_speed": final["original_speed"],
        "alternative_final_speed": final["alternative_speed"],
        "alternative_to_original_speed_ratio": speed_ratio,
    }
    (output_dir / "body_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    x = [row["timesteps"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    figure.suptitle("PPO control: robot body comparison", fontsize=14)
    axes[0].plot(x, [row["original_return"] for row in rows], marker="o", linewidth=2.3, label=original_label)
    axes[0].plot(x, [row["alternative_return"] for row in rows], marker="o", linewidth=2.3, label=alternative_label)
    axes[0].set_ylabel("Deterministic return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, [row["original_speed"] for row in rows], marker="o", linewidth=2.3, label=original_label)
    axes[1].plot(x, [row["alternative_speed"] for row in rows], marker="o", linewidth=2.3, label=alternative_label)
    axes[1].set_xlabel("Training timesteps")
    axes[1].set_ylabel("Mean displacement / step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.text(
        0.5,
        0.01,
        f"At {final['timesteps']:,} steps, {alternative_label} speed is "
        f"{speed_ratio * 100:.1f}% of the original body",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(output_dir / "body_comparison.png", dpi=180)
    plt.close(figure)

    print(f"身体对照已保存：{output_dir}")
    print(
        f"original speed={final['original_speed']:.8f}; "
        f"alternative speed={final['alternative_speed']:.8f}; ratio={speed_ratio:.3f}"
    )


if __name__ == "__main__":
    main()
