"""高さ一パイロットの図表、失敗回放、検収明細を生成する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import build_course


def read_evaluations(path: Path) -> list[dict[str, object]]:
    """五千歩評価CSVを数値化して返す。"""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "step": int(row["step"]),
                    "group": row["group"],
                    "success_rate": float(row["success_rate"]),
                    "raw_clearance_rate": float(row["raw_clearance_rate"]),
                    "recovery_rate": float(row["recovery_rate"]),
                    "hard_fall_rate": float(row["hard_fall_rate"]),
                    "mean_max_x": float(row["mean_max_x"]),
                    "mean_steps": float(row["mean_steps"]),
                }
            )
    return rows


def rows_for_group(
    rows: list[dict[str, object]],
    group: str,
) -> list[dict[str, object]]:
    """指定評価群を歩数順に抽出する。"""
    return sorted(
        (
            row
            for row in rows
            if row["group"] == group
            and (
                int(row["step"]) == 0
                or int(row["step"]) % 5_000 == 0
            )
        ),
        key=lambda row: int(row["step"]),
    )


def make_rate_chart(rows: list[dict[str, object]], output: Path) -> None:
    """成功、安全、完全通過を同一尺度で描く。"""
    seen = rows_for_group(rows, "seen_height1")
    steps = [int(row["step"]) / 1000 for row in seen]
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    for key, label, marker in (
        ("success_rate", "Course success", "o"),
        ("raw_clearance_rate", "Full-body clearance", "s"),
        ("recovery_rate", "Post-obstacle recovery", "^"),
        ("hard_fall_rate", "Hard fall", "x"),
    ):
        axis.plot(
            steps,
            [100.0 * float(row[key]) for row in seen],
            marker=marker,
            linewidth=2,
            label=label,
        )
    axis.set_title("Height-1 Pilot: Capability and Safety Every 5k Steps")
    axis.set_xlabel("Training steps (thousand)")
    axis.set_ylabel("Evaluation rate (%)")
    axis.set_ylim(-2, 102)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    axis.text(
        0.5,
        0.46,
        "No full-body clearance was observed",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="#a6332a",
        bbox={"facecolor": "white", "edgecolor": "#a6332a", "alpha": 0.9},
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def make_progress_chart(rows: list[dict[str, object]], output: Path) -> None:
    """絶対前進記録と停止終了までの時間を別軸で描く。"""
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for group, label, color in (
        ("seen_height1", "Seen height-1 shapes", "#2468a2"),
        ("unseen_width", "Unseen platform width", "#d07620"),
    ):
        selected = rows_for_group(rows, group)
        steps = [int(row["step"]) / 1000 for row in selected]
        axes[0].plot(
            steps,
            [float(row["mean_max_x"]) for row in selected],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        axes[1].plot(
            steps,
            [float(row["mean_steps"]) for row in selected],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axes[0].set_title("Height-1 Pilot: Progress Diagnostics Every 5k Steps")
    axes[0].set_ylabel("Mean maximum COM x (m)")
    axes[1].set_ylabel("Mean episode length (steps)")
    axes[1].set_xlabel("Training steps (thousand)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def render_replay(
    model_path: Path,
    output: Path,
    *,
    template_name: str,
    frame_skip: int,
) -> dict[str, object]:
    """固定高さ一コースを決定論的に再生してGIFへ保存する。"""
    model = PPO.load(model_path, device="cpu")
    course = build_course(
        [template_name],
        split="pilot_replay",
        seed=52_000,
        difficulty=1,
        start_runway_voxels=20,
    )
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array",
    )
    frames: list[np.ndarray] = []
    try:
        observation, info = environment.reset(seed=52_000)
        first = environment.render()
        if first is not None:
            frames.append(np.asarray(first))
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            steps += 1
            if steps % frame_skip == 0:
                frame = environment.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
        final_frame = environment.render()
        if final_frame is not None:
            frames.append(np.asarray(final_frame))
    finally:
        environment.close()
    if not frames:
        raise RuntimeError("GIF回放用フレームを取得できなかった。")
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=12, loop=0)
    return {
        "model": str(model_path.resolve()),
        "template": template_name,
        "steps": steps,
        "frames": len(frames),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "course_complete": bool(info["course_complete"]),
        "hard_fall": bool(info["hard_fall"]),
        "stall_limit_reached": bool(info["stall_limit_reached"]),
        "maximum_com_x": float(info["max_x_position"]),
        "failure_reason": str(info["failure_reason"]),
    }


def main() -> None:
    """指定パイロットフォルダへ提出可能な診断成果をまとめる。"""
    parser = argparse.ArgumentParser(description="生成高度1标定的图表与回放。")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    artifact_dir = run_dir / "artifacts"
    chart_dir = artifact_dir / "charts"
    gif_dir = artifact_dir / "gifs"
    chart_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    rows = read_evaluations(run_dir / "evaluation_5k.csv")
    make_rate_chart(rows, chart_dir / "01_capability_and_safety_5k.png")
    make_progress_chart(rows, chart_dir / "02_progress_diagnostics_5k.png")

    replay_specs = (
        ("00_initial_low_hurdle_failure.gif", run_dir / "initial_model.zip", "low_hurdle"),
        (
            "01_15000_low_hurdle_failure.gif",
            run_dir / "checkpoints" / "model_15000_steps.zip",
            "low_hurdle",
        ),
        ("02_final_low_hurdle_failure.gif", run_dir / "final_model.zip", "low_hurdle"),
        (
            "03_final_short_platform_failure.gif",
            run_dir / "final_model.zip",
            "low_platform_short",
        ),
    )
    replay_rows = [
        {
            "file": name,
            **render_replay(
                model_path,
                gif_dir / name,
                template_name=template_name,
                frame_skip=5,
            ),
        }
        for name, model_path, template_name in replay_specs
    ]
    manifest = {
        "run_dir": str(run_dir.resolve()),
        "charts": [
            "charts/01_capability_and_safety_5k.png",
            "charts/02_progress_diagnostics_5k.png",
        ],
        "replays": replay_rows,
        "interpretation": "calibration_failed_no_full_body_clearance",
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
