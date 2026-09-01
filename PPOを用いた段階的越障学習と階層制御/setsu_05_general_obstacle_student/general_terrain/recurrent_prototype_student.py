"""成功示範へ局所再整列する九十五次元循環プロトタイプ学生。"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = (
    PROJECT_ROOT
    / "runs"
    / "height1_recurrent_dagger_student"
    / "height1_recurrent_dagger_seed7_v1"
    / "datasets"
    / "round_00"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "height1_recurrent_prototype_student"
POSITIONS = tuple(range(20, 31))


class RecurrentPrototypeStudent:
    """観測だけで成功軌跡を選び、履歴近傍で毎歩閉ループ再整列する。"""

    def __init__(
        self,
        dataset_dir: Path,
        *,
        look_behind: int = 12,
        look_ahead: int = 36,
        global_realign_threshold: float = 0.08,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.look_behind = look_behind
        self.look_ahead = look_ahead
        self.global_realign_threshold = global_realign_threshold
        self.trajectories = {}
        for position in POSITIONS:
            data = np.load(dataset_dir / f"x{position}.npz")
            self.trajectories[position] = (
                np.asarray(data["observations"], dtype=np.float32),
                np.asarray(data["actions"], dtype=np.float32),
            )
        all_observations = np.concatenate(
            [trajectory[0] for trajectory in self.trajectories.values()]
        )
        scale = np.std(all_observations, axis=0)
        self.feature_scale = np.maximum(scale, 0.03).astype(np.float32)
        self.position = POSITIONS[0]
        self.index_hint = 0
        self.last_distance = 0.0

    def _distance(self, observations: np.ndarray, target: np.ndarray) -> np.ndarray:
        """特徴分散で正規化した平均二乗観測距離を返す。"""
        normalized = (observations - target[None, :]) / self.feature_scale[None, :]
        return np.mean(normalized * normalized, axis=1)

    def reset(self, observation: np.ndarray) -> int:
        """初期相対地形を含む観測から最も近い成功原型を選ぶ。"""
        target = np.asarray(observation, dtype=np.float32)
        distances = {
            position: float(self._distance(values[0][0:1], target)[0])
            for position, values in self.trajectories.items()
        }
        self.position = min(distances, key=distances.get)
        self.index_hint = 0
        self.last_distance = distances[self.position]
        return self.position

    def predict(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        """履歴近傍の最良状態へ再整列し対応する教師動作を返す。"""
        observations, actions = self.trajectories[self.position]
        target = np.asarray(observation, dtype=np.float32)
        start = max(0, self.index_hint - self.look_behind)
        end = min(len(observations), self.index_hint + self.look_ahead + 1)
        local_distances = self._distance(observations[start:end], target)
        matched_index = start + int(np.argmin(local_distances))
        matched_distance = float(np.min(local_distances))
        globally_realigned = False
        if matched_distance > self.global_realign_threshold:
            global_distances = self._distance(observations, target)
            global_index = int(np.argmin(global_distances))
            global_distance = float(global_distances[global_index])
            if global_distance < matched_distance:
                matched_index = global_index
                matched_distance = global_distance
                globally_realigned = True
        self.index_hint = min(matched_index + 1, len(observations) - 1)
        self.last_distance = matched_distance
        return np.asarray(actions[matched_index], dtype=np.float32), {
            "prototype_position": self.position,
            "matched_index": matched_index,
            "observation_distance": matched_distance,
            "globally_realigned": globally_realigned,
        }


def run_episode(
    student: RecurrentPrototypeStudent,
    *,
    position: int,
    seed: int,
    output_gif: Path | None,
    action_noise_std: float = 0.0,
    action_noise_probability: float = 0.0,
) -> dict[str, object]:
    """学生単独で一壁位置を実行し厳格結果と再整列品質を返す。"""
    course = build_course(
        ["low_hurdle"],
        split="prototype_student_validation",
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array" if output_gif is not None else None,
    )
    frames = []
    rng = np.random.default_rng(seed + 1_000_000)
    try:
        observation, info = environment.reset(seed=seed)
        selected_position = student.reset(observation)
        terminated = False
        truncated = False
        steps = 0
        maximum_angle = 0.0
        upper_contact_steps = 0
        maximum_alignment_distance = 0.0
        global_realignments = 0
        while not (terminated or truncated):
            action, alignment = student.predict(observation)
            maximum_alignment_distance = max(
                maximum_alignment_distance,
                float(alignment["observation_distance"]),
            )
            global_realignments += int(bool(alignment["globally_realigned"]))
            if rng.random() < action_noise_probability:
                action = np.clip(
                    action + rng.normal(0.0, action_noise_std, size=action.shape),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            observation, _, terminated, truncated, info = environment.step(action)
            steps += 1
            maximum_angle = max(maximum_angle, float(info["orientation_error"]))
            upper_contact_steps += int(bool(info["upper_body_grounded"]))
            if output_gif is not None and steps % 5 == 0:
                frame = environment.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
    finally:
        environment.close()
    if output_gif is not None and frames:
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, frames, fps=12, loop=0)
    return {
        "position": position,
        "selected_prototype_position": selected_position,
        "seed": seed,
        "steps": steps,
        "course_complete": bool(info["course_complete"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_angle_degrees": math.degrees(maximum_angle),
        "upper_body_contact_steps": upper_contact_steps,
        "maximum_com_x": float(info["max_x_position"]),
        "maximum_alignment_distance": maximum_alignment_distance,
        "global_realignments": global_realignments,
        "action_noise_std": action_noise_std,
        "action_noise_probability": action_noise_probability,
        "gif": str(output_gif.resolve()) if output_gif is not None else None,
    }


def package_model(output_dir: Path, source_dataset: Path) -> Path:
    """検証済み原型配列と実行設定を一つの学生モデルフォルダへ複製する。"""
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for position in POSITIONS:
        shutil.copy2(source_dataset / f"x{position}.npz", model_dir / f"x{position}.npz")
    manifest = {
        "model_type": "recurrent_local_prototype_policy",
        "observation_dimension": 95,
        "action_dimension": 6,
        "privileged_inputs": False,
        "state": "selected_prototype_and_alignment_index",
        "look_behind": 12,
        "look_ahead": 36,
        "global_realign_threshold": 0.08,
        "source": str(source_dataset.resolve()),
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model_dir


def main() -> None:
    """統一循環学生を梱包し全十一位置の単独合格率を保存する。"""
    parser = argparse.ArgumentParser(description="验收循环原型统一学生。")
    parser.add_argument("--run-name", default="height1_prototype_seed7_v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--action-noise-probability", type=float, default=0.0)
    parser.add_argument("--positions", nargs="+", type=int, default=list(POSITIONS))
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = package_model(output_dir, SOURCE_DATASET)
    student = RecurrentPrototypeStudent(model_dir)
    episodes = []
    for index, position in enumerate(args.positions):
        gif = output_dir / "gifs" / f"student_x{position}.gif" if args.render else None
        episodes.append(
            run_episode(
                student,
                position=position,
                seed=90_000 + index,
                output_gif=gif,
                action_noise_std=args.action_noise_std,
                action_noise_probability=args.action_noise_probability,
            )
        )
    result = {
        "student": "recurrent_local_prototype_policy",
        "observation_dimension": 95,
        "action_dimension": 6,
        "privileged_inputs": False,
        "action_noise_std": args.action_noise_std,
        "action_noise_probability": args.action_noise_probability,
        "episodes": episodes,
        "prototype_selection_accuracy": float(
            np.mean(
                [
                    item["position"] == item["selected_prototype_position"]
                    for item in episodes
                ]
            )
        ),
        "success_rate": float(np.mean([item["course_complete"] for item in episodes])),
        "clearance_rate": float(np.mean([item["raw_clearances"] >= 1 for item in episodes])),
        "hard_fall_rate": float(np.mean([item["hard_fall"] for item in episodes])),
        "all_positions_passed": all(item["course_complete"] for item in episodes),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
