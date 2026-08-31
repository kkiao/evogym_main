"""基準示範と安全回復分岐の間を観測で切り替える循環学生。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.recurrent_prototype_student import POSITIONS, SOURCE_DATASET
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCH_RUN = (
    PROJECT_ROOT
    / "runs"
    / "height1_teacher_recovery_branches"
    / "height1_recovery_branches_seed7_v1"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "height1_branching_prototype_student"


class BranchingPrototypeStudent:
    """位置別の複数成功軌跡へ時間局所整列し回復動作を選ぶ。"""

    def __init__(
        self,
        baseline_dir: Path,
        branch_run: Path,
        *,
        look_behind: int = 24,
        look_ahead: int = 48,
    ) -> None:
        self.look_behind = look_behind
        self.look_ahead = look_ahead
        summary = json.loads(
            (branch_run / "summary.json").read_text(encoding="utf-8")
        )
        self.trajectories: dict[int, list[dict[str, object]]] = {
            position: [] for position in POSITIONS
        }
        for position in POSITIONS:
            baseline = np.load(baseline_dir / f"x{position}.npz")
            self.trajectories[position].append(
                {
                    "name": "baseline",
                    "observations": np.asarray(
                        baseline["observations"], dtype=np.float32
                    ),
                    "actions": np.asarray(baseline["actions"], dtype=np.float32),
                }
            )
        for item in summary["branch_index"]:
            position = int(item["position"])
            data = np.load(branch_run / "branches" / item["file"])
            self.trajectories[position].append(
                {
                    "name": item["file"],
                    "observations": np.asarray(
                        data["observations"], dtype=np.float32
                    ),
                    "actions": np.asarray(data["actions"], dtype=np.float32),
                }
            )
        all_observations = np.concatenate(
            [
                np.asarray(trajectory["observations"])
                for values in self.trajectories.values()
                for trajectory in values
            ]
        )
        scale = np.std(all_observations, axis=0)
        self.feature_scale = np.maximum(scale, 0.03).astype(np.float32)
        self.position = POSITIONS[0]
        self.time_hint = 0
        self.current_branch = "baseline"
        self.branch_switches = 0

    def _distance(self, observations: np.ndarray, target: np.ndarray) -> np.ndarray:
        """全九十五特徴を分散正規化した平均二乗距離へ変換する。"""
        normalized = (observations - target[None, :]) / self.feature_scale[None, :]
        return np.mean(normalized * normalized, axis=1)

    def reset(self, observation: np.ndarray) -> int:
        """初期相対地形から位置原型を選び分岐履歴を初期化する。"""
        target = np.asarray(observation, dtype=np.float32)
        distances = {}
        for position in POSITIONS:
            baseline = np.asarray(self.trajectories[position][0]["observations"])
            distances[position] = float(self._distance(baseline[0:1], target)[0])
        self.position = min(distances, key=distances.get)
        self.time_hint = 0
        self.current_branch = "baseline"
        self.branch_switches = 0
        return self.position

    def predict(self, observation: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        """全成功分岐の局所候補から最も近い状態と動作を選ぶ。"""
        target = np.asarray(observation, dtype=np.float32)
        best: tuple[float, str, int, np.ndarray] | None = None
        for trajectory in self.trajectories[self.position]:
            observations = np.asarray(trajectory["observations"])
            actions = np.asarray(trajectory["actions"])
            start = max(0, self.time_hint - self.look_behind)
            end = min(len(observations), self.time_hint + self.look_ahead + 1)
            if start >= end:
                continue
            distances = self._distance(observations[start:end], target)
            local_index = int(np.argmin(distances))
            candidate = (
                float(distances[local_index]),
                str(trajectory["name"]),
                start + local_index,
                actions[start + local_index],
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            raise RuntimeError("利用可能な回復分岐候補が存在しない。")
        distance, branch, index, action = best
        if branch != self.current_branch:
            self.branch_switches += 1
            self.current_branch = branch
        self.time_hint = index + 1
        return np.asarray(action, dtype=np.float32), {
            "prototype_position": self.position,
            "branch": branch,
            "matched_index": index,
            "observation_distance": distance,
            "branch_switches": self.branch_switches,
        }


def run_episode(
    student: BranchingPrototypeStudent,
    *,
    position: int,
    seed: int,
    action_noise_std: float,
    action_noise_probability: float,
    output_gif: Path | None,
) -> dict[str, object]:
    """独立乱数の動作摂動下で学生を最後まで厳格検収する。"""
    course = build_course(
        ["low_hurdle"],
        split="branching_student_validation",
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array" if output_gif is not None else None,
    )
    rng = np.random.default_rng(seed + 2_000_000)
    frames = []
    try:
        observation, info = environment.reset(seed=seed)
        selected_position = student.reset(observation)
        terminated = False
        truncated = False
        steps = 0
        maximum_angle = 0.0
        upper_contact_steps = 0
        disturbance_count = 0
        maximum_alignment_distance = 0.0
        while not (terminated or truncated):
            action, alignment = student.predict(observation)
            maximum_alignment_distance = max(
                maximum_alignment_distance,
                float(alignment["observation_distance"]),
            )
            if rng.random() < action_noise_probability:
                action = np.clip(
                    action + rng.normal(0.0, action_noise_std, size=action.shape),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                disturbance_count += 1
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
        "disturbance_count": disturbance_count,
        "course_complete": bool(info["course_complete"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_angle_degrees": math.degrees(maximum_angle),
        "upper_body_contact_steps": upper_contact_steps,
        "maximum_com_x": float(info["max_x_position"]),
        "maximum_alignment_distance": maximum_alignment_distance,
        "branch_switches": student.branch_switches,
        "final_branch": student.current_branch,
        "gif": str(output_gif.resolve()) if output_gif is not None else None,
    }


def main() -> None:
    """分岐学生を十一位置で測り九対十一かつ無硬側倒の門を判定する。"""
    parser = argparse.ArgumentParser(description="验收成功恢复分支扩展后的统一学生。")
    parser.add_argument("--run-name", default="height1_branching_seed7_v1")
    parser.add_argument("--branch-run", default=str(DEFAULT_BRANCH_RUN))
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument("--action-noise-probability", type=float, default=0.01)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    student = BranchingPrototypeStudent(SOURCE_DATASET, Path(args.branch_run))
    episodes = []
    for index, position in enumerate(POSITIONS):
        gif = output_dir / "gifs" / f"student_x{position}.gif" if args.render else None
        episodes.append(
            run_episode(
                student,
                position=position,
                seed=120_000 + index,
                action_noise_std=args.action_noise_std,
                action_noise_probability=args.action_noise_probability,
                output_gif=gif,
            )
        )
    success_count = sum(item["course_complete"] for item in episodes)
    hard_fall_count = sum(item["hard_fall"] for item in episodes)
    result = {
        "student": "recurrent_branching_prototype_policy_v1",
        "observation_dimension": 95,
        "privileged_inputs": False,
        "branch_run": str(Path(args.branch_run).resolve()),
        "action_noise_std": args.action_noise_std,
        "action_noise_probability": args.action_noise_probability,
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "clearance_rate": float(
            np.mean([item["raw_clearances"] >= 1 for item in episodes])
        ),
        "hard_fall_count": hard_fall_count,
        "hard_fall_rate": hard_fall_count / len(episodes),
        "robustness_gate_passed": bool(success_count >= 9 and hard_fall_count == 0),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
