"""成功分岐の近傍状態を使う九十五次元記憶型学生を厳格評価する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.train_noisy_height1_teacher import POSITIONS, course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCH_RUN = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_teacher_search"
    / "noisy_teacher_portfolio_seed7_v1"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "height1_branch_memory_student"


class BranchMemoryStudent:
    """絶対位置を使わず現在観測と時系列近傍から教師動作を選ぶ。"""

    def __init__(
        self,
        branch_run: Path,
        *,
        search_radius: int = 20,
        out_of_distribution_threshold: float = float("inf"),
    ) -> None:
        self.sequences = []
        all_observations = []
        for branch_path in sorted((branch_run / "branches").glob("x*_rescued.npz")):
            data = np.load(branch_path)
            observations = np.asarray(data["observations"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            self.sequences.append((observations, actions, branch_path.name))
            all_observations.append(observations)
        if not self.sequences:
            raise RuntimeError(f"安全分岐が見つからない：{branch_run}")
        stacked = np.concatenate(all_observations)
        self.mean = np.mean(stacked, axis=0).astype(np.float32)
        self.scale = np.maximum(np.std(stacked, axis=0), 0.02).astype(np.float32)
        self.search_radius = search_radius
        self.out_of_distribution_threshold = out_of_distribution_threshold
        self.step_index = 0
        self.last_branch = ""
        self.last_distance = 0.0
        self.safe_fallback = False

    def reset(self) -> None:
        """一回分の時刻と診断値を初期化する。"""
        self.step_index = 0
        self.last_branch = ""
        self.last_distance = 0.0
        self.safe_fallback = False

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """現在時刻付近の全分岐から最も近い状態の動作を返す。"""
        normalized = (np.asarray(observation, dtype=np.float32) - self.mean) / self.scale
        best_distance = float("inf")
        best_action = None
        best_branch = ""
        for observations, actions, name in self.sequences:
            start = max(0, self.step_index - self.search_radius)
            end = min(len(observations), self.step_index + self.search_radius + 1)
            if start >= end:
                continue
            candidates = (observations[start:end] - self.mean) / self.scale
            distances = np.mean((candidates - normalized) ** 2, axis=1)
            local_index = int(np.argmin(distances))
            distance = float(distances[local_index])
            if distance < best_distance:
                best_distance = distance
                best_action = actions[start + local_index]
                best_branch = name
        if best_action is None:
            raise RuntimeError("時系列近傍動作を選択できなかった。")
        self.step_index += 1
        self.last_branch = best_branch
        self.last_distance = best_distance
        if best_distance > self.out_of_distribution_threshold:
            self.safe_fallback = True
        if self.safe_fallback:
            self.last_branch = "safe_fallback"
            return np.full(6, -0.2, dtype=np.float32)
        return np.asarray(best_action, dtype=np.float32)


def evaluate(
    student: BranchMemoryStudent,
    *,
    seed: int,
    noise_std: float,
    noise_probability: float,
) -> dict[str, object]:
    """位置情報を学生へ渡さず十一位置の摂動成績を返す。"""
    episodes = []
    for index, position in enumerate(POSITIONS):
        episode_seed = seed + index
        environment = GeneralObstacleEnv(
            course=course(position, episode_seed, "branch_memory_validation"),
            resample_on_reset=False,
        )
        rng = np.random.default_rng(episode_seed + 5_000_000)
        branch_counts: dict[str, int] = {}
        disturbance_count = 0
        maximum_angle = 0.0
        upper_contact_steps = 0
        maximum_branch_distance = 0.0
        try:
            observation, info = environment.reset(seed=episode_seed)
            student.reset()
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                action = student.predict(observation)
                maximum_branch_distance = max(
                    maximum_branch_distance,
                    student.last_distance,
                )
                branch_counts[student.last_branch] = (
                    branch_counts.get(student.last_branch, 0) + 1
                )
                if rng.random() < noise_probability:
                    action = np.clip(
                        action + rng.normal(0.0, noise_std, size=action.shape),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                    disturbance_count += 1
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                maximum_angle = max(maximum_angle, float(info["orientation_error"]))
                upper_contact_steps += int(bool(info["upper_body_grounded"]))
        finally:
            environment.close()
        episodes.append(
            {
                "position": position,
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
                "branch_usage": branch_counts,
                "maximum_branch_distance": maximum_branch_distance,
                "safe_fallback_used": "safe_fallback" in branch_counts,
            }
        )
    success_count = sum(item["course_complete"] for item in episodes)
    hard_fall_count = sum(item["hard_fall"] for item in episodes)
    return {
        "episodes": episodes,
        "success_count": success_count,
        "clearance_count": sum(item["raw_clearances"] >= 1 for item in episodes),
        "hard_fall_count": hard_fall_count,
        "robustness_gate_passed": bool(
            success_count >= 9 and hard_fall_count == 0
        ),
    }


def main() -> None:
    """分岐記憶学生の固定門と未見摂動を連続評価する。"""
    parser = argparse.ArgumentParser(description="验收95维分支记忆学生。")
    parser.add_argument("--run-name", default="branch_memory_student_seed7_v1")
    parser.add_argument("--branch-run", default=str(DEFAULT_BRANCH_RUN))
    parser.add_argument("--search-radius", type=int, default=20)
    parser.add_argument("--ood-threshold", type=float, default=float("inf"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    student = BranchMemoryStudent(
        Path(args.branch_run),
        search_radius=args.search_radius,
        out_of_distribution_threshold=args.ood_threshold,
    )
    fixed_gate = evaluate(
        student,
        seed=args.seed + 70_000,
        noise_std=args.noise_std,
        noise_probability=args.noise_probability,
    )
    held_out = evaluate(
        student,
        seed=args.seed + 80_000,
        noise_std=args.noise_std,
        noise_probability=args.noise_probability,
    )
    summary = {
        "method": "nonparametric_sequence_branch_memory",
        "observation_dimension": 95,
        "privileged_student_inputs": False,
        "absolute_position_input": False,
        "search_radius": args.search_radius,
        "out_of_distribution_threshold": args.ood_threshold,
        "fixed_gate_evaluation": fixed_gate,
        "held_out_evaluation": held_out,
        "robustness_gate_passed": fixed_gate["robustness_gate_passed"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
