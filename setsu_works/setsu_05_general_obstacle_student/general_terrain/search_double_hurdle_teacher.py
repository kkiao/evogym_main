"""三身体幅間隔の二つの低壁を連続処理する教師設定を探索する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.search_noisy_teacher_portfolio import configurations
from general_terrain.search_noisy_teacher_portfolio import ROBUST_FLAT_MODEL
from general_terrain.terrain import BODY_WIDTH_VOXELS, build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "double_hurdle_search"
SINGLE_SEARCH = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_teacher_search"
    / "noisy_teacher_portfolio_seed7_v1"
    / "summary.json"
)
POSITIONS = tuple(range(20, 31))


def course(position: int, seed: int):
    """二つの低壁を三身体幅で配置した固定課程を返す。"""
    return build_course(
        ["low_hurdle", "low_hurdle"],
        split="double_hurdle_teacher_search",
        seed=seed,
        difficulty=1,
        gaps=[3 * BODY_WIDTH_VOXELS],
        start_runway_voxels=position,
    )


class SequentialHeight1Teacher:
    """障害回復数に応じて同じ閉ループ技能を次の壁へ移す。"""

    def __init__(
        self,
        first_configuration: dict[str, object],
        second_configuration: dict[str, object] | None = None,
    ) -> None:
        self.configurations = (
            first_configuration,
            second_configuration or first_configuration,
        )
        self.robust_flat_model = (
            PPO.load(ROBUST_FLAT_MODEL, device="cpu")
            if any(bool(item["robust_flat"]) for item in self.configurations)
            else None
        )
        self.controller = ClosedLoopHeight1Teacher(
            post_clear_mode=str(first_configuration["post_clear_mode"]),
            clearance_blend=1.0,
            handoff_distance=float(first_configuration["handoff_distance"]),
            adaptive_handoff=True,
            clearance_family=str(first_configuration["clearance_family"]),
            first_switch_fraction=float(first_configuration["first_switch_fraction"]),
            obstacle_index=0,
        )
        self.active_obstacle = 0

    def _apply_configuration(self, obstacle_index: int) -> None:
        """指定壁に対応する検証済み切替設定を制御器へ反映する。"""
        configuration = self.configurations[obstacle_index]
        self.controller.post_clear_mode = str(configuration["post_clear_mode"])
        self.controller.handoff_distance = float(configuration["handoff_distance"])
        self.controller.clearance_family = str(configuration["clearance_family"])
        self.controller.first_switch_fraction = float(
            configuration["first_switch_fraction"]
        )
        self.controller.robust_flat_model = (
            self.robust_flat_model if bool(configuration["robust_flat"]) else None
        )

    def reset(self, environment: GeneralObstacleEnv) -> None:
        """最初の壁へ制御対象と位相履歴を戻す。"""
        self.active_obstacle = 0
        self.controller.obstacle_index = 0
        self._apply_configuration(0)
        self.controller.reset(environment)

    def predict(
        self,
        environment: GeneralObstacleEnv,
        info: dict[str, object],
    ) -> tuple[np.ndarray, str]:
        """回復済み壁の次へ位相を切り替えて現在動作を返す。"""
        recovered = int(info["recovered_obstacles"])
        obstacle_count = len(environment.unwrapped.course.obstacles)
        if recovered > self.active_obstacle and self.active_obstacle + 1 < obstacle_count:
            self.active_obstacle += 1
            self.controller.obstacle_index = self.active_obstacle
            self._apply_configuration(self.active_obstacle)
            self.controller.reset(environment)
        action, stage = self.controller.predict(environment, info)
        return action, f"obstacle_{self.active_obstacle}:{stage}"


def run_episode(
    teacher: SequentialHeight1Teacher | None,
    *,
    position: int,
    seed: int,
    noise_std: float,
    noise_probability: float,
    save_trajectory: bool,
    stop_after_recovered: int | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    """一つの二壁課程を固定摂動で実行し厳格結果を返す。"""
    environment = GeneralObstacleEnv(course=course(position, seed), resample_on_reset=False)
    rng = np.random.default_rng(seed + 9_000_000)
    observations = []
    actions = []
    stages = []
    disturbance_count = 0
    maximum_angle = 0.0
    upper_contact_steps = 0
    try:
        observation, info = environment.reset(seed=seed)
        if teacher is not None:
            teacher.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            if teacher is None:
                action = np.full(environment.action_space.shape, -0.2, dtype=np.float32)
                stage = "safe_stall"
            else:
                action, stage = teacher.predict(environment, info)
            if save_trajectory:
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                stages.append(stage)
            executed_action = np.asarray(action, dtype=np.float32)
            if rng.random() < noise_probability:
                executed_action = np.clip(
                    executed_action
                    + rng.normal(0.0, noise_std, size=executed_action.shape),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                disturbance_count += 1
            observation, _, terminated, truncated, info = environment.step(executed_action)
            steps += 1
            maximum_angle = max(maximum_angle, float(info["orientation_error"]))
            upper_contact_steps += int(bool(info["upper_body_grounded"]))
            if (
                stop_after_recovered is not None
                and int(info["recovered_obstacles"]) >= stop_after_recovered
            ):
                break
    finally:
        environment.close()
    result = {
        "position": position,
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
    }
    trajectory = None
    if save_trajectory:
        trajectory = {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "stages": np.asarray(stages),
        }
    return result, trajectory


def main() -> None:
    """位置別教師を探索し二壁成功分岐を保存する。"""
    parser = argparse.ArgumentParser(description="搜索連続する二つの低壁教师。")
    parser.add_argument("--run-name", default="double_hurdle_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-successes", type=int, default=9)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "branches").mkdir()
    single_summary = json.loads(SINGLE_SEARCH.read_text(encoding="utf-8"))
    first_configurations = {
        int(position): item["configuration"]
        for position, item in single_summary["solutions"].items()
    }
    solutions: dict[int, dict[str, object]] = {}
    search_rows = []
    for configuration_index, configuration in enumerate(configurations()):
        if len(solutions) >= args.target_successes:
            break
        if bool(configuration["robust_flat"]):
            continue
        for position in POSITIONS:
            if position in solutions or position not in first_configurations:
                continue
            teacher = SequentialHeight1Teacher(
                first_configurations[position],
                configuration,
            )
            seed = args.seed + 91_000 + (position - POSITIONS[0])
            result, _ = run_episode(
                teacher,
                position=position,
                seed=seed,
                noise_std=args.noise_std,
                noise_probability=args.noise_probability,
                save_trajectory=False,
            )
            search_rows.append(
                {
                    "configuration_index": configuration_index,
                    "configuration": configuration,
                    "result": result,
                }
            )
            if result["course_complete"] and not result["hard_fall"]:
                solutions[position] = {
                    "configuration_index": configuration_index,
                    "first_configuration": first_configurations[position],
                    "second_configuration": configuration,
                }
                print(
                    json.dumps(
                        {
                            "solved_position": position,
                            "configuration_index": configuration_index,
                            "solved_count": len(solutions),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    validation = []
    for position in POSITIONS:
        solution = solutions.get(position)
        teacher = (
            SequentialHeight1Teacher(
                solution["first_configuration"],
                solution["second_configuration"],
            )
            if solution
            else None
        )
        seed = args.seed + 91_000 + (position - POSITIONS[0])
        result, trajectory = run_episode(
            teacher,
            position=position,
            seed=seed,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
            save_trajectory=solution is not None,
        )
        result["configuration_index"] = (
            int(solution["configuration_index"]) if solution else None
        )
        validation.append(result)
        if trajectory is not None and result["course_complete"] and not result["hard_fall"]:
            np.savez_compressed(
                output_dir / "branches" / f"x{position}_double_hurdle.npz",
                **trajectory,
            )
    success_count = sum(item["course_complete"] for item in validation)
    hard_fall_count = sum(item["hard_fall"] for item in validation)
    summary = {
        "course": "two_low_hurdles_gap_three_body_widths",
        "solutions": {str(key): value for key, value in solutions.items()},
        "validation": validation,
        "success_count": success_count,
        "hard_fall_count": hard_fall_count,
        "curriculum_gate_passed": bool(
            success_count >= args.target_successes and hard_fall_count == 0
        ),
        "search_evaluations": len(search_rows),
    }
    (output_dir / "search_rows.json").write_text(
        json.dumps(search_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
