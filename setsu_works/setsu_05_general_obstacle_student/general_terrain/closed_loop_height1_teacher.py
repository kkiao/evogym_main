"""旧専門方策を統一物理環境内の閉ループ高さ一教師へ接続する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import CourseSpec, build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
LEGACY_PROJECT = REPOSITORY_ROOT / "setsu_04_long_legged_hierarchical"
TEACHER_OUTPUT = PROJECT_ROOT / "training_only_teacher" / "generated"
FLAT_TEACHER_PATH = (
    LEGACY_PROJECT
    / "models"
    / "flat_level0_best.zip"
)


class ClosedLoopHeight1Teacher:
    """相対幾何と身体状態で旧専門方策を切り替える訓練専用教師。"""

    def __init__(
        self,
        *,
        post_clear_mode: str = "landing_then_restart",
        clearance_blend: float = 1.0,
        handoff_distance: float = 0.45,
        adaptive_handoff: bool = False,
        clearance_family: str = "first",
        first_switch_fraction: float = 0.5,
        robust_flat_model_path: Path | None = None,
        robust_flat_max_steps: int | None = None,
        obstacle_index: int = 0,
    ) -> None:
        from ll7.final_true_noroll_v2 import ACTION_SCALES, MODEL_PATHS

        required = (
            "approach",
            "first_to_50",
            "first_safe_to_full",
            "first_landing",
            "first_restart",
            "second_to_33",
            "second_to_50",
            "second_to_full",
            "second_landing",
            "second_restart",
        )
        self.models = {
            name: PPO.load(MODEL_PATHS[name].resolve(), device="cpu")
            for name in required
        }
        self.models["flat"] = PPO.load(FLAT_TEACHER_PATH, device="cpu")
        self.robust_flat_model = (
            PPO.load(robust_flat_model_path, device="cpu")
            if robust_flat_model_path is not None
            else None
        )
        if robust_flat_max_steps is not None and robust_flat_max_steps < 1:
            raise ValueError("頑健平地教師の最大歩数は一以上でなければならない。")
        self.robust_flat_max_steps = robust_flat_max_steps
        self.action_scales = ACTION_SCALES
        if post_clear_mode not in {
            "flat",
            "restart",
            "restart_then_flat",
            "restart_brake_flat",
            "landing_then_restart",
        }:
            raise ValueError(f"未知越墙后控制模式：{post_clear_mode}")
        if not 0.0 < clearance_blend <= 1.0:
            raise ValueError("越墙动作混合比例必须位于(0, 1]。")
        if clearance_family not in {"first", "second"}:
            raise ValueError(f"未知越壁方策系列：{clearance_family}")
        if not 0.1 <= first_switch_fraction <= 0.95:
            raise ValueError("第一系列的動作切替率は0.1以上0.95以下でなければならない。")
        self.post_clear_mode = post_clear_mode
        self.clearance_blend = clearance_blend
        self.handoff_distance = handoff_distance
        self.adaptive_handoff = adaptive_handoff
        self.clearance_family = clearance_family
        self.first_switch_fraction = first_switch_fraction
        if obstacle_index < 0:
            raise ValueError("障害物番号は零以上でなければならない。")
        self.obstacle_index = obstacle_index
        self.clearance_started = False
        self.returned_to_flat = False
        self.stage = "flat"
        self.landing_safe_streak = 0
        self.clear_com_x: float | None = None
        self.upright_reference = 0.0
        self.elapsed_steps = 0

    def reset(self, environment: GeneralObstacleEnv) -> None:
        """一回分の閉ループ段階履歴を初期化する。"""
        self.stage = "flat"
        self.landing_safe_streak = 0
        self.clear_com_x = None
        self.clearance_started = False
        self.returned_to_flat = False
        self.elapsed_steps = 0
        base = environment.unwrapped
        self.upright_reference = base.object_orientation_at_time(
            base.get_time(),
            "robot",
        )

    @staticmethod
    def _wrapped_angle(angle: float, reference: float) -> float:
        """初期姿勢からの最短絶対角度を返す。"""
        return abs(math.atan2(math.sin(angle - reference), math.cos(angle - reference)))

    def _select_stage(
        self,
        environment: GeneralObstacleEnv,
        info: dict[str, object],
    ) -> tuple[str, float, float]:
        """固定世界座標を使わず相対通過状態から専門動作を選ぶ。"""
        base = environment.unwrapped
        positions = base.object_pos_at_time(base.get_time(), "robot")
        if self.obstacle_index >= len(base.course.obstacles):
            raise RuntimeError("教師の障害物番号が現在コースの範囲外である。")
        obstacle = base.course.obstacles[self.obstacle_index]
        obstacle_start = obstacle.start_x * base.VOXEL_SIZE
        obstacle_end = (obstacle.end_x + 1) * base.VOXEL_SIZE
        com_x = float(np.mean(positions[0]))
        left_x = float(np.min(positions[0]))
        crossed_fraction = float(np.mean(positions[0] > obstacle_end))
        strictly_cleared = left_x > obstacle_end

        if not strictly_cleared:
            self.landing_safe_streak = 0
            distance = obstacle_start - com_x
            if self.adaptive_handoff and not self.clearance_started:
                velocity_y = float(base.get_vel_com_obs("robot")[1])
                orientation = base.object_orientation_at_time(base.get_time(), "robot")
                angle = self._wrapped_angle(orientation, self.upright_reference)
                favorable_phase = bool(
                    0.25 <= distance <= self.handoff_distance
                    and -0.60 <= velocity_y <= 0.0
                    and angle <= math.radians(10.0)
                )
                self.clearance_started = favorable_phase or distance <= 0.25
            elif not self.adaptive_handoff and distance <= self.handoff_distance:
                self.clearance_started = True
            if not self.clearance_started:
                self.stage = "flat"
            elif self.clearance_family == "second" and crossed_fraction < 1.0 / 3.0:
                self.stage = "second_to_33"
            elif self.clearance_family == "second" and crossed_fraction < 0.5:
                self.stage = "second_to_50"
            elif self.clearance_family == "second":
                self.stage = "second_to_full"
            elif crossed_fraction < self.first_switch_fraction:
                self.stage = "first_to_50"
            else:
                self.stage = "first_safe_to_full"
            return self.stage, 0.0, 0.0

        if self.clear_com_x is None:
            self.clear_com_x = com_x
            if self.post_clear_mode == "flat":
                self.stage = "flat"
            elif self.post_clear_mode in {
                "restart",
                "restart_then_flat",
                "restart_brake_flat",
            }:
                self.stage = (
                    "second_restart"
                    if self.clearance_family == "second"
                    else "first_restart"
                )
            else:
                self.stage = (
                    "second_landing"
                    if self.clearance_family == "second"
                    else "first_landing"
                )

        if self.returned_to_flat:
            self.stage = "flat"
        elif self.post_clear_mode == "flat":
            self.stage = "flat"
        elif self.post_clear_mode in {
            "restart",
            "restart_then_flat",
            "restart_brake_flat",
        }:
            landing_stage = (
                "second_landing"
                if self.clearance_family == "second"
                else "first_landing"
            )
            if self.stage != landing_stage:
                self.stage = (
                    "second_restart"
                    if self.clearance_family == "second"
                    else "first_restart"
                )

        orientation = base.object_orientation_at_time(base.get_time(), "robot")
        angle = self._wrapped_angle(orientation, self.upright_reference)
        speed = float(np.linalg.norm(base.get_vel_com_obs("robot")))
        bottom_y = float(np.min(positions[1]))
        safe_landing = bool(
            angle <= math.radians(45.0)
            and speed <= 0.20
            and bottom_y <= 0.14
            and not bool(info["upper_body_grounded"])
        )
        if self.stage in {"first_landing", "second_landing"}:
            self.landing_safe_streak = (
                self.landing_safe_streak + 1 if safe_landing else 0
            )
            if self.landing_safe_streak >= 20:
                if self.post_clear_mode == "restart_brake_flat":
                    self.returned_to_flat = True
                    self.stage = "flat"
                else:
                    self.stage = (
                        "second_restart"
                        if self.clearance_family == "second"
                        else "first_restart"
                    )

        recovery_distance = max(0.0, com_x - float(self.clear_com_x))
        velocity_y = float(base.get_vel_com_obs("robot")[1])
        safe_flat_handoff = bool(
            angle <= math.radians(20.0)
            and abs(float(info["angular_velocity"])) <= 0.015
            and abs(velocity_y) <= 2.0
            and not bool(info["upper_body_grounded"])
        )
        may_return_to_flat = bool(
            int(info["recovered_obstacles"]) >= 1
            and self.post_clear_mode != "restart"
            and (
                self.post_clear_mode
                not in {"restart_then_flat", "restart_brake_flat"}
                or safe_flat_handoff
            )
        )
        if may_return_to_flat:
            self.returned_to_flat = True
            self.stage = "flat"
        elif (
            int(info["recovered_obstacles"]) >= 1
            and self.post_clear_mode == "restart_brake_flat"
            and not self.returned_to_flat
        ):
            self.stage = (
                "second_landing"
                if self.clearance_family == "second"
                else "first_landing"
            )
        stable_fraction = min(1.0, self.landing_safe_streak / 20.0)
        restart_fraction = min(1.0, recovery_distance / 0.5)
        return self.stage, stable_fraction, restart_fraction

    def _legacy_observation(
        self,
        environment: GeneralObstacleEnv,
        stable_fraction: float,
        restart_fraction: float,
    ) -> np.ndarray:
        """統一物理状態から旧専門方策の九十七次元入力を再構成する。"""
        base = environment.unwrapped
        phase = (
            "landing"
            if self.stage in {"first_landing", "second_landing"}
            else "restart"
            if self.stage in {"first_restart", "second_restart"}
            else "approach"
        )
        phase_values = np.asarray(
            [
                float(phase == "approach"),
                float(phase == "landing"),
                float(phase == "restart"),
                stable_fraction,
                restart_fraction,
            ],
            dtype=np.float32,
        )
        observation = np.concatenate(
            (
                base.get_vel_com_obs("robot"),
                base.get_ort_obs("robot"),
                base.get_relative_pos_obs("robot"),
                base.get_floor_obs("robot", ["ground"], 20),
                phase_values,
            )
        ).astype(np.float32)
        if observation.shape != (97,):
            raise RuntimeError(f"旧教师观测维度异常：{observation.shape}")
        return observation

    def predict(
        self,
        environment: GeneralObstacleEnv,
        info: dict[str, object],
    ) -> tuple[np.ndarray, str]:
        """現物理状態に対する正規化教師動作と段階名を返す。"""
        stage, stable_fraction, restart_fraction = self._select_stage(
            environment,
            info,
        )
        model_key = stage
        legacy_observation = self._legacy_observation(
            environment,
            stable_fraction,
            restart_fraction,
        )
        action, _ = self.models[model_key].predict(
            legacy_observation,
            deterministic=True,
        )
        use_robust_flat = bool(
            stage == "flat"
            and self.robust_flat_model is not None
            and (
                self.robust_flat_max_steps is None
                or self.elapsed_steps < self.robust_flat_max_steps
            )
        )
        if use_robust_flat:
            student_observation = environment.unwrapped._observation()
            action, _ = self.robust_flat_model.predict(
                student_observation,
                deterministic=True,
            )
        scale = float(self.action_scales.get(stage, 1.0))
        normalized = scale * np.asarray(action, dtype=np.float32)
        if stage in {
            "first_to_50",
            "first_safe_to_full",
            "second_to_33",
            "second_to_50",
            "second_to_full",
        }:
            neutral = np.full_like(normalized, -0.2)
            normalized = neutral + self.clearance_blend * (normalized - neutral)
        self.elapsed_steps += 1
        return np.clip(normalized, -1.0, 1.0), stage

    def predict_flat(self, environment: GeneralObstacleEnv) -> np.ndarray:
        """現在状態に対する長距離平地歩行教師動作だけを返す。"""
        self.stage = "flat"
        if self.robust_flat_model is not None:
            student_observation = environment.unwrapped._observation()
            action, _ = self.robust_flat_model.predict(
                student_observation,
                deterministic=True,
            )
            return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        legacy_observation = self._legacy_observation(environment, 0.0, 0.0)
        action, _ = self.models["flat"].predict(
            legacy_observation,
            deterministic=True,
        )
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def run_teacher_episode(
    teacher: ClosedLoopHeight1Teacher,
    course: CourseSpec,
    *,
    seed: int,
    render: bool,
    output_gif: Path | None = None,
) -> dict[str, object]:
    """一コースで教師を最後まで実行し厳格検収結果を返す。"""
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array" if render else None,
    )
    frames: list[np.ndarray] = []
    stage_events: list[dict[str, object]] = []
    try:
        observation, info = environment.reset(seed=seed)
        teacher.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        maximum_angle = 0.0
        upper_contact_steps = 0
        while not (terminated or truncated):
            action, stage = teacher.predict(environment, info)
            if not stage_events or stage_events[-1]["stage"] != stage:
                velocities = environment.unwrapped.get_vel_com_obs("robot")
                stage_events.append(
                    {
                        "step": steps,
                        "stage": stage,
                        "x_position": float(info["x_position"]),
                        "orientation_degrees": math.degrees(
                            float(info["orientation_error"])
                        ),
                        "angular_velocity": float(info["angular_velocity"]),
                        "velocity_x": float(velocities[0]),
                        "velocity_y": float(velocities[1]),
                    }
                )
            observation, _, terminated, truncated, info = environment.step(action)
            del observation
            steps += 1
            maximum_angle = max(maximum_angle, float(info["orientation_error"]))
            upper_contact_steps += int(bool(info["upper_body_grounded"]))
            if render and steps % 4 == 0:
                frame = environment.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
    finally:
        environment.close()
    if output_gif is not None and frames:
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, frames, fps=12, loop=0)
    return {
        "course_id": course.course_id,
        "start_runway_voxels": course.obstacles[0].start_x,
        "seed": seed,
        "steps": steps,
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "course_complete": bool(info["course_complete"]),
        "hard_fall": bool(info["hard_fall"]),
        "stall_limit_reached": bool(info["stall_limit_reached"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_angle_degrees": math.degrees(maximum_angle),
        "upper_body_contact_steps": upper_contact_steps,
        "maximum_com_x": float(info["max_x_position"]),
        "stage_events": stage_events,
        "gif": str(output_gif.resolve()) if output_gif is not None else None,
    }


def main() -> None:
    """三つの壁位置で教師の閉ループ成立を検査する。"""
    parser = argparse.ArgumentParser(description="验证统一环境中的闭环矮墙教师。")
    parser.add_argument("--output-dir", default=str(TEACHER_OUTPUT / "closed_loop_v1"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--post-clear-mode",
        choices=(
            "flat",
            "restart",
            "restart_then_flat",
            "restart_brake_flat",
            "landing_then_restart",
        ),
        default="landing_then_restart",
    )
    parser.add_argument("--clearance-blend", type=float, default=1.0)
    parser.add_argument("--handoff-distance", type=float, default=0.45)
    parser.add_argument("--adaptive-handoff", action="store_true")
    parser.add_argument(
        "--clearance-family",
        choices=("first", "second"),
        default="first",
    )
    parser.add_argument("--first-switch-fraction", type=float, default=0.5)
    parser.add_argument("--robust-flat-model")
    parser.add_argument("--positions", nargs="+", type=int, default=[20, 25, 30])
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher = ClosedLoopHeight1Teacher(
        post_clear_mode=args.post_clear_mode,
        clearance_blend=args.clearance_blend,
        handoff_distance=args.handoff_distance,
        adaptive_handoff=args.adaptive_handoff,
        clearance_family=args.clearance_family,
        first_switch_fraction=args.first_switch_fraction,
        robust_flat_model_path=(
            Path(args.robust_flat_model) if args.robust_flat_model else None
        ),
    )
    rows = []
    for index, start in enumerate(args.positions):
        course = build_course(
            ["low_hurdle"],
            split="teacher_validation",
            seed=60_000 + index,
            difficulty=1,
            start_runway_voxels=start,
        )
        gif = output_dir / f"teacher_x{start}.gif" if args.render else None
        rows.append(
            run_teacher_episode(
                teacher,
                course,
                seed=61_000 + index,
                render=args.render,
                output_gif=gif,
            )
        )
    result = {
        "teacher_type": "privileged_closed_loop_training_teacher",
        "student_observation_dimension": 95,
        "post_clear_mode": args.post_clear_mode,
        "clearance_blend": args.clearance_blend,
        "handoff_distance": args.handoff_distance,
        "adaptive_handoff": args.adaptive_handoff,
        "clearance_family": args.clearance_family,
        "first_switch_fraction": args.first_switch_fraction,
        "robust_flat_model": (
            str(Path(args.robust_flat_model).resolve())
            if args.robust_flat_model
            else None
        ),
        "positions_tested": args.positions,
        "episodes": rows,
        "all_positions_passed": all(row["course_complete"] for row in rows),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
