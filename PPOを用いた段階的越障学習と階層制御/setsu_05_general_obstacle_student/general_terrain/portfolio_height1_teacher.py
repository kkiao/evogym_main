"""複数の検証済み専門動作を状態フィードバックで束ねた高さ一教師。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import build_course
from general_terrain.train_recovery_teacher import safe_flat_handoff


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "recovery_runs"
RAW_RECOVERY_MODEL = (
    RECOVERY_ROOT / "_smoke_recovery_teacher_seed7_v1" / "best_model.zip"
)
HALF_RECOVERY_MODEL = (
    RECOVERY_ROOT
    / "specialist_x23_fraction50_ppo10k_seed7_v4"
    / "best_model.zip"
)


class PortfolioHeight1Teacher:
    """壁位置ごとに実証済み動作を選び、その後は毎歩状態で閉ループ制御する。"""

    def __init__(self, *, flat_model_path: Path | None = None) -> None:
        self.raw_recovery_model = PPO.load(RAW_RECOVERY_MODEL, device="cpu")
        self.half_recovery_model = PPO.load(HALF_RECOVERY_MODEL, device="cpu")
        self.robust_flat_model = (
            PPO.load(flat_model_path, device="cpu")
            if flat_model_path is not None
            else None
        )
        self.controller: ClosedLoopHeight1Teacher | None = None
        self.profile = "early_direct"
        self.phase = "prefix"
        self.safe_streak = 0

    @staticmethod
    def _profile_for_position(position: int) -> str:
        """探索で確認した最小専門集合へ壁位置を割り当てる。"""
        if position == 20:
            return "raw_recovery"
        if position == 23:
            return "half_recovery"
        if position == 25:
            return "mid_direct"
        if position == 26:
            return "tuned_direct"
        return "early_direct"

    @staticmethod
    def _make_controller(profile: str) -> ClosedLoopHeight1Teacher:
        """選択済みプロファイルに対応する旧動作連結器を生成する。"""
        switch_fraction = {
            "early_direct": 0.25,
            "mid_direct": 0.40,
            "tuned_direct": 0.38,
            "raw_recovery": 0.50,
            "half_recovery": 0.50,
        }[profile]
        handoff_distance = 0.40 if profile == "tuned_direct" else 0.45
        return ClosedLoopHeight1Teacher(
            post_clear_mode="restart_then_flat",
            clearance_blend=1.0,
            handoff_distance=handoff_distance,
            adaptive_handoff=True,
            first_switch_fraction=switch_fraction,
        )

    def reset(self, environment: GeneralObstacleEnv) -> None:
        """コースごとの教師プロファイルと安全復帰履歴を初期化する。"""
        position = int(environment.unwrapped.course.obstacles[0].start_x)
        self.profile = self._profile_for_position(position)
        self.controller = self._make_controller(self.profile)
        self.controller.reset(environment)
        self.phase = "prefix"
        self.safe_streak = 0

    @staticmethod
    def _crossed_fraction(environment: GeneralObstacleEnv) -> float:
        """全身体質点のうち壁後端を越えた割合を返す。"""
        base = environment.unwrapped
        positions = base.object_pos_at_time(base.get_time(), "robot")
        obstacle = base.course.obstacles[0]
        obstacle_end = (obstacle.end_x + 1) * base.VOXEL_SIZE
        return float(np.mean(positions[0] > obstacle_end))

    def predict(
        self,
        environment: GeneralObstacleEnv,
        observation: np.ndarray,
        info: dict[str, object],
    ) -> tuple[np.ndarray, str]:
        """現在状態から専門段階を再判定し九十五次元対応動作を返す。"""
        if self.controller is None:
            raise RuntimeError("教師はreset後に使用しなければならない。")

        if int(info["recovered_obstacles"]) >= int(info["obstacle_count"]):
            self.phase = "flat_finish"

        if self.profile == "raw_recovery" and self.phase == "prefix":
            if int(info["raw_clearances"]) >= 1:
                self.phase = "raw_recovery"
        elif self.profile == "half_recovery" and self.phase == "prefix":
            if self._crossed_fraction(environment) >= 0.5:
                self.phase = "half_recovery"

        if self.phase == "prefix":
            action, stage = self.controller.predict(environment, info)
            if stage == "flat" and self.robust_flat_model is not None:
                action, _ = self.robust_flat_model.predict(
                    observation,
                    deterministic=True,
                )
            return action, f"{self.profile}:{stage}"
        if self.phase == "flat_finish":
            if self.robust_flat_model is not None:
                action, _ = self.robust_flat_model.predict(
                    observation,
                    deterministic=True,
                )
                return np.asarray(action, dtype=np.float32), f"{self.profile}:flat_finish"
            return self.controller.predict_flat(environment), f"{self.profile}:flat_finish"

        model = (
            self.raw_recovery_model
            if self.phase == "raw_recovery"
            else self.half_recovery_model
        )
        action, _ = model.predict(observation, deterministic=True)
        ready = safe_flat_handoff(environment, info)
        self.safe_streak = self.safe_streak + 1 if ready else 0
        if self.safe_streak >= 20:
            self.phase = "flat_finish"
        return np.asarray(action, dtype=np.float32), f"{self.profile}:{self.phase}"


def run_episode(
    teacher: PortfolioHeight1Teacher,
    *,
    position: int,
    seed: int,
    output_gif: Path | None,
) -> dict[str, object]:
    """一つの壁位置で閉ループ教師を厳格終了条件まで検査する。"""
    course = build_course(
        ["low_hurdle"],
        split="portfolio_teacher_validation",
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array" if output_gif is not None else None,
    )
    frames: list[np.ndarray] = []
    events = []
    try:
        observation, info = environment.reset(seed=seed)
        teacher.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        maximum_angle = 0.0
        upper_contact_steps = 0
        while not (terminated or truncated):
            action, stage = teacher.predict(environment, observation, info)
            if not events or events[-1]["stage"] != stage:
                events.append({"step": steps, "stage": stage})
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
        "profile": teacher.profile,
        "seed": seed,
        "steps": steps,
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "course_complete": bool(info["course_complete"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_angle_degrees": math.degrees(maximum_angle),
        "upper_body_contact_steps": upper_contact_steps,
        "maximum_com_x": float(info["max_x_position"]),
        "events": events,
        "gif": str(output_gif.resolve()) if output_gif is not None else None,
    }


def main() -> None:
    """全十一位置を検収し教師ゲートの合否を保存する。"""
    parser = argparse.ArgumentParser(description="验收95维环境中的闭环教师组合。")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "training_only_teacher" / "generated" / "portfolio_height1_teacher_v1"),
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--positions", nargs="+", type=int, default=list(range(20, 31)))
    parser.add_argument("--flat-model")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher = PortfolioHeight1Teacher(
        flat_model_path=Path(args.flat_model) if args.flat_model else None
    )
    episodes = []
    for index, position in enumerate(args.positions):
        gif = output_dir / "gifs" / f"teacher_x{position}.gif" if args.render else None
        episodes.append(
            run_episode(
                teacher,
                position=position,
                seed=84_000 + index,
                output_gif=gif,
            )
        )
    result = {
        "teacher": "closed_loop_specialist_portfolio_v1",
        "environment_observation_dimension": 95,
        "student_privileged_inputs": False,
        "teacher_routing_is_training_only": True,
        "flat_model": str(Path(args.flat_model).resolve()) if args.flat_model else None,
        "episodes": episodes,
        "success_rate": float(np.mean([item["course_complete"] for item in episodes])),
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
