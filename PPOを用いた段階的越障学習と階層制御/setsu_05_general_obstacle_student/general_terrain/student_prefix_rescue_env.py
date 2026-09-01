"""凍結学生プレフィックスを再生し、救援教師専用の課題を作る。"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Protocol

import gymnasium as gym
import numpy as np

from general_terrain.curriculum import sample_curriculum_course
from general_terrain.rescue_reset_manifest import RescueResetSpec
from general_terrain.terrain import CourseSpec


RESCUE_REWARD_VERSION = "student_prefix_rescue_teacher_v1"
PRE_HURDLE_PHASE = "pre_hurdle"
HURDLE_DEFORMATION_PHASE = "hurdle_deformation"
POST_CLEARANCE_RECOVERY_PHASE = "post_clearance_recovery"
POST_RECOVERY_STALL_PHASE = "post_recovery_stall"
RESCUE_PHASES = (
    PRE_HURDLE_PHASE,
    HURDLE_DEFORMATION_PHASE,
    POST_CLEARANCE_RECOVERY_PHASE,
    POST_RECOVERY_STALL_PHASE,
)


class FrozenPrefixStudent(Protocol):
    """救援開始状態まで再生する循環学生のインターフェースを定義する。"""

    def predict(
        self,
        observation: np.ndarray,
        *,
        state: Any,
        episode_start: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.ndarray, Any]:
        """現在観測から決定論的動作と次の循環状態を返す。"""


def classify_rescue_phase(
    environment: Any,
    info: Mapping[str, object],
    *,
    post_recovery_stall_steps: int = 20,
) -> str:
    """教師だけが使う物理状態を四つの監査段階に分類する。"""
    if post_recovery_stall_steps < 1:
        raise ValueError("回復後停滞歩数は一以上でなければならない。")
    recovered = int(info["recovered_obstacles"])
    obstacle_count = int(info["obstacle_count"])
    if (
        recovered >= obstacle_count
        and int(info["stall_steps"]) >= post_recovery_stall_steps
    ):
        return POST_RECOVERY_STALL_PHASE
    if int(info["raw_clearances"]) > 0:
        return POST_CLEARANCE_RECOVERY_PHASE
    base = environment.unwrapped
    positions = base.object_pos_at_time(base.get_time(), "robot")
    obstacle_start = base.course.obstacles[0].start_x * base.VOXEL_SIZE
    if float(np.max(positions[0])) >= obstacle_start:
        return HURDLE_DEFORMATION_PHASE
    return PRE_HURDLE_PHASE


class StudentPrefixRescueEnv(gym.Env):
    """学生の凍結プレフィックス後だけを救援教師に公開する。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        student: FrozenPrefixStudent,
        reset_specs: tuple[RescueResetSpec, ...],
        *,
        stage: str = "hurdle_single",
        max_rescue_steps: int = 800,
        reference_tolerance: float = 1e-9,
        enforce_reference_metrics: bool = True,
        environment_factory: Callable[[CourseSpec], Any] | None = None,
    ) -> None:
        if not reset_specs:
            raise ValueError("救援重置点は一つ以上必要である。")
        if max_rescue_steps < 1:
            raise ValueError("救援上限歩数は一以上でなければならない。")
        if reference_tolerance < 0.0:
            raise ValueError("出典許容誤差は零以上でなければならない。")
        self.student = student
        self.reset_specs = reset_specs
        self.specs_by_seed = {spec.seed: spec for spec in reset_specs}
        if len(self.specs_by_seed) != len(reset_specs):
            raise ValueError("救援重置点の乱数種が重複している。")
        self.stage = stage
        self.max_rescue_steps = max_rescue_steps
        self.reference_tolerance = reference_tolerance
        self.enforce_reference_metrics = enforce_reference_metrics
        self._factory = environment_factory
        first = reset_specs[0]
        first_course = sample_curriculum_course(first.seed, stage, "train")
        if environment_factory is None:
            from general_terrain.environment import GeneralObstacleEnv

            self.environment = GeneralObstacleEnv(
                course=first_course,
                resample_on_reset=False,
            )
        else:
            self.environment = environment_factory(first_course)
        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space
        self._next_index = 0
        self.current_spec: RescueResetSpec | None = None
        self.rescue_steps = 0
        self.previous_raw_clearances = 0
        self.previous_recovered_obstacles = 0

    @property
    def base_environment(self) -> Any:
        """物理監査のために内部環境を読み取り専用で返す。"""
        return self.environment

    @property
    def schema(self) -> tuple[str, ...]:
        """救援教師が受け取る観測名の固定順を返す。"""
        return tuple(self.environment.unwrapped.schema)

    def _select_spec(
        self,
        options: dict[str, object] | None,
        seed: int | None,
    ) -> RescueResetSpec:
        """明示的な乱数シードまたは循環順で次のリセット地点を選ぶ。"""
        if options and "prefix_seed" in options:
            prefix_seed = int(options["prefix_seed"])
            try:
                return self.specs_by_seed[prefix_seed]
            except KeyError as error:
                raise ValueError(
                    f"救援重置点にない訓練乱数種: {prefix_seed}"
                ) from error
        if seed is not None:
            return self.reset_specs[int(seed) % len(self.reset_specs)]
        selected = self.reset_specs[self._next_index % len(self.reset_specs)]
        self._next_index += 1
        return selected

    def _check_reference(self, info: Mapping[str, object]) -> None:
        """再生状態がM2.1の凍結済み引き継ぎ値と一致するか検査する。"""
        if not self.enforce_reference_metrics or self.current_spec is None:
            return
        spec = self.current_spec
        float_fields = (
            ("x_position", spec.x_position),
            ("orientation_error", spec.orientation_error),
            ("angular_velocity", spec.angular_velocity),
        )
        for name, expected in float_fields:
            if not math.isclose(
                float(info[name]),
                expected,
                rel_tol=0.0,
                abs_tol=self.reference_tolerance,
            ):
                raise RuntimeError(
                    f"救援重置状態が出典と一致しない: {name}"
                )
        integer_fields = (
            ("stall_steps", spec.stall_steps),
            ("raw_clearances", spec.raw_clearances),
            ("recovered_obstacles", spec.recovered_obstacles),
        )
        if any(int(info[name]) != expected for name, expected in integer_fields):
            raise RuntimeError("救援重置状態の計数値が出典と一致しない。")

    def _augment_info(self, info: Mapping[str, object]) -> dict[str, object]:
        """教師訓練専用の監査項目を複製した情報へ追加する。"""
        if self.current_spec is None:
            raise RuntimeError("救援環境はreset後に使用しなければならない。")
        enriched = dict(info)
        enriched.update(
            {
                "rescue_reward_version": RESCUE_REWARD_VERSION,
                "rescue_reset_seed": self.current_spec.seed,
                "student_prefix_steps": self.current_spec.prefix_steps,
                "rescue_steps": self.rescue_steps,
                "rescue_phase": classify_rescue_phase(self.environment, info),
                "teacher_training_only": True,
                "student_controller_active": False,
                "student_observation_privileged": False,
            }
        )
        return enriched

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """凍結学生を決定論的に引き継ぎステップまで再生し、教師へ渡す。"""
        gym.Env.reset(self, seed=seed)
        spec = self._select_spec(options, seed)
        course = sample_curriculum_course(spec.seed, self.stage, "train")
        if course.obstacles[0].start_x != spec.start_runway_voxels:
            raise RuntimeError("救援重置目録とコース開始位置が一致しない。")
        observation, info = self.environment.reset(
            seed=spec.seed,
            options={"course": course},
        )
        recurrent_state: Any = None
        episode_start = np.ones((1,), dtype=bool)
        for step in range(spec.prefix_steps):
            student_action, recurrent_state = self.student.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            observation, _, terminated, truncated, info = self.environment.step(
                np.asarray(student_action, dtype=np.float32)
            )
            episode_start[:] = False
            if terminated or truncated:
                raise RuntimeError(
                    f"学生前缀が凍結接管歩前に終了した: {spec.seed}, {step + 1}"
                )
        self.current_spec = spec
        self.rescue_steps = 0
        self.previous_raw_clearances = int(info["raw_clearances"])
        self.previous_recovered_obstacles = int(info["recovered_obstacles"])
        self._check_reference(info)
        return np.asarray(observation, dtype=np.float32), self._augment_info(info)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """救援教師動作を一歩実行し回復重視報酬と終了条件を返す。"""
        if self.current_spec is None:
            raise RuntimeError("救援環境はreset後に使用しなければならない。")
        observation, base_reward, terminated, truncated, info = (
            self.environment.step(np.asarray(action, dtype=np.float32))
        )
        self.rescue_steps += 1
        raw_clearances = int(info["raw_clearances"])
        recovered_obstacles = int(info["recovered_obstacles"])
        raw_gain = max(0, raw_clearances - self.previous_raw_clearances)
        recovery_gain = max(
            0,
            recovered_obstacles - self.previous_recovered_obstacles,
        )
        reward = float(base_reward) - 0.002
        reward += 8.0 * raw_gain + 15.0 * recovery_gain
        if bool(info["upper_body_grounded"]):
            reward -= 0.10
        if bool(info["hard_fall"]):
            reward -= 15.0
            terminated = True
            truncated = False
        if bool(info.get("sequence_failed", False)):
            terminated = True
            truncated = False
        success = bool(
            info["course_complete"]
            and not info["hard_fall"]
            and recovered_obstacles >= int(info["obstacle_count"])
        )
        if success:
            reward += 25.0
            terminated = True
            truncated = False
        elif self.rescue_steps >= self.max_rescue_steps and not terminated:
            truncated = True
        self.previous_raw_clearances = raw_clearances
        self.previous_recovered_obstacles = recovered_obstacles
        enriched = self._augment_info(info)
        enriched["rescue_success"] = success
        enriched["rescue_step_limit_reached"] = bool(
            self.rescue_steps >= self.max_rescue_steps
        )
        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            enriched,
        )

    def close(self) -> None:
        """内部のEvoGym環境と表示資源を閉じる。"""
        self.environment.close()
