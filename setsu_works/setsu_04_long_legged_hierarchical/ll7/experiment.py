"""長脚七段階実験の環境構築、評価、ファイル管理を共通化する。"""

from __future__ import annotations

import csv
import gc
import json
import math
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from evogym import get_full_connectivity
from stable_baselines3.common.monitor import Monitor

from ll7.environment import LongLeggedCurriculumEnv
from ll7.environment import LANDING_ANGLE_LIMIT
from ll7.environment import LANDING_SPEED_LIMIT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
EVALUATION_FIELDS = (
    "timesteps",
    "mean_return",
    "std_return",
    "mean_steps",
    "mean_displacement",
    "mean_speed",
    "mean_final_x",
    "mean_max_x",
    "mean_strict_clearances",
    "mean_stable_landings",
    "mean_restart_successes",
    "mean_validated_obstacles",
    "mean_obstacle_fraction",
    "success_rate",
)


class ActionRescaleWrapper(gym.ActionWrapper):
    """PPOの[-1, 1]行動をEvoGymの[0.6, 1.6]へ線形変換する。"""

    def __init__(self, env):
        super().__init__(env)
        self.native_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.native_high = np.asarray(env.action_space.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action):
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scale = (clipped + 1.0) / 2.0
        return self.native_low + scale * (self.native_high - self.native_low)


class TeacherClearancePrefixWrapper(gym.Wrapper):
    """教師が指定位置または完全通過へ達した状態から学習試行を開始する。"""

    def __init__(
        self,
        env,
        teacher_model,
        target_clearances: int,
        max_prefix_steps: int,
        handoff_x: float | None = None,
    ):
        super().__init__(env)
        self.teacher_model = teacher_model
        self.target_clearances = target_clearances
        self.max_prefix_steps = max_prefix_steps
        self.handoff_x = handoff_x

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prefix_return = 0.0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            action, _ = self.teacher_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            prefix_return += float(reward)
            if self.handoff_x is None:
                reached = (
                    int(info["strict_clearances"]) >= self.target_clearances
                    and info["phase"] == "landing"
                )
            else:
                reached = float(info["x_position"]) >= self.handoff_x
            if reached:
                break
            if terminated or truncated:
                raise RuntimeError("教师策略在到达严格越障状态前提前结束。")
        else:
            raise RuntimeError("教师策略未能在最大前缀步数内严格越过障碍。")
        info = dict(info)
        info["prefix_steps"] = prefix_steps
        info["prefix_return"] = prefix_return
        return obs, info


class UprightClearanceGoalWrapper(gym.Wrapper):
    """完全通過時の姿勢角だけを短い中間課題として評価する。"""

    def __init__(
        self,
        env,
        max_clearance_speed: float | None = None,
        max_clearance_angle: float = LANDING_ANGLE_LIMIT,
    ):
        super().__init__(env)
        self.max_clearance_speed = max_clearance_speed
        self.max_clearance_angle = max_clearance_angle

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["upright_clearance_success"] = False
        info["clearance_angle"] = math.pi
        info["clearance_speed"] = float("inf")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        success = False
        angle = float(info["orientation_error"])
        speed = float(info["com_speed"])
        if info.get("new_strict_clearance", False):
            quality = 0.5 * (1.0 + math.cos(angle))
            reward += 200.0 * quality
            speed_success = (
                self.max_clearance_speed is None
                or speed <= self.max_clearance_speed
            )
            if self.max_clearance_speed is not None:
                reward += 400.0 * math.exp(-speed / self.max_clearance_speed)
            success = angle <= self.max_clearance_angle and speed_success
            if success:
                reward += 800.0
            else:
                # 横転通過を低速ボーナスだけで選ばないよう超過量を強く罰する。
                angle_excess = max(0.0, angle / self.max_clearance_angle - 1.0)
                speed_excess = 0.0
                if self.max_clearance_speed is not None:
                    speed_excess = max(
                        0.0,
                        speed / self.max_clearance_speed - 1.0,
                    )
                reward -= 600.0 * min(
                    3.0,
                    2.0 * angle_excess + speed_excess,
                )
            terminated = True
        info["upright_clearance_success"] = success
        info["clearance_angle"] = angle
        info["clearance_speed"] = speed
        info["is_success"] = success
        return obs, reward, terminated, truncated, info


class HierarchicalClearancePrefixWrapper(gym.Wrapper):
    """接近教師と直立通過教師を順に使い、完全通過直後まで進める。"""

    def __init__(
        self,
        env,
        approach_model,
        clearance_model,
        handoff_x: float,
        max_prefix_steps: int,
    ):
        super().__init__(env)
        self.approach_model = approach_model
        self.clearance_model = clearance_model
        self.handoff_x = handoff_x
        self.max_prefix_steps = max_prefix_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prefix_return = 0.0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            use_approach = (
                info["phase"] == "approach"
                and float(info["x_position"]) < self.handoff_x
            )
            model = self.approach_model if use_approach else self.clearance_model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            prefix_return += float(reward)
            reached = int(info["strict_clearances"]) >= 1 and info["phase"] == "landing"
            if reached:
                break
            if terminated or truncated:
                raise RuntimeError("分层教师在直立完整越障前提前结束。")
        else:
            raise RuntimeError("分层教师未能在最大前缀步数内完整越障。")
        info = dict(info)
        info["prefix_steps"] = prefix_steps
        info["prefix_return"] = prefix_return
        return obs, info


class SafeBrakePrefixWrapper(gym.Wrapper):
    """安全な制動方策で低速状態を作り、扶正学習だけを切り出す。"""

    def __init__(
        self,
        env,
        brake_model,
        action_scale: float,
        target_speed: float,
        stable_steps: int,
        max_prefix_steps: int,
    ):
        super().__init__(env)
        self.brake_model = brake_model
        self.action_scale = action_scale
        self.target_speed = target_speed
        self.stable_steps = stable_steps
        self.max_prefix_steps = max_prefix_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        consecutive = 0
        maximum_orientation = float(info["orientation_error"])
        contact_steps = 0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            action, _ = self.brake_model.predict(obs, deterministic=True)
            action = self.action_scale * np.asarray(action, dtype=np.float32)
            obs, _, terminated, truncated, info = self.env.step(action)
            maximum_orientation = max(
                maximum_orientation,
                float(info["orientation_error"]),
                abs(float(info.get("unwrapped_orientation_error", 0.0))),
            )
            if info.get("upper_body_grounded", False):
                contact_steps += 1
            ready = bool(
                info["phase"] == "landing"
                and float(info["com_speed"]) <= self.target_speed
                and float(info["restart_space_margin"]) >= 0.0
                and not info.get("upper_body_grounded", False)
            )
            consecutive = consecutive + 1 if ready else 0
            if consecutive >= self.stable_steps:
                break
            if terminated or truncated:
                raise RuntimeError("制動教師が扶正開始状態に達する前に終了した。")
        else:
            raise RuntimeError("制動教師が最大前綴歩数内に低速状態へ到達しなかった。")
        info = dict(info)
        info["brake_prefix_steps"] = prefix_steps
        info["brake_prefix_maximum_orientation"] = maximum_orientation
        info["brake_prefix_contact_steps"] = contact_steps
        return obs, info


class StableRightingPrefixWrapper(gym.Wrapper):
    """扶正方策で安定着地まで進め、再前進学習だけを切り出す。"""

    def __init__(self, env, righting_model, action_scale: float, max_prefix_steps: int):
        super().__init__(env)
        self.righting_model = righting_model
        self.action_scale = action_scale
        self.max_prefix_steps = max_prefix_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        maximum_orientation = float(info["orientation_error"])
        contact_steps = 0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if info["phase"] == "restart":
                break
            action, _ = self.righting_model.predict(obs, deterministic=True)
            action = self.action_scale * np.asarray(action, dtype=np.float32)
            obs, _, terminated, truncated, info = self.env.step(action)
            maximum_orientation = max(
                maximum_orientation,
                float(info["orientation_error"]),
                abs(float(info.get("unwrapped_orientation_error", 0.0))),
            )
            contact_steps += int(bool(info.get("upper_body_grounded", False)))
            if terminated or truncated:
                raise RuntimeError("扶正教師が安定着地前に終了した。")
        else:
            raise RuntimeError("扶正教師が最大前綴歩数内に安定着地しなかった。")
        info = dict(info)
        info["righting_prefix_steps"] = prefix_steps - 1
        info["righting_prefix_maximum_orientation"] = maximum_orientation
        info["righting_prefix_contact_steps"] = contact_steps
        return obs, info


class TrueNoSideFallFirstObstaclePrefixWrapper(gym.Wrapper):
    """検証済み第一障害物制御鎖を実行し、第二障害物直前まで進める。"""

    def __init__(
        self,
        env,
        approach_model,
        clearance_model,
        brake_model,
        righting_model,
        restart_model,
        handoff_distance: float,
        max_prefix_steps: int,
    ):
        super().__init__(env)
        self.approach_model = approach_model
        self.clearance_model = clearance_model
        self.brake_model = brake_model
        self.righting_model = righting_model
        self.restart_model = restart_model
        self.handoff_distance = handoff_distance
        self.max_prefix_steps = max_prefix_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        landing_mode = "brake"
        brake_ready_steps = 0
        maximum_orientation = 0.0
        contact_steps = 0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if int(info["validated_obstacles"]) >= 1:
                obstacle_x = (
                    self.env.unwrapped.course.obstacles[1].start_x
                    * self.env.unwrapped.VOXEL_SIZE
                )
                if obstacle_x - float(info["x_position"]) <= self.handoff_distance:
                    break
                model = self.approach_model
                action_scale = 1.0
            elif info["phase"] == "landing":
                if landing_mode == "brake":
                    model = self.brake_model
                    action_scale = 0.55
                else:
                    model = self.righting_model
                    action_scale = 1.0
            elif info["phase"] == "restart":
                model = self.restart_model
                action_scale = 0.5
            else:
                model = (
                    self.approach_model
                    if float(info["x_position"]) < 0.95
                    else self.clearance_model
                )
                action_scale = 1.0
            action, _ = model.predict(obs, deterministic=True)
            action = action_scale * np.asarray(action, dtype=np.float32)
            obs, _, terminated, truncated, info = self.env.step(action)
            maximum_orientation = max(
                maximum_orientation,
                float(info["orientation_error"]),
                abs(float(info.get("unwrapped_orientation_error", 0.0))),
            )
            contact_steps += int(bool(info.get("upper_body_grounded", False)))
            if landing_mode == "brake" and info["phase"] == "landing":
                ready = bool(
                    float(info["com_speed"]) <= 0.10
                    and float(info["restart_space_margin"]) >= 0.0
                    and not info.get("upper_body_grounded", False)
                )
                brake_ready_steps = brake_ready_steps + 1 if ready else 0
                if brake_ready_steps >= 10:
                    landing_mode = "righting"
            if terminated or truncated:
                raise RuntimeError("第一障害物前綴が第二障害物直前まで到達できなかった。")
        else:
            raise RuntimeError("第一障害物前綴が最大歩数内に完了しなかった。")
        info = dict(info)
        info["true_noroll_prefix_steps"] = prefix_steps - 1
        info["true_noroll_prefix_maximum_orientation"] = maximum_orientation
        info["true_noroll_prefix_contact_steps"] = contact_steps
        return obs, info


class ImprovedFirstObstaclePrefixWrapper(gym.Wrapper):
    """改良済み第一障害物制御鎖を実行し、第二障害物直前へ進める。"""

    def __init__(
        self,
        env,
        approach_model,
        first_half_model,
        safe_clearance_model,
        landing_model,
        restart_model,
        handoff_distance: float,
        max_prefix_steps: int,
        landing_action_scale: float = 0.05,
        restart_action_scale: float = 0.75,
    ):
        super().__init__(env)
        self.approach_model = approach_model
        self.first_half_model = first_half_model
        self.safe_clearance_model = safe_clearance_model
        self.landing_model = landing_model
        self.restart_model = restart_model
        self.handoff_distance = handoff_distance
        self.max_prefix_steps = max_prefix_steps
        self.landing_action_scale = landing_action_scale
        self.restart_action_scale = restart_action_scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        maximum_orientation = 0.0
        contact_steps = 0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if int(info["validated_obstacles"]) >= 1:
                obstacle_x = (
                    self.env.unwrapped.course.obstacles[1].start_x
                    * self.env.unwrapped.VOXEL_SIZE
                )
                if obstacle_x - float(info["x_position"]) <= self.handoff_distance:
                    break
                model = self.approach_model
                action_scale = 1.0
            elif info["phase"] == "landing":
                model = self.landing_model
                action_scale = self.landing_action_scale
            elif info["phase"] == "restart":
                model = self.restart_model
                action_scale = self.restart_action_scale
            elif float(info["x_position"]) < 0.95:
                model = self.approach_model
                action_scale = 1.0
            elif float(info["maximum_crossed_fraction"]) < 0.5:
                model = self.first_half_model
                action_scale = 1.0
            else:
                model = self.safe_clearance_model
                action_scale = 1.0

            action, _ = model.predict(obs, deterministic=True)
            action = action_scale * np.asarray(action, dtype=np.float32)
            obs, _, terminated, truncated, info = self.env.step(action)
            if (
                float(info["maximum_crossed_fraction"]) >= 0.5
                or info["phase"] in {"landing", "restart"}
                or int(info["validated_obstacles"]) >= 1
            ):
                angle = max(
                    float(info["orientation_error"]),
                    abs(float(info.get("unwrapped_orientation_error", 0.0))),
                )
                maximum_orientation = max(maximum_orientation, angle)
                contact_steps += int(bool(info.get("upper_body_grounded", False)))
            if terminated or truncated:
                raise RuntimeError("改良第一障害物前綴が第二障害物直前まで到達できなかった。")
        else:
            raise RuntimeError("改良第一障害物前綴が最大歩数内に完了しなかった。")
        if contact_steps > 0:
            raise RuntimeError("改良第一障害物前綴で胴体接地が再発した。")
        info = dict(info)
        info["improved_prefix_steps"] = prefix_steps - 1
        info["improved_prefix_maximum_orientation"] = maximum_orientation
        info["improved_prefix_contact_steps"] = contact_steps
        return obs, info


class SecondPartialCrossingPrefixWrapper(gym.Wrapper):
    """保存済み方策で第二障害物を指定割合まで進め、その先だけを学習する。"""

    def __init__(self, env, prefix_model, target_fraction: float, max_prefix_steps: int):
        super().__init__(env)
        self.prefix_model = prefix_model
        self.target_fraction = target_fraction
        self.max_prefix_steps = max_prefix_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        maximum_orientation = float(
            info.get("second_prefix_maximum_orientation", 0.0)
        )
        contact_steps = int(info.get("second_prefix_contact_steps", 0))
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if (
                int(info["strict_clearances"]) >= 2
                or float(info["maximum_crossed_fraction"]) >= self.target_fraction
            ):
                break
            action, _ = self.prefix_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(action)
            maximum_orientation = max(
                maximum_orientation,
                float(info["orientation_error"]),
                abs(float(info.get("unwrapped_orientation_error", 0.0))),
            )
            contact_steps += int(bool(info.get("upper_body_grounded", False)))
            if terminated or truncated:
                raise RuntimeError("第二障害物部分通過前綴が目標割合前に終了した。")
        else:
            raise RuntimeError("第二障害物部分通過前綴が最大歩数内に完了しなかった。")
        info = dict(info)
        info["second_crossing_prefix_steps"] = prefix_steps - 1
        info["second_crossing_prefix_fraction"] = float(
            info["maximum_crossed_fraction"]
        )
        info["second_prefix_maximum_orientation"] = maximum_orientation
        info["second_prefix_contact_steps"] = contact_steps
        return obs, info


class FirstPartialClearancePrefixWrapper(gym.Wrapper):
    """第一障害物を指定割合まで教師で進め、着地準備を早期に学習へ渡す。"""

    def __init__(
        self,
        env,
        approach_model,
        clearance_model,
        handoff_x: float,
        target_fraction: float,
        max_prefix_steps: int,
    ):
        super().__init__(env)
        self.approach_model = approach_model
        self.clearance_model = clearance_model
        self.handoff_x = handoff_x
        self.target_fraction = target_fraction
        self.max_prefix_steps = max_prefix_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if float(info["maximum_crossed_fraction"]) >= self.target_fraction:
                break
            model = (
                self.approach_model
                if float(info["x_position"]) < self.handoff_x
                else self.clearance_model
            )
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                raise RuntimeError("第一障害物早期交接前綴が目標割合前に終了した。")
        else:
            raise RuntimeError("第一障害物早期交接前綴が最大歩数内に完了しなかった。")
        info = dict(info)
        info["first_partial_prefix_steps"] = prefix_steps - 1
        info["first_partial_prefix_fraction"] = float(
            info["maximum_crossed_fraction"]
        )
        return obs, info


class StableLandingPrefixWrapper(gym.Wrapper):
    """接近、直立通過、安定着地の三教師で再前進直前まで進める。"""

    def __init__(
        self,
        env,
        approach_model,
        clearance_model,
        landing_model,
        handoff_x: float,
        max_prefix_steps: int,
    ):
        super().__init__(env)
        self.approach_model = approach_model
        self.clearance_model = clearance_model
        self.landing_model = landing_model
        self.handoff_x = handoff_x
        self.max_prefix_steps = max_prefix_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prefix_return = 0.0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if info["phase"] == "restart":
                break
            if info["phase"] == "landing":
                model = self.landing_model
            elif float(info["x_position"]) < self.handoff_x:
                model = self.approach_model
            else:
                model = self.clearance_model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            prefix_return += float(reward)
            if terminated or truncated:
                raise RuntimeError("分层教师在稳定着地前提前结束。")
        else:
            raise RuntimeError("分层教师未能在最大前缀步数内完成稳定着地。")
        info = dict(info)
        info["prefix_steps"] = prefix_steps - 1
        info["prefix_return"] = prefix_return
        return obs, info


class LandingPhasePrefixWrapper(gym.Wrapper):
    """既に通過した障害物の着地方策を再前進段階まで実行する。"""

    def __init__(
        self,
        env,
        landing_model,
        max_prefix_steps: int,
        action_scale: float = 1.0,
    ):
        super().__init__(env)
        self.landing_model = landing_model
        self.max_prefix_steps = max_prefix_steps
        self.action_scale = action_scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prefix_return = 0.0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if info["phase"] == "restart":
                break
            action, _ = self.landing_model.predict(obs, deterministic=True)
            action = self.action_scale * np.asarray(action, dtype=np.float32)
            obs, reward, terminated, truncated, info = self.env.step(action)
            prefix_return += float(reward)
            if terminated or truncated:
                raise RuntimeError("着地教师在再前进阶段へ入る前に終了した。")
        else:
            raise RuntimeError("着地教师未能在最大前缀步数内完成稳定落地。")
        info = dict(info)
        info["prefix_steps"] = info.get("prefix_steps", 0) + prefix_steps - 1
        info["prefix_return"] = info.get("prefix_return", 0.0) + prefix_return
        return obs, info


def mask_unseen_future_obstacles(obs, env, visible_obstacle_count: int | None):
    """方策の学習時に存在しなかった将来障害物を地形観測だけから隠す。"""
    raw = env.unwrapped
    if (
        visible_obstacle_count is None
        or visible_obstacle_count >= len(raw.course.obstacles)
    ):
        return obs
    masked = np.array(obs, copy=True)
    floor_size = 2 * raw.sight_dist + 1
    floor_start = masked.size - 5 - floor_size
    floor = masked[floor_start : floor_start + floor_size]
    flat_distance = float(np.max(floor))
    positions = raw.object_pos_at_time(raw.get_time(), "robot")
    com_x_voxels = float(np.mean(positions[0])) / raw.VOXEL_SIZE
    sensor_centers = com_x_voxels + np.arange(-raw.sight_dist, raw.sight_dist + 1)
    for obstacle in raw.course.obstacles[visible_obstacle_count:]:
        point_left = float(obstacle.start_x)
        point_right = float(obstacle.end_x + 1)
        affected = (
            (sensor_centers > point_left - 0.5)
            & (sensor_centers < point_right + 0.5)
        )
        floor[affected] = flat_distance
    return masked


class ValidatedFourStagePrefixWrapper(gym.Wrapper):
    """四方策で指定数の障害物を検証し、次障害物直前の状態を返す。"""

    def __init__(
        self,
        env,
        approach_model,
        clearance_model,
        landing_model,
        restart_model,
        target_validated: int,
        handoff_distance: float,
        max_prefix_steps: int,
    ):
        super().__init__(env)
        self.approach_model = approach_model
        self.clearance_model = clearance_model
        self.landing_model = landing_model
        self.restart_model = restart_model
        self.target_validated = target_validated
        self.handoff_distance = handoff_distance
        self.max_prefix_steps = max_prefix_steps

    def _select_model(self, info):
        """現在段階と対象障害物までの相対距離から教師を選択する。"""
        raw = self.env.unwrapped
        if info["phase"] == "landing":
            return self.landing_model
        if info["phase"] == "restart":
            return self.restart_model
        obstacle = raw.course.obstacles[raw._active_obstacle]
        obstacle_x = obstacle.start_x * raw.VOXEL_SIZE
        return (
            self.approach_model
            if obstacle_x - float(info["x_position"]) > self.handoff_distance
            else self.clearance_model
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prefix_return = 0.0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if int(info["validated_obstacles"]) >= self.target_validated:
                break
            model = self._select_model(info)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            prefix_return += float(reward)
            if terminated or truncated:
                raise RuntimeError("四阶段教师在完成前缀障碍物前提前结束。")
        else:
            raise RuntimeError("四阶段教师未能在最大前缀步数内完成指定障碍物。")
        info = dict(info)
        info["prefix_steps"] = prefix_steps - 1
        info["prefix_return"] = prefix_return
        return obs, info


class ValidatedPolicyListPrefixWrapper(gym.Wrapper):
    """障害物ごとの三方策で検証済み前綴を正確に再生する。"""

    def __init__(
        self,
        env,
        approach_model,
        obstacle_policies,
        target_validated: int,
        handoff_distance: float,
        max_prefix_steps: int,
    ):
        super().__init__(env)
        self.approach_model = approach_model
        self.obstacle_policies = obstacle_policies
        self.target_validated = target_validated
        self.handoff_distance = handoff_distance
        self.max_prefix_steps = max_prefix_steps
        if len(obstacle_policies) < target_validated:
            raise ValueError("前綴方策の数が検証済み障害物数より少ない。")

    def _select_model(self, info):
        """現在の障害物番号と状態段階に対応する方策を返す。"""
        raw = self.env.unwrapped
        index = int(info["active_obstacle"])
        policy = self.obstacle_policies[index]
        if info["phase"] == "landing":
            return policy["landing"]
        if info["phase"] == "restart":
            return policy["restart"]
        obstacle_x = raw.course.obstacles[index].start_x * raw.VOXEL_SIZE
        approach_model = policy.get("approach", self.approach_model)
        return (
            approach_model
            if obstacle_x - float(info["x_position"]) > self.handoff_distance
            else policy["clearance"]
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prefix_return = 0.0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            if int(info["validated_obstacles"]) >= self.target_validated:
                break
            policy = self.obstacle_policies[int(info["active_obstacle"])]
            model = self._select_model(info)
            model_obs = mask_unseen_future_obstacles(
                obs,
                self.env,
                policy.get("visible_obstacle_count"),
            )
            action, _ = model.predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            prefix_return += float(reward)
            if terminated or truncated:
                raise RuntimeError("障害物別教師が前綴完了前に終了した。")
        else:
            raise RuntimeError("障害物別教師が最大前綴歩数内に完了しなかった。")
        info = dict(info)
        info["prefix_steps"] = prefix_steps - 1
        info["prefix_return"] = prefix_return
        return obs, info


class PostPrefixLandingThresholdWrapper(gym.Wrapper):
    """教師前綴の完了後だけ着地訓練閾値を切り替える。"""

    def __init__(self, env, landing_angle_limit: float, landing_speed_limit: float):
        super().__init__(env)
        self.landing_angle_limit = landing_angle_limit
        self.landing_speed_limit = landing_speed_limit

    def reset(self, **kwargs):
        raw = self.env.unwrapped
        raw.landing_angle_limit = LANDING_ANGLE_LIMIT
        raw.landing_speed_limit = LANDING_SPEED_LIMIT
        obs, info = self.env.reset(**kwargs)
        raw.landing_angle_limit = self.landing_angle_limit
        raw.landing_speed_limit = self.landing_speed_limit
        info = dict(info)
        info["landing_angle_limit"] = self.landing_angle_limit
        info["landing_speed_limit"] = self.landing_speed_limit
        return obs, info


class ValidatedObstacleGoalWrapper(gym.Wrapper):
    """指定数の障害物を完全検証した時点で中間課題を終了する。"""

    def __init__(self, env, target_validated: int, success_bonus: float = 100.0):
        super().__init__(env)
        self.target_validated = target_validated
        self.success_bonus = success_bonus

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["subgoal_success"] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reached = int(info["validated_obstacles"]) >= self.target_validated
        info = dict(info)
        info["subgoal_success"] = reached
        if reached:
            reward += self.success_bonus
            terminated = True
        return obs, reward, terminated, truncated, info


class StrictClearanceGoalWrapper(gym.Wrapper):
    """指定数の物理的完全通過を達成した瞬間に中間課題を終了する。"""

    def __init__(self, env, target_clearances: int, success_bonus: float = 100.0):
        super().__init__(env)
        self.target_clearances = target_clearances
        self.success_bonus = success_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reached = int(info["strict_clearances"]) >= self.target_clearances
        info = dict(info)
        info["clearance_subgoal_success"] = reached
        if reached:
            reward += self.success_bonus
            terminated = True
        return obs, reward, terminated, truncated, info


class CrossingFractionGoalWrapper(gym.Wrapper):
    """次障害物後端を越えた身体点の割合で短い中間課題を終了する。"""

    def __init__(
        self,
        env,
        target_clearances: int,
        target_fraction: float,
        success_bonus: float = 100.0,
        progress_bonus_scale: float = 0.0,
    ):
        super().__init__(env)
        self.target_clearances = target_clearances
        self.target_fraction = target_fraction
        self.success_bonus = success_bonus
        self.progress_bonus_scale = progress_bonus_scale
        self.previous_fraction = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_fraction = float(info.get("maximum_crossed_fraction", 0.0))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        clearance_count = int(info["strict_clearances"])
        fraction = float(info["maximum_crossed_fraction"])
        fraction_gain = max(0.0, fraction - self.previous_fraction)
        reward += self.progress_bonus_scale * fraction_gain
        reached = clearance_count >= self.target_clearances or (
            clearance_count == self.target_clearances - 1
            and fraction >= self.target_fraction
        )
        info = dict(info)
        info["crossing_fraction_subgoal_success"] = reached
        if reached:
            reward += self.success_bonus
            terminated = True
        self.previous_fraction = fraction
        return obs, reward, terminated, truncated, info


class StableLandingGoalWrapper(gym.Wrapper):
    """指定数の安定着地を達成した瞬間に中間課題を終了する。"""

    def __init__(self, env, target_landings: int, success_bonus: float = 100.0):
        super().__init__(env)
        self.target_landings = target_landings
        self.success_bonus = success_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reached = int(info["stable_landings"]) >= self.target_landings
        info = dict(info)
        info["landing_subgoal_success"] = reached
        if reached:
            reward += self.success_bonus
            terminated = True
        return obs, reward, terminated, truncated, info


class OrientationSafetyWrapper(gym.Wrapper):
    """学習対象区間の傾斜と上半身接地を制限して側倒を防ぐ。"""

    def __init__(
        self,
        env,
        max_orientation: float,
        preferred_orientation: float,
        failure_penalty: float = 300.0,
        forbid_upper_body_contact: bool = True,
    ):
        super().__init__(env)
        self.max_orientation = max_orientation
        self.preferred_orientation = preferred_orientation
        self.failure_penalty = failure_penalty
        self.forbid_upper_body_contact = forbid_upper_body_contact
        self.maximum_orientation = 0.0
        self.previous_orientation = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        angle = max(
            float(info["orientation_error"]),
            float(info.get("unwrapped_orientation_error", 0.0)),
        )
        self.maximum_orientation = angle
        self.previous_orientation = angle
        info = dict(info)
        info["maximum_orientation_error"] = angle
        info["orientation_limit"] = self.max_orientation
        info["no_rollout"] = angle < self.max_orientation
        info["no_side_fall"] = bool(
            angle < self.max_orientation
            and not info.get("upper_body_grounded", False)
        )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        angle = max(
            float(info["orientation_error"]),
            float(info.get("unwrapped_orientation_error", 0.0)),
        )
        self.maximum_orientation = max(self.maximum_orientation, angle)
        excess = max(0.0, angle - self.preferred_orientation)
        reward -= 2.5 * (excess / max(self.preferred_orientation, 1e-6)) ** 2
        improvement = self.previous_orientation - angle
        if info["phase"] == "landing":
            reward += 5.0 * max(0.0, improvement)
            reward -= 3.0 * max(0.0, -improvement)
        if info.get("new_stable_landing", False):
            # 安定着地だけでなく、着地までの最大傾斜が小さい軌跡を直接選ぶ。
            span = max(self.max_orientation - self.preferred_orientation, 1e-6)
            peak_quality = np.clip(
                (self.max_orientation - self.maximum_orientation) / span,
                0.0,
                1.0,
            )
            reward += 200.0 + 800.0 * float(peak_quality)
        upper_body_grounded = bool(info.get("upper_body_grounded", False))
        unsafe_orientation = self.maximum_orientation >= self.max_orientation
        unsafe_contact = self.forbid_upper_body_contact and upper_body_grounded
        unsafe = unsafe_orientation or unsafe_contact
        info = dict(info)
        info["maximum_orientation_error"] = self.maximum_orientation
        info["orientation_limit"] = self.max_orientation
        info["no_rollout"] = not unsafe
        info["no_side_fall"] = not unsafe
        if unsafe:
            reward -= self.failure_penalty
            terminated = True
            info["failure_reason"] = (
                "upper_body_ground_contact"
                if unsafe_contact
                else "orientation_limit_exceeded"
            )
            info["is_success"] = False
        self.previous_orientation = angle
        return obs, reward, terminated, truncated, info


class NeutralActionScaleWrapper(gym.ActionWrapper):
    """学習方策の動作を中立点の周囲へ縮小して安全な残差制御にする。"""

    def __init__(self, env, scale: float):
        super().__init__(env)
        if not 0.0 < scale <= 1.0:
            raise ValueError("動作縮小率は0より大きく1以下でなければならない。")
        self.scale = scale

    def action(self, action):
        return self.scale * np.asarray(action, dtype=np.float32)


def make_env(
    body: np.ndarray,
    level: int,
    max_steps: int,
    monitor_path: Path | None = None,
    monitor_override: bool = True,
    render_mode: str | None = None,
    render_options: dict | None = None,
    landing_angle_limit: float = LANDING_ANGLE_LIMIT,
    landing_speed_limit: float = LANDING_SPEED_LIMIT,
):
    """行動正規化と時間制限を備えた一つの環境を構築する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
        landing_angle_limit=landing_angle_limit,
        landing_speed_limit=landing_speed_limit,
        render_mode=render_mode,
        render_options=render_options,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = ActionRescaleWrapper(env)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
            override_existing=monitor_override,
        )
    return env


def make_recovery_env(
    body: np.ndarray,
    level: int,
    teacher_model,
    target_clearances: int,
    agent_max_steps: int,
    prefix_max_steps: int,
    handoff_x: float | None = None,
    target_validated: int | None = None,
    landing_angle_limit: float = LANDING_ANGLE_LIMIT,
    monitor_path: Path | None = None,
):
    """教師の途中状態から通過・着地・再前進をまとめて学ぶ環境を返す。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
        landing_angle_limit=landing_angle_limit,
    )
    env = ActionRescaleWrapper(env)
    env = TeacherClearancePrefixWrapper(
        env,
        teacher_model=teacher_model,
        target_clearances=target_clearances,
        max_prefix_steps=prefix_max_steps,
        handoff_x=handoff_x,
    )
    if target_validated is not None:
        env = ValidatedObstacleGoalWrapper(env, target_validated=target_validated)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
        )
    return env


def make_upright_clearance_env(
    body: np.ndarray,
    level: int,
    teacher_model,
    handoff_x: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    max_clearance_speed: float | None = None,
    max_clearance_angle: float = LANDING_ANGLE_LIMIT,
    monitor_path: Path | None = None,
):
    """教師交接後に直立した完全通過だけを学習する短区間環境を返す。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = TeacherClearancePrefixWrapper(
        env,
        teacher_model=teacher_model,
        target_clearances=1,
        max_prefix_steps=prefix_max_steps,
        handoff_x=handoff_x,
    )
    env = UprightClearanceGoalWrapper(
        env,
        max_clearance_speed=max_clearance_speed,
        max_clearance_angle=max_clearance_angle,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "upright_clearance_success",
                "clearance_angle",
                "clearance_speed",
                "max_x_position",
            ),
        )
    return env


def make_true_noroll_second_clearance_env(
    body: np.ndarray,
    approach_model,
    first_clearance_model,
    first_brake_model,
    first_righting_model,
    first_restart_model,
    agent_max_steps: int,
    prefix_max_steps: int,
    handoff_distance: float = 0.25,
    max_clearance_speed: float | None = None,
    max_clearance_angle: float = LANDING_ANGLE_LIMIT,
    monitor_path: Path | None = None,
):
    """無側倒の第一障害物前綴後から第二障害物通過だけを学習する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=2,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = TrueNoSideFallFirstObstaclePrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=first_clearance_model,
        brake_model=first_brake_model,
        righting_model=first_righting_model,
        restart_model=first_restart_model,
        handoff_distance=handoff_distance,
        max_prefix_steps=prefix_max_steps,
    )
    env = UprightClearanceGoalWrapper(
        env,
        max_clearance_speed=max_clearance_speed,
        max_clearance_angle=max_clearance_angle,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "upright_clearance_success",
                "clearance_angle",
                "clearance_speed",
                "max_x_position",
            ),
        )
    return env


def make_first_early_landing_env(
    body: np.ndarray,
    approach_model,
    clearance_model,
    prefix_fraction: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    handoff_x: float = 0.95,
    max_orientation: float | None = None,
    preferred_orientation: float = LANDING_ANGLE_LIMIT,
    agent_action_scale: float = 1.0,
    target_clearance_only: bool = False,
    render_mode: str | None = None,
    monitor_path: Path | None = None,
):
    """第一障害物の途中から胴体非接地の完全通過と着地を共同学習する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=2,
        connections=get_full_connectivity(body),
        render_mode=render_mode,
    )
    env = ActionRescaleWrapper(env)
    env = FirstPartialClearancePrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=clearance_model,
        handoff_x=handoff_x,
        target_fraction=prefix_fraction,
        max_prefix_steps=prefix_max_steps,
    )
    if agent_action_scale < 1.0:
        env = NeutralActionScaleWrapper(env, scale=agent_action_scale)
    if target_clearance_only:
        # 成功判定の外側に安全制約を置き、同一フレームの胴体接地も失敗で上書きする。
        env = StrictClearanceGoalWrapper(env, target_clearances=1)
        if max_orientation is not None:
            env = OrientationSafetyWrapper(
                env,
                max_orientation=max_orientation,
                preferred_orientation=preferred_orientation,
                forbid_upper_body_contact=True,
            )
    else:
        if max_orientation is not None:
            env = OrientationSafetyWrapper(
                env,
                max_orientation=max_orientation,
                preferred_orientation=preferred_orientation,
                forbid_upper_body_contact=True,
            )
        env = StableLandingGoalWrapper(env, target_landings=1)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
        )
    return env


def make_first_safe_landing_env(
    body: np.ndarray,
    approach_model,
    clearance_model,
    safe_clearance_model,
    prefix_fraction: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    handoff_x: float = 0.95,
    max_orientation: float | None = None,
    preferred_orientation: float = LANDING_ANGLE_LIMIT,
    agent_action_scale: float = 1.0,
    render_mode: str | None = None,
    monitor_path: Path | None = None,
):
    """胴体非接地の完全通過直後から第一障害物の着地だけを学習する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=2,
        connections=get_full_connectivity(body),
        render_mode=render_mode,
    )
    env = ActionRescaleWrapper(env)
    env = FirstPartialClearancePrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=clearance_model,
        handoff_x=handoff_x,
        target_fraction=prefix_fraction,
        max_prefix_steps=prefix_max_steps,
    )
    env = TeacherClearancePrefixWrapper(
        env,
        teacher_model=safe_clearance_model,
        target_clearances=1,
        max_prefix_steps=prefix_max_steps,
    )
    if agent_action_scale < 1.0:
        env = NeutralActionScaleWrapper(env, scale=agent_action_scale)
    if max_orientation is not None:
        env = OrientationSafetyWrapper(
            env,
            max_orientation=max_orientation,
            preferred_orientation=preferred_orientation,
            forbid_upper_body_contact=True,
        )
    env = StableLandingGoalWrapper(env, target_landings=1)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
        )
    return env


def make_first_safe_restart_env(
    body: np.ndarray,
    approach_model,
    clearance_model,
    safe_clearance_model,
    landing_model,
    prefix_fraction: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    landing_action_scale: float = 0.05,
    agent_action_scale: float = 1.0,
    max_orientation: float | None = None,
    preferred_orientation: float = LANDING_ANGLE_LIMIT,
    render_mode: str | None = None,
):
    """安全着地後から第一障害物の再前進だけを評価または学習する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=2,
        connections=get_full_connectivity(body),
        render_mode=render_mode,
    )
    env = ActionRescaleWrapper(env)
    env = FirstPartialClearancePrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=clearance_model,
        handoff_x=0.95,
        target_fraction=prefix_fraction,
        max_prefix_steps=prefix_max_steps,
    )
    env = TeacherClearancePrefixWrapper(
        env,
        teacher_model=safe_clearance_model,
        target_clearances=1,
        max_prefix_steps=prefix_max_steps,
    )
    env = LandingPhasePrefixWrapper(
        env,
        landing_model=landing_model,
        max_prefix_steps=prefix_max_steps,
        action_scale=landing_action_scale,
    )
    if agent_action_scale < 1.0:
        env = NeutralActionScaleWrapper(env, scale=agent_action_scale)
    if max_orientation is not None:
        env = OrientationSafetyWrapper(
            env,
            max_orientation=max_orientation,
            preferred_orientation=preferred_orientation,
            forbid_upper_body_contact=True,
        )
    env = ValidatedObstacleGoalWrapper(env, target_validated=1)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    return env


def make_improved_second_crossing_env(
    body: np.ndarray,
    approach_model,
    first_half_model,
    first_safe_clearance_model,
    first_landing_model,
    first_restart_model,
    target_fraction: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    second_handoff_distance: float = 0.25,
    first_landing_action_scale: float = 0.05,
    first_restart_action_scale: float = 0.75,
    second_prefix_model=None,
    second_prefix_fraction: float | None = None,
    second_prefix_model_2=None,
    second_prefix_fraction_2: float | None = None,
    max_orientation: float = math.radians(65.0),
    preferred_orientation: float = math.radians(20.0),
    monitor_path: Path | None = None,
):
    """改良第一障害物前綴後に第二障害物の部分通過を学習する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=2,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = ImprovedFirstObstaclePrefixWrapper(
        env,
        approach_model=approach_model,
        first_half_model=first_half_model,
        safe_clearance_model=first_safe_clearance_model,
        landing_model=first_landing_model,
        restart_model=first_restart_model,
        handoff_distance=second_handoff_distance,
        max_prefix_steps=prefix_max_steps,
        landing_action_scale=first_landing_action_scale,
        restart_action_scale=first_restart_action_scale,
    )
    if second_prefix_model is not None:
        env = SecondPartialCrossingPrefixWrapper(
            env,
            prefix_model=second_prefix_model,
            target_fraction=float(second_prefix_fraction),
            max_prefix_steps=prefix_max_steps,
        )
    if second_prefix_model_2 is not None:
        env = SecondPartialCrossingPrefixWrapper(
            env,
            prefix_model=second_prefix_model_2,
            target_fraction=float(second_prefix_fraction_2),
            max_prefix_steps=prefix_max_steps,
        )
    env = CrossingFractionGoalWrapper(
        env,
        target_clearances=2,
        target_fraction=target_fraction,
        progress_bonus_scale=400.0,
    )
    # 部分通過の成功と同一フレームで胴体が接地しても、安全失敗を優先する。
    env = OrientationSafetyWrapper(
        env,
        max_orientation=max_orientation,
        preferred_orientation=preferred_orientation,
        forbid_upper_body_contact=True,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "maximum_crossed_fraction",
                "crossing_fraction_subgoal_success",
                "max_x_position",
            ),
        )
    return env


def make_true_noroll_second_crossing_env(
    body: np.ndarray,
    approach_model,
    first_clearance_model,
    first_brake_model,
    first_righting_model,
    first_restart_model,
    target_fraction: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    handoff_distance: float = 0.25,
    second_prefix_model=None,
    second_prefix_fraction: float | None = None,
    second_prefix_model_2=None,
    second_prefix_fraction_2: float | None = None,
    monitor_path: Path | None = None,
):
    """第一障害物の無側倒前綴後に第二障害物の部分通過だけを学習する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=2,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = TrueNoSideFallFirstObstaclePrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=first_clearance_model,
        brake_model=first_brake_model,
        righting_model=first_righting_model,
        restart_model=first_restart_model,
        handoff_distance=handoff_distance,
        max_prefix_steps=prefix_max_steps,
    )
    if second_prefix_model is not None:
        if second_prefix_fraction is None:
            raise ValueError("第二障害物前綴方策には到達割合が必要です。")
        env = SecondPartialCrossingPrefixWrapper(
            env,
            prefix_model=second_prefix_model,
            target_fraction=second_prefix_fraction,
            max_prefix_steps=prefix_max_steps,
        )
    if second_prefix_model_2 is not None:
        if second_prefix_fraction_2 is None:
            raise ValueError("第二障害物の第二前綴方策には到達割合が必要です。")
        env = SecondPartialCrossingPrefixWrapper(
            env,
            prefix_model=second_prefix_model_2,
            target_fraction=second_prefix_fraction_2,
            max_prefix_steps=prefix_max_steps,
        )
    env = CrossingFractionGoalWrapper(
        env,
        target_clearances=2,
        target_fraction=target_fraction,
        progress_bonus_scale=400.0,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "maximum_crossed_fraction",
                "crossing_fraction_subgoal_success",
                "max_x_position",
            ),
        )
    return env


def make_true_noroll_second_landing_env(
    body: np.ndarray,
    approach_model,
    first_clearance_model,
    first_brake_model,
    first_righting_model,
    first_restart_model,
    second_prefix_model,
    second_prefix_model_2,
    second_final_clearance_model,
    agent_max_steps: int,
    prefix_max_steps: int,
    target_stable_landings: int = 2,
    max_orientation: float | None = None,
    preferred_orientation: float = LANDING_ANGLE_LIMIT,
    forbid_upper_body_contact: bool = True,
    agent_action_scale: float = 1.0,
    monitor_path: Path | None = None,
):
    """第二障害物の完全通過直後から安全着地だけを学習する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=2,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = TrueNoSideFallFirstObstaclePrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=first_clearance_model,
        brake_model=first_brake_model,
        righting_model=first_righting_model,
        restart_model=first_restart_model,
        handoff_distance=0.25,
        max_prefix_steps=prefix_max_steps,
    )
    env = SecondPartialCrossingPrefixWrapper(
        env,
        prefix_model=second_prefix_model,
        target_fraction=1.0 / 3.0,
        max_prefix_steps=prefix_max_steps,
    )
    env = SecondPartialCrossingPrefixWrapper(
        env,
        prefix_model=second_prefix_model_2,
        target_fraction=0.5,
        max_prefix_steps=prefix_max_steps,
    )
    env = TeacherClearancePrefixWrapper(
        env,
        teacher_model=second_final_clearance_model,
        target_clearances=2,
        max_prefix_steps=prefix_max_steps,
    )
    if agent_action_scale < 1.0:
        env = NeutralActionScaleWrapper(env, scale=agent_action_scale)
    if max_orientation is not None:
        env = OrientationSafetyWrapper(
            env,
            max_orientation=max_orientation,
            preferred_orientation=preferred_orientation,
            forbid_upper_body_contact=forbid_upper_body_contact,
        )
    env = StableLandingGoalWrapper(
        env,
        target_landings=target_stable_landings,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
        )
    return env


def make_true_noroll_second_restart_env(
    body: np.ndarray,
    approach_model,
    first_clearance_model,
    first_brake_model,
    first_righting_model,
    first_restart_model,
    second_prefix_model,
    second_prefix_model_2,
    second_final_clearance_model,
    second_landing_model,
    agent_max_steps: int,
    prefix_max_steps: int,
    second_landing_action_scale: float = 0.1,
    agent_action_scale: float = 1.0,
    max_orientation: float | None = None,
    preferred_orientation: float = LANDING_ANGLE_LIMIT,
    forbid_upper_body_contact: bool = True,
    monitor_path: Path | None = None,
):
    """第二障害物の安全着地後から最終再前進だけを学習する。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=2,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = TrueNoSideFallFirstObstaclePrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=first_clearance_model,
        brake_model=first_brake_model,
        righting_model=first_righting_model,
        restart_model=first_restart_model,
        handoff_distance=0.25,
        max_prefix_steps=prefix_max_steps,
    )
    env = SecondPartialCrossingPrefixWrapper(
        env,
        prefix_model=second_prefix_model,
        target_fraction=1.0 / 3.0,
        max_prefix_steps=prefix_max_steps,
    )
    env = SecondPartialCrossingPrefixWrapper(
        env,
        prefix_model=second_prefix_model_2,
        target_fraction=0.5,
        max_prefix_steps=prefix_max_steps,
    )
    env = TeacherClearancePrefixWrapper(
        env,
        teacher_model=second_final_clearance_model,
        target_clearances=2,
        max_prefix_steps=prefix_max_steps,
    )
    env = LandingPhasePrefixWrapper(
        env,
        landing_model=second_landing_model,
        max_prefix_steps=prefix_max_steps,
        action_scale=second_landing_action_scale,
    )
    if agent_action_scale < 1.0:
        env = NeutralActionScaleWrapper(env, scale=agent_action_scale)
    if max_orientation is not None:
        env = OrientationSafetyWrapper(
            env,
            max_orientation=max_orientation,
            preferred_orientation=preferred_orientation,
            forbid_upper_body_contact=forbid_upper_body_contact,
        )
    env = ValidatedObstacleGoalWrapper(env, target_validated=2)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
        )
    return env


def make_three_stage_recovery_env(
    body: np.ndarray,
    level: int,
    approach_model,
    clearance_model,
    handoff_x: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    target_validated: int | None = None,
    target_stable_landings: int | None = None,
    max_orientation: float | None = None,
    preferred_orientation: float = LANDING_ANGLE_LIMIT,
    forbid_upper_body_contact: bool = True,
    agent_action_scale: float = 1.0,
    brake_model=None,
    brake_action_scale: float = 1.0,
    brake_target_speed: float = 0.10,
    brake_stable_steps: int = 10,
    righting_prefix_model=None,
    righting_prefix_action_scale: float = 1.0,
    monitor_path: Path | None = None,
):
    """二教師の直立通過後から着地回復だけを学習する環境を返す。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = HierarchicalClearancePrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=clearance_model,
        handoff_x=handoff_x,
        max_prefix_steps=prefix_max_steps,
    )
    if brake_model is not None:
        env = SafeBrakePrefixWrapper(
            env,
            brake_model=brake_model,
            action_scale=brake_action_scale,
            target_speed=brake_target_speed,
            stable_steps=brake_stable_steps,
            max_prefix_steps=prefix_max_steps,
        )
    if righting_prefix_model is not None:
        env = StableRightingPrefixWrapper(
            env,
            righting_model=righting_prefix_model,
            action_scale=righting_prefix_action_scale,
            max_prefix_steps=prefix_max_steps,
        )
    if agent_action_scale < 1.0:
        env = NeutralActionScaleWrapper(env, scale=agent_action_scale)
    if max_orientation is not None:
        env = OrientationSafetyWrapper(
            env,
            max_orientation=max_orientation,
            preferred_orientation=preferred_orientation,
            forbid_upper_body_contact=forbid_upper_body_contact,
        )
    if target_stable_landings is not None:
        env = StableLandingGoalWrapper(
            env,
            target_landings=target_stable_landings,
        )
    if target_validated is not None:
        env = ValidatedObstacleGoalWrapper(env, target_validated=target_validated)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
        )
    return env


def make_restart_env(
    body: np.ndarray,
    level: int,
    approach_model,
    clearance_model,
    landing_model,
    handoff_x: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    target_validated: int | None = None,
    monitor_path: Path | None = None,
):
    """安定着地後の再前進だけを学習歩数として数える環境を返す。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = StableLandingPrefixWrapper(
        env,
        approach_model=approach_model,
        clearance_model=clearance_model,
        landing_model=landing_model,
        handoff_x=handoff_x,
        max_prefix_steps=prefix_max_steps,
    )
    if target_validated is not None:
        env = ValidatedObstacleGoalWrapper(env, target_validated=target_validated)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
        )
    return env


def make_next_obstacle_env(
    body: np.ndarray,
    level: int,
    approach_model,
    clearance_model,
    landing_model,
    restart_model,
    prefix_validated: int,
    target_validated: int | None,
    target_clearances: int | None,
    target_stable_landings: int | None,
    upright_clearance_speed: float | None,
    target_crossing_fraction: float | None,
    handoff_distance: float,
    agent_max_steps: int,
    prefix_max_steps: int,
    landing_angle_limit: float = LANDING_ANGLE_LIMIT,
    landing_speed_limit: float = 0.15,
    next_clearance_model=None,
    next_landing_model=None,
    prefix_clearances: int | None = None,
    upright_clearance_angle: float = LANDING_ANGLE_LIMIT,
    prefix_obstacle_policies=None,
    max_orientation: float | None = None,
    preferred_orientation: float = LANDING_ANGLE_LIMIT,
    forbid_upper_body_contact: bool = True,
    monitor_path: Path | None = None,
):
    """検証済み障害物の直後から次の一個だけを学習する環境を返す。"""
    env = LongLeggedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    if prefix_obstacle_policies is None:
        env = ValidatedFourStagePrefixWrapper(
            env,
            approach_model=approach_model,
            clearance_model=clearance_model,
            landing_model=landing_model,
            restart_model=restart_model,
            target_validated=prefix_validated,
            handoff_distance=handoff_distance,
            max_prefix_steps=prefix_max_steps,
        )
    else:
        env = ValidatedPolicyListPrefixWrapper(
            env,
            approach_model=approach_model,
            obstacle_policies=prefix_obstacle_policies,
            target_validated=prefix_validated,
            handoff_distance=handoff_distance,
            max_prefix_steps=prefix_max_steps,
        )
    env = PostPrefixLandingThresholdWrapper(
        env,
        landing_angle_limit=landing_angle_limit,
        landing_speed_limit=landing_speed_limit,
    )
    if next_clearance_model is not None:
        if prefix_clearances is None:
            raise ValueError("次障碍越障教师需要提供目标物理越障数。")
        env = TeacherClearancePrefixWrapper(
            env,
            teacher_model=next_clearance_model,
            target_clearances=prefix_clearances,
            max_prefix_steps=prefix_max_steps,
        )
    if next_landing_model is not None:
        env = LandingPhasePrefixWrapper(
            env,
            landing_model=next_landing_model,
            max_prefix_steps=prefix_max_steps,
        )
    if max_orientation is not None:
        env = OrientationSafetyWrapper(
            env,
            max_orientation=max_orientation,
            preferred_orientation=preferred_orientation,
            forbid_upper_body_contact=forbid_upper_body_contact,
        )
    target_count = sum(
        value is not None
        for value in (
            target_validated,
            target_clearances,
            target_stable_landings,
            upright_clearance_speed,
            target_crossing_fraction,
        )
    )
    if target_count != 1:
        raise ValueError("验证、物理越障、稳定落地和直立越障目标必须且只能设置一个。")
    if upright_clearance_speed is not None:
        env = UprightClearanceGoalWrapper(
            env,
            max_clearance_speed=upright_clearance_speed,
            max_clearance_angle=upright_clearance_angle,
        )
    elif target_crossing_fraction is not None:
        env = CrossingFractionGoalWrapper(
            env,
            target_clearances=prefix_validated + 1,
            target_fraction=target_crossing_fraction,
        )
    elif target_clearances is not None:
        env = StrictClearanceGoalWrapper(env, target_clearances=target_clearances)
    elif target_stable_landings is not None:
        env = StableLandingGoalWrapper(env, target_landings=target_stable_landings)
    else:
        env = ValidatedObstacleGoalWrapper(env, target_validated=target_validated)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "strict_clearances",
                "stable_landings",
                "restart_successes",
                "validated_obstacles",
                "is_success",
                "max_x_position",
            ),
        )
    return env


def write_json(path: Path, data: dict):
    """UTF-8のJSONを読みやすい形式で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_evaluation(path: Path, row: dict):
    """評価行を固定列順のCSVへ追記する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in EVALUATION_FIELDS})


def read_evaluations(path: Path) -> list[dict]:
    """既存評価CSVを数値辞書のリストとして読み込む。"""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key: int(float(value)) if key == "timesteps" else float(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def summarize_episodes(episodes: list[dict]) -> dict:
    """複数エピソードを厳格な学習判定指標へ集約する。"""
    def values(key: str) -> np.ndarray:
        return np.asarray([episode[key] for episode in episodes], dtype=float)

    returns = values("return")
    steps = values("steps")
    displacement = values("displacement")
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps)),
        "mean_displacement": float(np.mean(displacement)),
        "mean_speed": float(np.mean(displacement / np.maximum(steps, 1.0))),
        "mean_final_x": float(np.mean(values("final_x"))),
        "mean_max_x": float(np.mean(values("max_x"))),
        "mean_strict_clearances": float(np.mean(values("strict_clearances"))),
        "mean_stable_landings": float(np.mean(values("stable_landings"))),
        "mean_restart_successes": float(np.mean(values("restart_successes"))),
        "mean_validated_obstacles": float(np.mean(values("validated_obstacles"))),
        "mean_obstacle_fraction": float(np.mean(values("obstacle_fraction"))),
        "success_rate": float(np.mean(values("success"))),
    }


def evaluate_controller(
    body: np.ndarray,
    level: int,
    controller: Callable,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict:
    """一つの制御器を独立エピソードで評価して共通指標を返す。"""
    last_error = None
    for attempt in range(2):
        env = None
        try:
            env = make_env(body, level, max_steps)
            results = []
            for episode_number in range(episodes):
                obs, reset_info = env.reset(seed=seed + episode_number)
                total_reward = 0.0
                final_info = reset_info
                completed_steps = 0
                for completed_steps in range(1, max_steps + 1):
                    action = controller(obs, env)
                    obs, reward, terminated, truncated, final_info = env.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                results.append(
                    {
                        "return": total_reward,
                        "steps": completed_steps,
                        "displacement": final_info["forward_displacement"],
                        "final_x": final_info["x_position"],
                        "max_x": final_info["max_x_position"],
                        "strict_clearances": final_info["strict_clearances"],
                        "stable_landings": final_info["stable_landings"],
                        "restart_successes": final_info["restart_successes"],
                        "validated_obstacles": final_info["validated_obstacles"],
                        "obstacle_fraction": final_info["obstacle_fraction"],
                        "success": float(final_info["is_success"]),
                    }
                )
            return summarize_episodes(results)
        except Exception as exc:
            last_error = exc
            if "invalid vector<bool> subscript" not in str(exc) or attempt:
                raise
            gc.collect()
        finally:
            if env is not None:
                env.close()
    raise RuntimeError("环境评估失败。") from last_error


def evaluate_ppo(model, body, level, episodes, max_steps, seed) -> dict:
    """PPOの決定論的平均行動を評価する。"""
    def controller(obs, _env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_random(body, level, episodes, max_steps, seed) -> dict:
    """一様ランダム行動を未学習基準として評価する。"""
    rng = np.random.default_rng(seed)

    def controller(_obs, env):
        return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_hierarchical(
    approach_model,
    recovery_model,
    body,
    level,
    episodes,
    max_steps,
    seed,
    handoff_x: float | None = None,
) -> dict:
    """接近越障教師と着地回復方策を段階に応じて切り替えて評価する。"""
    def controller(obs, env):
        phase = env.unwrapped._phase
        if handoff_x is None:
            use_recovery = phase != "approach"
        else:
            positions = env.unwrapped.object_pos_at_time(
                env.unwrapped.get_time(),
                "robot",
            )
            use_recovery = phase != "approach" or float(np.mean(positions[0])) >= handoff_x
        model = recovery_model if use_recovery else approach_model
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_three_stage(
    approach_model,
    clearance_model,
    recovery_model,
    handoff_x,
    body,
    level,
    episodes,
    max_steps,
    seed,
    recovery_action_scale: float = 1.0,
) -> dict:
    """接近、直立通過、着地回復の三方策を一つの完全環境で評価する。"""
    def controller(obs, env):
        raw = env.unwrapped
        phase = raw._phase
        if phase != "approach":
            model = recovery_model
        else:
            positions = raw.object_pos_at_time(raw.get_time(), "robot")
            com_x = float(np.mean(positions[0]))
            model = approach_model if com_x < handoff_x else clearance_model
        action, _ = model.predict(obs, deterministic=True)
        if phase != "approach" and recovery_action_scale < 1.0:
            action = recovery_action_scale * np.asarray(action, dtype=np.float32)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_four_stage(
    approach_model,
    clearance_model,
    landing_model,
    restart_model,
    handoff_x,
    body,
    level,
    episodes,
    max_steps,
    seed,
) -> dict:
    """接近、通過、着地、再前進の四方策を完全環境で評価する。"""
    def controller(obs, env):
        raw = env.unwrapped
        phase = raw._phase
        if phase == "landing":
            model = landing_model
        elif phase == "restart":
            model = restart_model
        else:
            positions = raw.object_pos_at_time(raw.get_time(), "robot")
            com_x = float(np.mean(positions[0]))
            model = approach_model if com_x < handoff_x else clearance_model
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_repeating_stages(
    approach_model,
    clearance_model,
    landing_model,
    restart_model,
    handoff_distance,
    body,
    level,
    episodes,
    max_steps,
    seed,
) -> dict:
    """各障害物で四段階方策を相対距離に基づいて繰り返し評価する。"""
    def controller(obs, env):
        raw = env.unwrapped
        phase = raw._phase
        if phase == "landing":
            model = landing_model
        elif phase == "restart":
            model = restart_model
        elif raw._active_obstacle < len(raw.course.obstacles):
            positions = raw.object_pos_at_time(raw.get_time(), "robot")
            com_x = float(np.mean(positions[0]))
            obstacle_x = (
                raw.course.obstacles[raw._active_obstacle].start_x * raw.VOXEL_SIZE
            )
            model = (
                approach_model
                if obstacle_x - com_x > handoff_distance
                else clearance_model
            )
        else:
            model = restart_model
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_prefix_then_agent(
    approach_model,
    clearance_model,
    landing_model,
    restart_model,
    agent_model,
    prefix_validated,
    handoff_distance,
    body,
    level,
    episodes,
    max_steps,
    seed,
) -> dict:
    """教師前綴の完了後だけ学習方策へ切り替えて全環境で評価する。"""
    def controller(obs, env):
        raw = env.unwrapped
        if raw._validated_count >= prefix_validated:
            model = agent_model
        elif raw._phase == "landing":
            model = landing_model
        elif raw._phase == "restart":
            model = restart_model
        else:
            positions = raw.object_pos_at_time(raw.get_time(), "robot")
            com_x = float(np.mean(positions[0]))
            obstacle_x = (
                raw.course.obstacles[raw._active_obstacle].start_x * raw.VOXEL_SIZE
            )
            model = (
                approach_model
                if obstacle_x - com_x > handoff_distance
                else clearance_model
            )
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_prefix_clearance_then_agent(
    approach_model,
    clearance_model,
    landing_model,
    restart_model,
    next_clearance_model,
    agent_model,
    prefix_validated,
    prefix_clearances,
    handoff_distance,
    body,
    level,
    episodes,
    max_steps,
    seed,
) -> dict:
    """既検証前綴と次通過教師の後だけ学習方策へ切り替えて評価する。"""
    def controller(obs, env):
        raw = env.unwrapped
        if raw._validated_count >= prefix_validated:
            model = (
                next_clearance_model
                if raw._strict_cleared_count < prefix_clearances
                else agent_model
            )
        elif raw._phase == "landing":
            model = landing_model
        elif raw._phase == "restart":
            model = restart_model
        else:
            positions = raw.object_pos_at_time(raw.get_time(), "robot")
            com_x = float(np.mean(positions[0]))
            obstacle_x = (
                raw.course.obstacles[raw._active_obstacle].start_x * raw.VOXEL_SIZE
            )
            model = (
                approach_model
                if obstacle_x - com_x > handoff_distance
                else clearance_model
            )
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_prefix_clearance_landing_then_agent(
    approach_model,
    clearance_model,
    landing_model,
    restart_model,
    next_clearance_model,
    next_landing_model,
    agent_model,
    prefix_validated,
    prefix_clearances,
    handoff_distance,
    body,
    level,
    episodes,
    max_steps,
    seed,
) -> dict:
    """次障害物の着地完了後だけ再前進学習方策へ切り替えて評価する。"""
    def controller(obs, env):
        raw = env.unwrapped
        if raw._validated_count >= prefix_validated:
            if raw._strict_cleared_count < prefix_clearances:
                model = next_clearance_model
            elif raw._phase == "landing":
                model = next_landing_model
            else:
                model = agent_model
        elif raw._phase == "landing":
            model = landing_model
        elif raw._phase == "restart":
            model = restart_model
        else:
            positions = raw.object_pos_at_time(raw.get_time(), "robot")
            com_x = float(np.mean(positions[0]))
            obstacle_x = (
                raw.course.obstacles[raw._active_obstacle].start_x * raw.VOXEL_SIZE
            )
            model = (
                approach_model
                if obstacle_x - com_x > handoff_distance
                else clearance_model
            )
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_policy_list_then_agent(
    approach_model,
    obstacle_policies,
    agent_model,
    prefix_validated,
    handoff_distance,
    body,
    level,
    episodes,
    max_steps,
    seed,
    next_clearance_model=None,
    next_landing_model=None,
    prefix_clearances=None,
) -> dict:
    """障害物別の検証済み前綴と現在学習中の方策を正式条件で評価する。"""
    def controller(obs, env):
        raw = env.unwrapped
        index = int(raw._active_obstacle)
        model_obs = obs
        if raw._validated_count < prefix_validated:
            policy = obstacle_policies[index]
            if raw._phase == "landing":
                model = policy["landing"]
            elif raw._phase == "restart":
                model = policy["restart"]
            else:
                positions = raw.object_pos_at_time(raw.get_time(), "robot")
                com_x = float(np.mean(positions[0]))
                obstacle_x = raw.course.obstacles[index].start_x * raw.VOXEL_SIZE
                approach_model_for_obstacle = policy.get("approach", approach_model)
                model = (
                    approach_model_for_obstacle
                    if obstacle_x - com_x > handoff_distance
                    else policy["clearance"]
                )
            model_obs = mask_unseen_future_obstacles(
                obs,
                env,
                policy.get("visible_obstacle_count"),
            )
        elif (
            next_clearance_model is not None
            and raw._strict_cleared_count < prefix_clearances
        ):
            model = next_clearance_model
        elif next_landing_model is not None and raw._phase == "landing":
            model = next_landing_model
        else:
            model = agent_model
        action, _ = model.predict(model_obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def score_metrics(metrics: dict) -> tuple[float, ...]:
    """真の成功に近い順でモデルを辞書式比較する値を返す。"""
    return (
        float(metrics["success_rate"]),
        float(metrics["mean_validated_obstacles"]),
        float(metrics["mean_restart_successes"]),
        float(metrics["mean_stable_landings"]),
        float(metrics["mean_strict_clearances"]),
        float(metrics["mean_max_x"]),
        float(metrics["mean_return"]),
    )
