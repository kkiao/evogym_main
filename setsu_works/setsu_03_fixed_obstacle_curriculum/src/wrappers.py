"""PPO学習に使用するGymnasiumラッパー。"""

import gymnasium as gym
import numpy as np


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


class TeacherPrefixResetWrapper(gym.Wrapper):
    """教師方策が指定障害物数を通過した直後から学習エピソードを開始する。"""

    def __init__(
        self,
        env,
        teacher_model,
        target_cleared: int,
        max_prefix_steps: int,
        post_prefix_steps: int = 0,
    ):
        super().__init__(env)
        self.teacher_model = teacher_model
        self.target_cleared = target_cleared
        self.max_prefix_steps = max_prefix_steps
        self.post_prefix_steps = post_prefix_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prefix_return = 0.0
        for prefix_steps in range(1, self.max_prefix_steps + 1):
            action, _ = self.teacher_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            prefix_return += float(reward)
            if int(info["obstacles_cleared"]) >= self.target_cleared:
                break
            if terminated or truncated:
                raise RuntimeError("教师策略未能完成课程前缀。")
        else:
            raise RuntimeError("教师策略在最大前缀步数内未完成目标。")

        clearance_steps = prefix_steps
        for _ in range(self.post_prefix_steps):
            action, _ = self.teacher_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            prefix_steps += 1
            prefix_return += float(reward)
            if terminated or truncated:
                raise RuntimeError("教师策略在交接过渡阶段提前结束。")

        raw = self.env.unwrapped
        positions = raw.object_pos_at_time(raw.get_time(), "robot")
        com_x = float(np.mean(positions[0]))
        raw._initial_com_x = com_x
        raw._maximum_com_x = com_x
        raw._maximum_com_y = float(np.mean(positions[1]))
        raw._maximum_bottom_y = float(np.min(positions[1]))
        raw._stall_steps = 0
        raw._last_progress_x = com_x
        info = raw._info(positions, success=False)
        info["prefix_steps"] = prefix_steps
        info["clearance_steps"] = clearance_steps
        info["post_prefix_steps"] = self.post_prefix_steps
        info["prefix_return"] = prefix_return
        return obs, info


class RecoveryGoalWrapper(gym.Wrapper):
    """教師から引き継いだ位置を基準に短い回復移動目標を与える。"""

    def __init__(self, env, goal_distance: float, success_bonus: float = 30.0):
        super().__init__(env)
        self.goal_distance = float(goal_distance)
        self.success_bonus = float(success_bonus)
        self.start_x = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.start_x = float(info["x_position"])
        info = dict(info)
        info["recovery_progress"] = 0.0
        info["recovery_goal_distance"] = self.goal_distance
        info["recovery_goal_reached"] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        progress = float(info["x_position"]) - self.start_x
        reached = progress >= self.goal_distance
        info = dict(info)
        info["recovery_progress"] = progress
        info["recovery_goal_distance"] = self.goal_distance
        info["recovery_goal_reached"] = reached
        if reached:
            reward += self.success_bonus
            terminated = True
            info["is_success"] = True
        return obs, reward, terminated, truncated, info


class ApproachPrefixResetWrapper(gym.Wrapper):
    """回復方策で次の障害物手前まで進めてから学習区間を開始する。"""

    def __init__(
        self,
        env,
        approach_model,
        target_distance: float,
        max_approach_steps: int,
    ):
        super().__init__(env)
        self.approach_model = approach_model
        self.target_distance = float(target_distance)
        self.max_approach_steps = max_approach_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        start_x = float(info["x_position"])
        approach_return = 0.0
        for approach_steps in range(1, self.max_approach_steps + 1):
            action, _ = self.approach_model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = self.env.step(action)
            approach_return += float(reward)
            if float(info["x_position"]) - start_x >= self.target_distance:
                break
            if terminated or truncated:
                raise RuntimeError("接近策略在目标位置之前提前结束。")
        else:
            raise RuntimeError("接近策略未能在最大步数内到达目标位置。")

        raw = self.env.unwrapped
        positions = raw.object_pos_at_time(raw.get_time(), "robot")
        com_x = float(np.mean(positions[0]))
        raw._initial_com_x = com_x
        raw._maximum_com_x = com_x
        raw._maximum_com_y = float(np.mean(positions[1]))
        raw._maximum_bottom_y = float(np.min(positions[1]))
        raw._active_obstacle = raw._cleared_count
        raw._maximum_near_com_y = -float("inf")
        raw._maximum_near_bottom_y = -float("inf")
        raw._maximum_crossed_fraction = 0.0
        raw._maximum_crossing_score = 0.0
        raw._maximum_front_x = float(np.max(positions[0]))
        raw._stall_steps = 0
        raw._last_progress_x = com_x
        info = raw._info(positions, success=False)
        info["approach_steps"] = approach_steps
        info["approach_return"] = approach_return
        info["approach_distance"] = com_x - start_x
        return obs, info
