"""本プロジェクトで使用するGymnasium環境ラッパー。"""

import gymnasium as gym
import numpy as np


class NormalizeActionSpace(gym.ActionWrapper):
    """エージェントの[-1, 1]行動を環境本来の行動範囲へ線形変換する。"""

    def __init__(self, env):
        super().__init__(env)
        self.original_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.original_high = np.asarray(env.action_space.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action):
        normalized = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        original = self.original_low + (normalized + 1.0) * 0.5 * (
            self.original_high - self.original_low
        )
        return np.clip(original, self.original_low, self.original_high)

    def reverse_action(self, action):
        original = np.asarray(action, dtype=np.float32)
        normalized = 2.0 * (original - self.original_low) / (
            self.original_high - self.original_low
        ) - 1.0
        return np.clip(normalized, -1.0, 1.0)
