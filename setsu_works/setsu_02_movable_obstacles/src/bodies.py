"""本プロジェクトで固定して使用する二種類のロボット形状。"""

from __future__ import annotations

import numpy as np


BODY_NAMES = ("original", "layered")


def make_original_body() -> np.ndarray:
    """旧プロジェクトの四アクチュエータ形状。旧実験を変更せず定義だけを複製する。"""
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 3, 1, 3, 1],
            [0, 4, 0, 4, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


def make_layered_body() -> np.ndarray:
    """上段が剛体、中段が横方向、下段が縦方向アクチュエータの層状形状。"""
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4],
            [0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


def make_body(body_name: str) -> np.ndarray:
    if body_name == "original":
        return make_original_body()
    if body_name == "layered":
        return make_layered_body()
    raise ValueError(f"未知身体 {body_name!r}；可选值：{', '.join(BODY_NAMES)}")
