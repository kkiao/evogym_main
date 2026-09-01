"""公平な比較に使用する三種類の固定ロボット形状。"""

from __future__ import annotations

import numpy as np


BODY_NAMES = ("original", "layered", "long_legged")


def make_original_body() -> np.ndarray:
    """四つのアクチュエータを持つ元の形状を返す。"""
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
    """上段が剛体で、二層のアクチュエータを持つ層状形状を返す。"""
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


def make_long_legged_body() -> np.ndarray:
    """縦方向アクチュエータを一段延長した六アクチュエータ形状を返す。"""
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 3, 1, 3, 1],
            [0, 4, 0, 4, 0],
            [0, 4, 0, 4, 0],
        ],
        dtype=int,
    )


def make_body(body_name: str) -> np.ndarray:
    """登録名に対応する形状の新しい配列を返す。"""
    factories = {
        "original": make_original_body,
        "layered": make_layered_body,
        "long_legged": make_long_legged_body,
    }
    try:
        return factories[body_name]()
    except KeyError as exc:
        raise ValueError(
            f"未知身体 {body_name!r}；可选值：{', '.join(BODY_NAMES)}"
        ) from exc
