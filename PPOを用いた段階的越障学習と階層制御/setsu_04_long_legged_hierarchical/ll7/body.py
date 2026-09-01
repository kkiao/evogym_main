"""長脚ロボットの形状と寸法を定義する。"""

from __future__ import annotations

import numpy as np


BODY_NAME = "long_legged"


def make_body() -> np.ndarray:
    """六個のアクチュエータを持つ長脚形状の新しい配列を返す。"""
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


def occupied_width_voxels(body: np.ndarray) -> int:
    """空白の外周を除いた実効身体幅をボクセル数で返す。"""
    occupied_columns = np.flatnonzero(np.any(body != 0, axis=0))
    if occupied_columns.size == 0:
        raise ValueError("身体中没有有效体素。")
    return int(occupied_columns[-1] - occupied_columns[0] + 1)


BODY_WIDTH_VOXELS = occupied_width_voxels(make_body())

