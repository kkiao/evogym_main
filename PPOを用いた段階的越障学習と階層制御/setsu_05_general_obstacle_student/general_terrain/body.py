"""汎用障害物計画で固定使用する長脚形状と寸法を定義する。"""

from __future__ import annotations

import numpy as np


BODY_NAME = "long_legged"


def make_body() -> np.ndarray:
    """既存教師と同一の長脚ロボット配列を返す。"""
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


def occupied_dimensions(body: np.ndarray) -> tuple[int, int]:
    """空白外周を除いた実効幅と実効高さを返す。"""
    rows = np.flatnonzero(np.any(body != 0, axis=1))
    columns = np.flatnonzero(np.any(body != 0, axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("机器人身体不能为空。")
    width = int(columns[-1] - columns[0] + 1)
    height = int(rows[-1] - rows[0] + 1)
    return width, height


BODY_WIDTH_VOXELS, BODY_HEIGHT_VOXELS = occupied_dimensions(make_body())
