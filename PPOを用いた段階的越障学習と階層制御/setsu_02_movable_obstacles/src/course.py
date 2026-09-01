"""固定された混合障害物コースの構造定義。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


STATIC_VOXEL = 5
COURSE_VERSION = "mixed_obstacles_v1"
COURSE_WIDTH = 100
COURSE_HEIGHT = 7
ROBOT_START_X = 2
LEGACY_ROBOT_START_Y = 4
GROUNDED_ROBOT_START_Y = 1


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    start_x: int
    heights: tuple[int, ...]
    description: str

    @property
    def end_x(self) -> int:
        return self.start_x + len(self.heights) - 1

    @property
    def max_height(self) -> int:
        return max(self.heights)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["heights"] = list(self.heights)
        data["end_x"] = self.end_x
        data["max_height"] = self.max_height
        return data


# 高さ・幅・輪郭の違いを含め、易しい順に並べる。障害物間には姿勢回復用の間隔を置く。
OBSTACLES = (
    ObstacleSpec("low_narrow", 14, (1,), "低い細壁"),
    ObstacleSpec("tall_narrow", 25, (2,), "高い細壁"),
    ObstacleSpec("low_wide", 36, (1, 1), "低い幅広壁"),
    ObstacleSpec("tall_wide", 48, (2, 2), "高い幅広壁"),
    ObstacleSpec("stepped_mound", 61, (1, 2, 1), "階段状の小丘"),
    ObstacleSpec("double_low", 74, (1, 0, 1), "連続する二つの低壁"),
    ObstacleSpec("final_tall", 86, (2,), "終点前の高壁"),
)

FINAL_OBSTACLE_END_X = OBSTACLES[-1].end_x
FINISH_X = 91
BOUNDARY_X = COURSE_WIDTH - 1


def make_course_array(include_task_obstacles: bool = True) -> np.ndarray:
    """EvoGym用のボクセル配列を返す。配列の最終行がワールド座標y=0になる。"""
    course = np.zeros((COURSE_HEIGHT, COURSE_WIDTH), dtype=int)
    course[-1, :] = STATIC_VOXEL

    if include_task_obstacles:
        for obstacle in OBSTACLES:
            for offset, height in enumerate(obstacle.heights):
                x = obstacle.start_x + offset
                for level in range(height):
                    course[-2 - level, x] = STATIC_VOXEL

    # 右端境界はロボットの画面外離脱だけを防ぎ、七つの課題障害物には数えない。
    for level in range(COURSE_HEIGHT - 1):
        course[-2 - level, BOUNDARY_X] = STATIC_VOXEL
    return course


def make_rigid_obstacle_array(obstacle: ObstacleSpec) -> np.ndarray:
    """高さ輪郭を可動の剛体ボクセルオブジェクトへ変換する。"""
    shape = np.zeros((obstacle.max_height, len(obstacle.heights)), dtype=int)
    for column, height in enumerate(obstacle.heights):
        if height:
            shape[-height:, column] = 1
    return shape


def rigid_obstacle_components(obstacle: ObstacleSpec) -> list[tuple[int, np.ndarray]]:
    """高さゼロの列で分割し、非連結な剛体が一つの動的物体に入るのを防ぐ。"""
    components = []
    start = None
    for index, height in enumerate((*obstacle.heights, 0)):
        if height > 0 and start is None:
            start = index
        if height == 0 and start is not None:
            heights = obstacle.heights[start:index]
            component = ObstacleSpec(
                name=obstacle.name,
                start_x=obstacle.start_x + start,
                heights=heights,
                description=obstacle.description,
            )
            components.append((start, make_rigid_obstacle_array(component)))
            start = None
    return components


def course_metadata(robot_start_y: int = GROUNDED_ROBOT_START_Y) -> dict:
    return {
        "version": COURSE_VERSION,
        "width": COURSE_WIDTH,
        "height": COURSE_HEIGHT,
        "robot_start": [ROBOT_START_X, robot_start_y],
        "finish_x": FINISH_X,
        "boundary_x": BOUNDARY_X,
        "obstacle_count": len(OBSTACLES),
        "obstacles": [obstacle.as_dict() for obstacle in OBSTACLES],
    }
