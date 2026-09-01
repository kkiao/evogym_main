"""固定障害物を段階的に増やすカリキュラムコース定義。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


STATIC_VOXEL = 5
COURSE_VERSION = "fixed_curriculum_v3"
COURSE_HEIGHT = 7
ROBOT_START_X = 2
ROBOT_START_Y = 1
CURRICULUM_LEVELS = tuple(range(8))


@dataclass(frozen=True)
class ObstacleSpec:
    """一つの固定障害物の位置と高さ輪郭を保持する。"""

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


@dataclass(frozen=True)
class CourseSpec:
    """一段階のコース全体と終了条件を保持する。"""

    level: int
    name: str
    width: int
    finish_x: int
    max_steps: int
    obstacles: tuple[ObstacleSpec, ...]
    description: str

    def as_dict(self) -> dict:
        return {
            "version": COURSE_VERSION,
            "level": self.level,
            "name": self.name,
            "width": self.width,
            "height": COURSE_HEIGHT,
            "finish_x": self.finish_x,
            "max_steps": self.max_steps,
            "robot_start": [ROBOT_START_X, ROBOT_START_Y],
            "obstacle_count": len(self.obstacles),
            "obstacles": [item.as_dict() for item in self.obstacles],
            "description": self.description,
        }


LOW_1 = ObstacleSpec("low_narrow_1", 14, (1,), "低い細壁")
LOW_2 = ObstacleSpec("low_narrow_2", 28, (1,), "二つ目の低い細壁")
DENSE_LOW_2 = ObstacleSpec("dense_low_narrow_2", 20, (1,), "近距離にある二つ目の低い細壁")
ADJACENT_LOW_2 = ObstacleSpec("adjacent_low_narrow_2", 16, (1,), "隣接する二つ目の低い細壁")
LOW_WIDE = ObstacleSpec("low_wide", 28, (1, 1), "低い幅広壁")
LOW_3 = ObstacleSpec("low_narrow_3", 44, (1,), "三つ目の低い細壁")
STEPPED = ObstacleSpec("stepped_mound", 44, (1, 2, 1), "階段状障害物")
TALL_NARROW = ObstacleSpec("tall_narrow", 62, (2,), "高い細壁")

FULL_OBSTACLES = (
    ObstacleSpec("low_narrow", 14, (1,), "低い細壁"),
    ObstacleSpec("tall_narrow", 25, (2,), "高い細壁"),
    ObstacleSpec("low_wide", 36, (1, 1), "低い幅広壁"),
    ObstacleSpec("tall_wide", 48, (2, 2), "高い幅広壁"),
    ObstacleSpec("stepped_mound", 61, (1, 2, 1), "階段状の小丘"),
    ObstacleSpec("double_low", 74, (1, 0, 1), "連続する二つの低壁"),
    ObstacleSpec("final_tall", 86, (2,), "終点前の高壁"),
)


COURSES = {
    0: CourseSpec(0, "flat_walk", 35, 20, 600, (), "平地歩行"),
    1: CourseSpec(1, "single_low", 38, 22, 1200, (LOW_1,), "単一固定低壁"),
    2: CourseSpec(
        2,
        "double_low",
        50,
        36,
        1600,
        (LOW_1, LOW_2),
        "連続する二つの固定低壁",
    ),
    3: CourseSpec(
        3,
        "low_mixed",
        62,
        52,
        1800,
        (LOW_1, LOW_WIDE, LOW_3),
        "細い低壁と幅広低壁の組合せ",
    ),
    4: CourseSpec(
        4,
        "step_and_tall",
        82,
        70,
        2200,
        (LOW_1, LOW_WIDE, STEPPED, TALL_NARROW),
        "階段と高い細壁を追加した課程",
    ),
    5: CourseSpec(
        5,
        "full_seven",
        100,
        91,
        2600,
        FULL_OBSTACLES,
        "七組の固定障害物からなる完全課程",
    ),
    6: CourseSpec(
        6,
        "dense_double_low_bridge",
        40,
        28,
        1600,
        (LOW_1, DENSE_LOW_2),
        "最初の越障運動量を利用する密集二低壁の橋渡し課程",
    ),
    7: CourseSpec(
        7,
        "adjacent_double_low_bridge",
        34,
        24,
        1600,
        (LOW_1, ADJACENT_LOW_2),
        "二つの固定低壁の間に地面一体素を残した橋渡し課程",
    ),
}


def get_course(level: int) -> CourseSpec:
    """指定レベルの不変なコース定義を返す。"""
    try:
        return COURSES[level]
    except KeyError as exc:
        raise ValueError(f"未知课程等级：{level}；可选值：0-7") from exc


def make_course_array(level: int) -> np.ndarray:
    """地面と全障害物を一体の静的ボクセル配列として生成する。"""
    course = get_course(level)
    terrain = np.zeros((COURSE_HEIGHT, course.width), dtype=int)
    terrain[-1, :] = STATIC_VOXEL
    for obstacle in course.obstacles:
        for offset, height in enumerate(obstacle.heights):
            x = obstacle.start_x + offset
            for voxel_level in range(height):
                terrain[-2 - voxel_level, x] = STATIC_VOXEL
    return terrain
