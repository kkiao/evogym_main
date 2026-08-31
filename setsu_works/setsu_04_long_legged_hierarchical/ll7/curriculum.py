"""三身体幅の間隔を持つ七段階固定障害物コースを定義する。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ll7.body import BODY_WIDTH_VOXELS


STATIC_VOXEL = 5
COURSE_VERSION = "long_legged_gap3_curriculum_v2"
COURSE_HEIGHT = 8
ROBOT_START_X = 2
ROBOT_START_Y = 1
GAP_BODY_WIDTHS = 3
GAP_VOXELS = GAP_BODY_WIDTHS * BODY_WIDTH_VOXELS
FIRST_OBSTACLE_X = 14
CURRICULUM_LEVELS = tuple(range(8))


@dataclass(frozen=True)
class ObstacleTemplate:
    """障害物の名称、高さ輪郭、説明を保持する。"""

    name: str
    heights: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class ObstacleSpec:
    """配置済み障害物の位置と高さ輪郭を保持する。"""

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
            "body_width_voxels": BODY_WIDTH_VOXELS,
            "gap_body_widths": GAP_BODY_WIDTHS,
            "gap_voxels": GAP_VOXELS,
            "obstacle_count": len(self.obstacles),
            "obstacles": [item.as_dict() for item in self.obstacles],
            "description": self.description,
        }


LOW_1 = ObstacleTemplate("low_narrow_1", (1,), "第一个低い細壁")
LOW_2 = ObstacleTemplate("low_narrow_2", (1,), "二つ目の低い細壁")
LOW_3 = ObstacleTemplate("low_narrow_3", (1,), "三つ目の低い細壁")
LOW_WIDE = ObstacleTemplate("low_wide", (1, 1), "低い幅広壁")
TALL_NARROW = ObstacleTemplate("tall_narrow", (2,), "高い細壁")
TALL_WIDE = ObstacleTemplate("tall_wide", (2, 2), "高い幅広壁")
STEPPED = ObstacleTemplate("stepped_mound", (1, 2, 1), "階段状障害物")
FINAL_TALL = ObstacleTemplate("final_tall", (2,), "終点前の高壁")


def _place(templates: tuple[ObstacleTemplate, ...]) -> tuple[ObstacleSpec, ...]:
    """各障害物間に正確に三身体幅の空地を残して順に配置する。"""
    placed = []
    next_x = FIRST_OBSTACLE_X
    for template in templates:
        obstacle = ObstacleSpec(
            template.name,
            next_x,
            template.heights,
            template.description,
        )
        placed.append(obstacle)
        next_x = obstacle.end_x + 1 + GAP_VOXELS
    return tuple(placed)


LEVEL_TEMPLATES = {
    1: (LOW_1,),
    2: (LOW_1, LOW_2),
    3: (LOW_1, LOW_2, LOW_3),
    4: (LOW_1, LOW_2, LOW_3, LOW_WIDE),
    5: (LOW_1, LOW_2, LOW_3, LOW_WIDE, TALL_NARROW),
    6: (LOW_1, LOW_2, LOW_3, LOW_WIDE, TALL_NARROW, STEPPED),
    7: (
        LOW_1,
        LOW_2,
        LOW_3,
        LOW_WIDE,
        TALL_NARROW,
        STEPPED,
        TALL_WIDE,
    ),
}

LEVEL_NAMES = {
    0: "flat_walk_and_restart",
    1: "single_low",
    2: "two_spaced_low",
    3: "three_with_width",
    4: "four_with_height",
    5: "five_tall_and_wide",
    6: "six_with_step",
    7: "full_seven_mixed",
}

LEVEL_DESCRIPTIONS = {
    0: "平地歩行の事前学習",
    1: "越过单个低い細壁，稳定落地并恢复前进",
    2: "连续完成两个相距三身体宽度的低い細壁",
    3: "连续完成三个相同的低い細壁",
    4: "在三个低い細壁后加入低い幅広壁",
    5: "加入较高的窄墙",
    6: "加入阶梯形或不同轮廓障碍",
    7: "完成七个混合固定障碍",
}

LEVEL_MAX_STEPS = {
    0: 900,
    1: 2_000,
    2: 3_000,
    3: 3_800,
    4: 4_600,
    5: 5_400,
    6: 6_200,
    7: 7_200,
}


def _make_course(level: int) -> CourseSpec:
    """段階番号から不変のコース定義を構築する。"""
    obstacles = _place(LEVEL_TEMPLATES.get(level, ()))
    if obstacles:
        finish_x = obstacles[-1].end_x + GAP_VOXELS + 10
    else:
        finish_x = 25
    width = finish_x + 10
    return CourseSpec(
        level=level,
        name=LEVEL_NAMES[level],
        width=width,
        finish_x=finish_x,
        max_steps=LEVEL_MAX_STEPS[level],
        obstacles=obstacles,
        description=LEVEL_DESCRIPTIONS[level],
    )


COURSES = {level: _make_course(level) for level in CURRICULUM_LEVELS}


def get_course(level: int) -> CourseSpec:
    """指定レベルの不変なコース定義を返す。"""
    try:
        return COURSES[level]
    except KeyError as exc:
        raise ValueError(f"未知课程等级：{level}；可选值：0-7") from exc


def obstacle_empty_gaps(course: CourseSpec) -> tuple[int, ...]:
    """隣接障害物間の空地ボクセル数を返す。"""
    return tuple(
        current.start_x - previous.end_x - 1
        for previous, current in zip(course.obstacles, course.obstacles[1:])
    )


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
