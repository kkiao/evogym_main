"""形状多様性を持つ接地型障害物と手続き型コースを定義する。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np

from general_terrain.body import BODY_WIDTH_VOXELS


STATIC_VOXEL = 5
WORLD_HEIGHT = 10
ROBOT_START_X = 2
ROBOT_START_Y = 1
START_RUNWAY_VOXELS = 4 * BODY_WIDTH_VOXELS
MAX_START_RUNWAY_VOXELS = 6 * BODY_WIDTH_VOXELS
END_RUNWAY_VOXELS = 4 * BODY_WIDTH_VOXELS
MIN_GAP_VOXELS = 3 * BODY_WIDTH_VOXELS
MAX_GAP_VOXELS = 5 * BODY_WIDTH_VOXELS
COURSE_VERSION = "general_grounded_obstacles_v1"


@dataclass(frozen=True)
class ObstacleTemplate:
    """一つの接地障害物の高さ輪郭と使用区分を保持する。"""

    name: str
    heights: tuple[int, ...]
    family: str
    intended_skill: str
    difficulty: int
    split: str
    empirical_status: str
    description: str

    @property
    def width(self) -> int:
        return len(self.heights)

    @property
    def max_height(self) -> int:
        return max(self.heights)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["heights"] = list(self.heights)
        data["width"] = self.width
        data["max_height"] = self.max_height
        return data


@dataclass(frozen=True)
class PlacedObstacle:
    """コース上へ配置された障害物と開始位置を保持する。"""

    template: ObstacleTemplate
    start_x: int

    @property
    def end_x(self) -> int:
        return self.start_x + self.template.width - 1

    def as_dict(self) -> dict:
        data = self.template.as_dict()
        data["start_x"] = self.start_x
        data["end_x"] = self.end_x
        return data


@dataclass(frozen=True)
class CourseSpec:
    """再生成可能な一つの手続き型障害物コースを保持する。"""

    course_id: str
    split: str
    seed: int
    difficulty: int
    width: int
    finish_x: int
    obstacles: tuple[PlacedObstacle, ...]

    def as_dict(self) -> dict:
        return {
            "version": COURSE_VERSION,
            "course_id": self.course_id,
            "split": self.split,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "width": self.width,
            "height": WORLD_HEIGHT,
            "finish_x": self.finish_x,
            "robot_start": [ROBOT_START_X, ROBOT_START_Y],
            "start_runway_voxels": self.obstacles[0].start_x,
            "minimum_gap_voxels": MIN_GAP_VOXELS,
            "obstacles": [item.as_dict() for item in self.obstacles],
        }


TEMPLATES = (
    ObstacleTemplate(
        "low_hurdle",
        (1,),
        "hurdle",
        "step_over",
        1,
        "train",
        "teacher_verified",
        "一体素の低い壁で、既存教師による通過証拠がある。",
    ),
    ObstacleTemplate(
        "low_platform_short",
        (1, 1, 1),
        "platform",
        "climb_walk_down",
        1,
        "train",
        "geometry_screened",
        "低く短い台で、登った後も歩行を継続できる。",
    ),
    ObstacleTemplate(
        "low_platform_body_width",
        (1, 1, 1, 1, 1),
        "platform",
        "climb_walk_down",
        2,
        "validation",
        "geometry_screened",
        "幅が身体一個分に等しい低い台。",
    ),
    ObstacleTemplate(
        "triangular_mound",
        (1, 2, 1),
        "mound",
        "step_up_and_down",
        2,
        "train",
        "geometry_screened",
        "体素を積み上げた三角形輪郭の小丘。",
    ),
    ObstacleTemplate(
        "staircase_plateau",
        (1, 1, 2, 2, 1, 1),
        "stairs",
        "stair_walk",
        2,
        "train",
        "geometry_screened",
        "二段の上昇、短い台、二段の下降からなる階段。",
    ),
    ObstacleTemplate(
        "asymmetric_terrace",
        (1, 1, 2, 2, 2, 1),
        "terrace",
        "asymmetric_step_control",
        2,
        "train",
        "geometry_screened",
        "上りと下りの長さが異なる非対称台地。",
    ),
    ObstacleTemplate(
        "narrow_high_hurdle",
        (2,),
        "jump_wall",
        "jump_or_climb",
        3,
        "train",
        "morphology_upper_bound",
        "高さが身体有効高の半分である細い壁。",
    ),
    ObstacleTemplate(
        "short_high_block",
        (2, 2),
        "jump_wall",
        "jump_or_climb",
        3,
        "validation",
        "morphology_upper_bound",
        "高さ二、幅二の短い高壁。",
    ),
    ObstacleTemplate(
        "double_peak",
        (1, 2, 1, 2, 1),
        "wave",
        "repeated_step_control",
        3,
        "validation",
        "geometry_screened",
        "二つの連続峰からなる波状輪郭。",
    ),
    ObstacleTemplate(
        "stepped_wave",
        (1, 1, 2, 1, 1, 2, 1),
        "wave",
        "unseen_repeated_steps",
        3,
        "holdout",
        "geometry_screened",
        "訓練中に現れない長い波状積層組合せ。",
    ),
    ObstacleTemplate(
        "terraced_wedge",
        (1, 2, 2, 1),
        "wedge",
        "unseen_asymmetric_mound",
        3,
        "holdout",
        "geometry_screened",
        "階段状体素からなる非矩形のくさび形障害物。",
    ),
    ObstacleTemplate(
        "broad_mound",
        (1, 2, 2, 2, 1),
        "mound",
        "unseen_broad_peak",
        3,
        "holdout",
        "geometry_screened",
        "頂部が広く、両側を段階的に上り下りできる小丘。",
    ),
)

CATALOG = {item.name: item for item in TEMPLATES}


def build_course(
    template_names: list[str] | tuple[str, ...],
    *,
    split: str,
    seed: int,
    difficulty: int,
    gaps: list[int] | tuple[int, ...] | None = None,
    start_runway_voxels: int = START_RUNWAY_VOXELS,
) -> CourseSpec:
    """指定テンプレート列と間隔から不変のコースを構築する。"""
    if not template_names:
        raise ValueError("场地至少需要一个障碍。")
    templates = [CATALOG[name] for name in template_names]
    if gaps is None:
        rng = random.Random(seed)
        gaps = [rng.randint(MIN_GAP_VOXELS, MAX_GAP_VOXELS) for _ in templates[:-1]]
    if len(gaps) != len(templates) - 1:
        raise ValueError("障碍间距数量必须等于障碍数量减一。")
    if start_runway_voxels < START_RUNWAY_VOXELS:
        raise ValueError("首个障碍前至少需要四个身体宽度的助跑区。")

    placed = []
    next_x = int(start_runway_voxels)
    for index, template in enumerate(templates):
        obstacle = PlacedObstacle(template=template, start_x=next_x)
        placed.append(obstacle)
        if index < len(gaps):
            next_x = obstacle.end_x + 1 + int(gaps[index])
    finish_x = placed[-1].end_x + 1 + END_RUNWAY_VOXELS
    width = finish_x + BODY_WIDTH_VOXELS
    names = "-".join(template_names)
    return CourseSpec(
        course_id=f"{split}_d{difficulty}_s{seed}_x{start_runway_voxels}_{names}",
        split=split,
        seed=seed,
        difficulty=difficulty,
        width=width,
        finish_x=finish_x,
        obstacles=tuple(placed),
    )


def sample_course(
    seed: int,
    difficulty: int,
    obstacle_count: int,
    split: str = "train",
) -> CourseSpec:
    """難度、区分、乱数種から再現可能な複合コースを標本化する。"""
    if difficulty not in (1, 2, 3):
        raise ValueError("难度必须是1、2或3。")
    if obstacle_count < 1:
        raise ValueError("障碍数量必须至少为1。")
    candidates = [
        item
        for item in TEMPLATES
        if item.split == split and item.difficulty <= difficulty
    ]
    if not candidates:
        raise ValueError(f"区分 {split} 在难度 {difficulty} 下没有可选障碍。")
    rng = random.Random(seed)
    selected = []
    for _ in range(obstacle_count):
        choices = [item for item in candidates if not selected or item.name != selected[-1]]
        selected.append(rng.choice(choices or candidates))
    gaps = [rng.randint(MIN_GAP_VOXELS, MAX_GAP_VOXELS) for _ in range(obstacle_count - 1)]
    start_runway = rng.randint(START_RUNWAY_VOXELS, MAX_START_RUNWAY_VOXELS)
    return build_course(
        [item.name for item in selected],
        split=split,
        seed=seed,
        difficulty=difficulty,
        gaps=gaps,
        start_runway_voxels=start_runway,
    )


def make_course_array(course: CourseSpec) -> np.ndarray:
    """接地障害物を静的ボクセルで満たしたEvoGym配列を返す。"""
    terrain = np.zeros((WORLD_HEIGHT, course.width), dtype=int)
    terrain[-1, :] = STATIC_VOXEL
    for obstacle in course.obstacles:
        for offset, height in enumerate(obstacle.template.heights):
            x = obstacle.start_x + offset
            for level in range(height):
                terrain[-2 - level, x] = STATIC_VOXEL
    return terrain
