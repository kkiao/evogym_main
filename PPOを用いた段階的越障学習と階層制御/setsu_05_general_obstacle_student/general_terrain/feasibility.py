"""長脚形状に対する保守的な幾何可行性包絡を検査する。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from general_terrain.body import BODY_HEIGHT_VOXELS, BODY_WIDTH_VOXELS
from general_terrain.terrain import (
    END_RUNWAY_VOXELS,
    MIN_GAP_VOXELS,
    START_RUNWAY_VOXELS,
    CourseSpec,
    ObstacleTemplate,
    make_course_array,
)


@dataclass(frozen=True)
class CapabilityEnvelope:
    """実証前の候補地形を絞る保守的な形態上限を保持する。"""

    body_width: int = BODY_WIDTH_VOXELS
    body_height: int = BODY_HEIGHT_VOXELS
    max_obstacle_height: int = 2
    max_profile_width: int = 7
    max_walkable_rise: int = 1
    max_jump_rise: int = 2
    max_vertical_top_width: int = 2
    minimum_gap: int = MIN_GAP_VOXELS
    minimum_start_runway: int = START_RUNWAY_VOXELS
    minimum_end_runway: int = END_RUNWAY_VOXELS


DEFAULT_ENVELOPE = CapabilityEnvelope()


def template_errors(
    template: ObstacleTemplate,
    envelope: CapabilityEnvelope = DEFAULT_ENVELOPE,
) -> list[str]:
    """一障害物が能力包絡を外れる理由を列挙する。"""
    errors = []
    heights = template.heights
    if not heights:
        errors.append("高度轮廓不能为空")
        return errors
    if any(not isinstance(value, int) or value <= 0 for value in heights):
        errors.append("障碍轮廓必须由接地的正整数高度组成")
    if max(heights) > envelope.max_obstacle_height:
        errors.append("障碍高度超过保守上限")
    if len(heights) > envelope.max_profile_width:
        errors.append("单个组合障碍过宽")
    if max(heights) * 2 > envelope.body_height:
        errors.append("障碍高度超过身体有效高度的一半")

    surface = (0,) + heights + (0,)
    rises = [current - previous for previous, current in zip(surface, surface[1:])]
    if max(rises) > envelope.max_jump_rise:
        errors.append("局部上升超过跳跃候选上限")
    if max(rises) > envelope.max_walkable_rise:
        if template.family != "jump_wall":
            errors.append("非跳跃障碍出现了过大的垂直上升")
        if template.width > envelope.max_vertical_top_width:
            errors.append("垂直高墙顶部过宽")
    if abs(min(rises)) > envelope.max_jump_rise:
        errors.append("局部下降超过安全落差上限")
    return errors


def course_errors(
    course: CourseSpec,
    envelope: CapabilityEnvelope = DEFAULT_ENVELOPE,
) -> list[str]:
    """複合コースの助走、間隔、着地余白と配列整合性を検査する。"""
    errors = []
    if not course.obstacles:
        errors.append("场地没有障碍")
        return errors
    if course.obstacles[0].start_x < envelope.minimum_start_runway:
        errors.append("首个障碍前的助跑距离不足")
    for obstacle in course.obstacles:
        errors.extend(
            f"{obstacle.template.name}: {message}"
            for message in template_errors(obstacle.template, envelope)
        )
    for previous, current in zip(course.obstacles, course.obstacles[1:]):
        gap = current.start_x - previous.end_x - 1
        if gap < envelope.minimum_gap:
            errors.append("相邻障碍间距不足三个身体宽度")
    landing_space = course.finish_x - course.obstacles[-1].end_x - 1
    if landing_space < envelope.minimum_end_runway:
        errors.append("最后障碍后的稳定落地区不足")
    terrain = make_course_array(course)
    if terrain.shape[1] != course.width:
        errors.append("地形数组宽度与课程定义不一致")
    if not np.all(terrain[-1] != 0):
        errors.append("地面存在断裂")
    if np.any(terrain[:-1] != 0):
        occupied = np.argwhere(terrain[:-1] != 0)
        for row, column in occupied:
            if not np.any(terrain[row + 1 :, column] != 0):
                errors.append("存在悬空静态体素")
                break
    return errors


def assert_template_feasible(
    template: ObstacleTemplate,
    envelope: CapabilityEnvelope = DEFAULT_ENVELOPE,
) -> None:
    """障害物が包絡外なら理由付き例外を送出する。"""
    errors = template_errors(template, envelope)
    if errors:
        raise ValueError(f"障碍 {template.name} 不可加入候选库：{'；'.join(errors)}")


def assert_course_feasible(
    course: CourseSpec,
    envelope: CapabilityEnvelope = DEFAULT_ENVELOPE,
) -> None:
    """コースが包絡外なら理由付き例外を送出する。"""
    errors = course_errors(course, envelope)
    if errors:
        raise ValueError(f"场地 {course.course_id} 未通过可行性检查：{'；'.join(errors)}")
