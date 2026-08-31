"""長脚ロボット向けの手続き型汎用障害物場を提供する。"""

from general_terrain.body import make_body
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.feasibility import assert_course_feasible, assert_template_feasible
from general_terrain.terrain import CATALOG, build_course, make_course_array, sample_course

__all__ = (
    "CATALOG",
    "GeneralObstacleEnv",
    "assert_course_feasible",
    "assert_template_feasible",
    "build_course",
    "make_body",
    "make_course_array",
    "sample_course",
)
