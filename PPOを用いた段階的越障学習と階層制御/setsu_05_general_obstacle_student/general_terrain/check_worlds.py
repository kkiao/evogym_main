"""候補場がEvoGym物理世界として初期化・中立作動できるか確認する。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from evogym import EvoSim, EvoWorld, get_full_connectivity

from general_terrain.body import make_body
from general_terrain.feasibility import assert_course_feasible
from general_terrain.terrain import CATALOG, ROBOT_START_X, ROBOT_START_Y, build_course, make_course_array, sample_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def smoke_course(course):
    """一つの場で物理器を初期化し、中立動作十歩の有限性を確認する。"""
    assert_course_feasible(course)
    body = make_body()
    world = EvoWorld()
    world.add_from_array("ground", make_course_array(course), 0, 0)
    world.add_from_array(
        "robot",
        body,
        ROBOT_START_X,
        ROBOT_START_Y,
        connections=get_full_connectivity(body),
    )
    sim = EvoSim(world)
    sim.reset()
    neutral = np.ones(sim.get_dim_action_space("robot"), dtype=float)
    unstable = False
    for _ in range(10):
        sim.set_action("robot", neutral)
        unstable = bool(sim.step()) or unstable
    positions = sim.object_pos_at_time(sim.get_time(), "robot")
    if not np.all(np.isfinite(positions)):
        raise RuntimeError(f"场地 {course.course_id} 产生了非有限坐标。")
    if unstable:
        raise RuntimeError(f"场地 {course.course_id} 在中立动作冒烟测试中不稳定。")
    return {"course_id": course.course_id, "steps": 10, "finite": True, "unstable": False}


def main():
    courses = [
        build_course(
            [template.name],
            split=template.split,
            seed=500 + index,
            difficulty=template.difficulty,
        )
        for index, template in enumerate(CATALOG.values())
    ]
    courses.extend(
        (
            sample_course(601, 1, 4, "train"),
            sample_course(602, 2, 5, "train"),
            sample_course(603, 3, 6, "train"),
            sample_course(604, 3, 4, "validation"),
            sample_course(605, 3, 4, "holdout"),
        )
    )
    results = [smoke_course(course) for course in courses]
    output = PROJECT_ROOT / "assets" / "physics_smoke_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"physics_worlds={len(results)} all_stable=True", flush=True)


if __name__ == "__main__":
    main()
