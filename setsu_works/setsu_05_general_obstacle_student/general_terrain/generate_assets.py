"""障害物図鑑、単体場、複合場とメタデータを一括生成する。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from general_terrain.body import make_body
from general_terrain.feasibility import assert_course_feasible, assert_template_feasible
from general_terrain.terrain import (
    CATALOG,
    ROBOT_START_X,
    WORLD_HEIGHT,
    build_course,
    make_course_array,
    sample_course,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "assets"
MATERIAL_COLORS = ListedColormap(
    ["#ffffff", "#20252b", "#bbbbbb", "#f28e2b", "#73b6e6", "#111111"]
)

COMPOSITE_SPECS = (
    ("train_easy_seed11", 11, 1, 4, "train"),
    ("train_shapes_seed21", 21, 2, 5, "train"),
    ("train_mixed_seed31", 31, 3, 6, "train"),
    ("validation_seed41", 41, 3, 4, "validation"),
    ("validation_seed43", 43, 3, 5, "validation"),
    ("holdout_seed51", 51, 3, 4, "holdout"),
    ("holdout_seed53", 53, 3, 5, "holdout"),
)


def visual_array(course):
    """地形配列へ開始位置のロボット形状を重ねた表示配列を返す。"""
    visual = make_course_array(course).copy()
    body = make_body()
    bottom_row = WORLD_HEIGHT - 1
    top_row = bottom_row - body.shape[0]
    body_slice = visual[top_row:bottom_row, ROBOT_START_X : ROBOT_START_X + body.shape[1]]
    body_slice[body != 0] = body[body != 0]
    return visual


def save_course_preview(course, path, title):
    """一つの場を身体寸法付きPNGとして保存する。"""
    array = visual_array(course)
    width = max(9.0, min(20.0, course.width / 5.0))
    figure, ax = plt.subplots(figsize=(width, 3.2), constrained_layout=True)
    ax.imshow(array, cmap=MATERIAL_COLORS, vmin=0, vmax=5, interpolation="none")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("x voxel")
    ax.set_yticks([])
    ax.set_xlim(0, course.width - 1)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_catalog_preview(path):
    """全障害物輪郭を同一縮尺の図鑑へまとめる。"""
    templates = list(CATALOG.values())
    figure, axes = plt.subplots(3, 4, figsize=(14, 8.5), constrained_layout=True)
    for ax, template in zip(axes.flat, templates):
        x = np.arange(template.width)
        ax.bar(x, template.heights, width=1.0, align="center", color="#222222", edgecolor="white")
        ax.set_ylim(0, 4.2)
        ax.set_xlim(-0.75, max(1.75, template.width - 0.25))
        ax.set_yticks((0, 1, 2, 3, 4))
        ax.grid(axis="y", alpha=0.2)
        ax.set_title(
            f"{template.name}\n{template.family} | D{template.difficulty} | {template.split}",
            fontsize=9,
        )
    figure.suptitle("Long-legged robot: conservative traversable obstacle catalog", fontsize=15, weight="bold")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_composite_preview(courses, path):
    """代表複合コースを一枚の比較図へまとめる。"""
    figure, axes = plt.subplots(len(courses), 1, figsize=(16, 2.5 * len(courses)), constrained_layout=True)
    if len(courses) == 1:
        axes = [axes]
    for ax, (name, course) in zip(axes, courses):
        ax.imshow(visual_array(course), cmap=MATERIAL_COLORS, vmin=0, vmax=5, interpolation="none")
        ax.set_title(f"{name}: {', '.join(item.template.name for item in course.obstacles)}", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(0, course.width - 1)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_course(course, directory, name):
    """一つの場の配列、メタデータ、PNGを同名で保存する。"""
    assert_course_feasible(course)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{name}.npy", make_course_array(course))
    (directory / f"{name}.json").write_text(
        json.dumps(course.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_course_preview(course, directory / f"{name}.png", name)


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for template in CATALOG.values():
        assert_template_feasible(template)
    (ASSET_DIR / "obstacle_catalog.json").write_text(
        json.dumps([item.as_dict() for item in CATALOG.values()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_catalog_preview(ASSET_DIR / "obstacle_catalog.png")

    single_dir = ASSET_DIR / "single_obstacles"
    for index, template in enumerate(CATALOG.values(), start=1):
        course = build_course(
            [template.name],
            split=template.split,
            seed=100 + index,
            difficulty=template.difficulty,
        )
        write_course(course, single_dir, template.name)

    composite_dir = ASSET_DIR / "composite_courses"
    composites = []
    for name, seed, difficulty, count, split in COMPOSITE_SPECS:
        course = sample_course(seed, difficulty, count, split)
        write_course(course, composite_dir, name)
        composites.append((name, course))
    save_composite_preview(composites, ASSET_DIR / "composite_course_catalog.png")

    summary = {
        "templates": len(CATALOG),
        "single_obstacle_courses": len(CATALOG),
        "composite_courses": len(composites),
        "maximum_candidate_height": 2,
        "minimum_gap_voxels": 15,
        "empirically_verified_templates": [
            item.name for item in CATALOG.values() if item.empirical_status == "teacher_verified"
        ],
        "geometry_screened_templates": [
            item.name for item in CATALOG.values() if item.empirical_status != "teacher_verified"
        ],
    }
    (ASSET_DIR / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
