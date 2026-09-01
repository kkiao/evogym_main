"""教師を生成も参照もせず循環学生だけを固定検証集合で評価する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from general_terrain.curriculum import (
    CurriculumStage,
    get_curriculum_stage,
    sample_curriculum_course,
)
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.seed_manifest import DEFAULT_SEED_MANIFEST, load_seed_manifest
from general_terrain.terrain import CourseSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "runs" / "student_only_evaluation"


class RecurrentStudent(Protocol):
    """教師なし評価に必要な循環学生予測契約を定義する。"""

    def predict(
        self,
        observation: np.ndarray,
        *,
        state: Any,
        episode_start: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.ndarray, Any]:
        """学生動作と次の循環状態を返す。"""


@dataclass(frozen=True)
class StudentEpisodeResult:
    """教師なし一回評価の厳格検収値を保持する。"""

    seed: int
    course_id: str
    start_runway_voxels: int
    steps: int
    course_complete: bool
    hard_fall: bool
    failure_reason: str
    raw_clearances: int
    recovered_obstacles: int
    maximum_angle_degrees: float
    upper_body_contact_steps: int
    maximum_com_x: float
    teacher_interventions: int = 0

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存可能な辞書形式を返す。"""
        return asdict(self)


def evaluate_student_episode(
    environment: Any,
    student: RecurrentStudent,
    *,
    seed: int,
) -> StudentEpisodeResult:
    """外部制御器を受け取らず学生だけで一回を最後まで実行する。"""
    observation, info = environment.reset(seed=seed)
    recurrent_state: Any = None
    episode_start = np.ones((1,), dtype=bool)
    terminated = False
    truncated = False
    steps = 0
    maximum_angle = 0.0
    upper_body_contact_steps = 0
    while not (terminated or truncated):
        action, recurrent_state = student.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        observation, _, terminated, truncated, info = environment.step(
            np.asarray(action, dtype=np.float32)
        )
        episode_start[:] = False
        steps += 1
        maximum_angle = max(maximum_angle, float(info["orientation_error"]))
        upper_body_contact_steps += int(bool(info["upper_body_grounded"]))
    return StudentEpisodeResult(
        seed=seed,
        course_id=str(info["course_id"]),
        start_runway_voxels=int(environment.unwrapped.course.obstacles[0].start_x),
        steps=steps,
        course_complete=bool(info["course_complete"]),
        hard_fall=bool(info["hard_fall"]),
        failure_reason=str(info["failure_reason"]),
        raw_clearances=int(info["raw_clearances"]),
        recovered_obstacles=int(info["recovered_obstacles"]),
        maximum_angle_degrees=math.degrees(maximum_angle),
        upper_body_contact_steps=upper_body_contact_steps,
        maximum_com_x=float(info["max_x_position"]),
    )


def evaluate_student_batch(
    student: RecurrentStudent,
    *,
    seeds: tuple[int, ...],
    stage: CurriculumStage,
    split: str = "validation",
    environment_factory: Callable[[CourseSpec], Any] | None = None,
) -> dict[str, object]:
    """固定検証乱数種上の学生単独成績を集計する。"""
    if split not in {"train", "validation", "holdout"}:
        raise ValueError("学生単独評価の区分が未知である。")
    factory = environment_factory or (
        lambda course: GeneralObstacleEnv(course=course, resample_on_reset=False)
    )
    episodes: list[StudentEpisodeResult] = []
    for seed in seeds:
        course = sample_curriculum_course(seed, stage.name, split)
        environment = factory(course)
        try:
            episodes.append(
                evaluate_student_episode(environment, student, seed=seed)
            )
        finally:
            environment.close()
    serialized = [episode.as_dict() for episode in episodes]
    return {
        "method": "recurrent_student_only_evaluation",
        "controller_mode": "student_only",
        "stage": stage.name,
        "split": split,
        "evaluation_episodes": len(episodes),
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
        "success_count": sum(episode.course_complete for episode in episodes),
        "hard_fall_count": sum(episode.hard_fall for episode in episodes),
        "mean_raw_clearances": float(
            np.mean([episode.raw_clearances for episode in episodes])
        ),
        "mean_recovered_obstacles": float(
            np.mean([episode.recovered_obstacles for episode in episodes])
        ),
        "mean_max_x": float(np.mean([episode.maximum_com_x for episode in episodes])),
        "episodes": serialized,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """学生単独検証だけに必要な引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="凍結検証集合で循環学生だけを評価する。"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", default="hurdle_single")
    parser.add_argument("--seed-manifest", default=str(DEFAULT_SEED_MANIFEST))
    return parser


def main() -> None:
    """教師経路を持たない入口から固定検証評価を実行する。"""
    from sb3_contrib import RecurrentPPO

    args = build_argument_parser().parse_args()
    manifest = load_seed_manifest(Path(args.seed_manifest))
    if manifest.stage != args.stage:
        raise ValueError("乱数種目録と評価段階が一致しない。")
    stage = get_curriculum_stage(args.stage)
    output_dir = EVALUATION_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    student = RecurrentPPO.load(Path(args.model), device="cpu")
    summary = evaluate_student_batch(
        student,
        seeds=manifest.for_split("validation"),
        stage=stage,
    )
    summary["seed_manifest"] = manifest.as_dict()
    summary["student_model"] = str(Path(args.model).resolve())
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
