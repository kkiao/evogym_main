"""循環学生を実環境で動かし、訓練専用教師の成功救援を収集する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import imageio.v2 as imageio
import numpy as np

from general_terrain.curriculum import (
    CurriculumStage,
    get_curriculum_stage,
    sample_curriculum_course,
)
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.interactive_rescue import (
    InteractiveRescueController,
    RescueConfig,
    SuccessfulRescueBuffer,
    local_terrain_is_visible,
)
from general_terrain.seed_manifest import DEFAULT_SEED_MANIFEST, load_seed_manifest
from general_terrain.rescue_profiles import (
    M2_DEFAULT_PROFILE,
    RESCUE_PROFILE_NAMES,
    get_rescue_profile,
)
from general_terrain.terrain import CourseSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = PROJECT_ROOT / "runs" / "interactive_rescue_collection"


class RecurrentStudent(Protocol):
    """循環学生に必要な予測インターフェースを定義する。"""

    def predict(
        self,
        observation: np.ndarray,
        *,
        state: Any,
        episode_start: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.ndarray, Any]:
        """現在観測と循環状態から動作と次状態を返す。"""


class TrainingTeacher(Protocol):
    """収集時だけ許可される教師インターフェースを定義する。"""

    def reset(self, environment: Any) -> None:
        """一回分の教師内部状態を初期化する。"""

    def predict(
        self,
        environment: Any,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> tuple[np.ndarray, str]:
        """教師ラベル動作と監査用段階名を返す。"""


@dataclass(frozen=True)
class CollectionEpisodeResult:
    """一回分の救援収集結果と保存可否を保持する。"""

    seed: int
    course_id: str
    start_runway_voxels: int
    steps: int
    course_complete: bool
    hard_fall: bool
    failure_reason: str
    raw_clearances: int
    recovered_obstacles: int
    rescue_count: int
    teacher_control_steps: int
    student_control_steps: int
    branch_accepted: bool
    branch_path: str | None
    gif_path: str | None
    events: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存可能な辞書形式を返す。"""
        return asdict(self)


def collect_rescue_episode(
    environment: Any,
    student: RecurrentStudent,
    teacher: TrainingTeacher,
    *,
    seed: int,
    output_path: Path,
    rescue_config: RescueConfig = RescueConfig(),
    metadata: Mapping[str, object] | None = None,
    output_gif: Path | None = None,
    frame_interval: int = 5,
) -> CollectionEpisodeResult:
    """学生履歴を途切れさせず一回分の連続教師救援を収集する。"""
    if frame_interval < 1:
        raise ValueError("描画間隔は一歩以上でなければならない。")
    observation, info = environment.reset(seed=seed)
    teacher.reset(environment)
    rescue = InteractiveRescueController(rescue_config)
    buffer = SuccessfulRescueBuffer()
    recurrent_state: Any = None
    episode_start = np.ones((1,), dtype=bool)
    terminated = False
    truncated = False
    steps = 0
    events: list[dict[str, object]] = []
    frames: list[np.ndarray] = []
    timeout_events: set[int] = set()
    schema = tuple(environment.unwrapped.schema)

    while not (terminated or truncated):
        student_action, recurrent_state = student.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        teacher_action, teacher_stage = teacher.predict(
            environment,
            observation,
            info,
        )
        decision = rescue.decide(
            info,
            np.asarray(student_action, dtype=np.float32),
            np.asarray(teacher_action, dtype=np.float32),
            local_terrain_visible=local_terrain_is_visible(
                observation,
                schema,
                maximum_rise_offset=(
                    rescue_config.disagreement_maximum_rise_offset
                ),
            ),
        )
        if decision.event == "start":
            start_hook = getattr(teacher, "on_rescue_start", None)
            if callable(start_hook):
                teacher_action, teacher_stage = start_hook(
                    environment,
                    observation,
                    info,
                )
        executed_action = teacher_action if decision.use_teacher else student_action
        buffer.append(
            observation,
            student_action,
            teacher_action,
            executed_action,
            decision,
            teacher_stage,
        )
        if decision.event in {"start", "release"}:
            events.append(
                {
                    "step": steps,
                    "event": decision.event,
                    "reason": decision.reason,
                    "rescue_id": decision.rescue_id,
                    "teacher_stage": teacher_stage,
                    "x_position": float(info["x_position"]),
                    "orientation_error": float(info["orientation_error"]),
                    "angular_velocity": float(info["angular_velocity"]),
                    "stall_steps": int(info["stall_steps"]),
                    "raw_clearances": int(info["raw_clearances"]),
                    "recovered_obstacles": int(info["recovered_obstacles"]),
                    "upper_body_grounded": bool(info["upper_body_grounded"]),
                }
            )
        if decision.timed_out and decision.rescue_id not in timeout_events:
            timeout_events.add(decision.rescue_id)
            events.append(
                {
                    "step": steps,
                    "event": "teacher_timeout",
                    "reason": decision.reason,
                    "rescue_id": decision.rescue_id,
                }
            )
        if decision.event == "release":
            release_hook = getattr(teacher, "on_rescue_release", None)
            if callable(release_hook):
                release_hook(environment)
        observation, _, terminated, truncated, info = environment.step(
            np.asarray(executed_action, dtype=np.float32)
        )
        episode_start[:] = False
        steps += 1
        if output_gif is not None and steps % frame_interval == 0:
            frame = environment.render()
            if frame is not None:
                frames.append(np.asarray(frame))

    if output_gif is not None and frames:
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, frames, fps=12, loop=0)

    accepted = buffer.commit(
        output_path,
        info,
        metadata={
            "seed": seed,
            "rescue_config": asdict(rescue_config),
            **dict(metadata or {}),
        },
    )
    teacher_steps = buffer.teacher_steps
    return CollectionEpisodeResult(
        seed=seed,
        course_id=str(info["course_id"]),
        start_runway_voxels=int(environment.unwrapped.course.obstacles[0].start_x),
        steps=steps,
        course_complete=bool(info["course_complete"]),
        hard_fall=bool(info["hard_fall"]),
        failure_reason=str(info["failure_reason"]),
        raw_clearances=int(info["raw_clearances"]),
        recovered_obstacles=int(info["recovered_obstacles"]),
        rescue_count=rescue.rescue_id,
        teacher_control_steps=teacher_steps,
        student_control_steps=steps - teacher_steps,
        branch_accepted=accepted,
        branch_path=str(output_path.resolve()) if accepted else None,
        gif_path=(
            str(output_gif.resolve())
            if output_gif is not None and output_gif.exists()
            else None
        ),
        events=tuple(events),
    )


def collect_rescue_batch(
    student: RecurrentStudent,
    teacher: TrainingTeacher,
    *,
    seeds: tuple[int, ...],
    stage: CurriculumStage,
    output_dir: Path,
    rescue_config: RescueConfig = RescueConfig(),
    rescue_profile_name: str = "custom",
    environment_factory: Callable[[CourseSpec], Any] | None = None,
) -> dict[str, object]:
    """固定訓練乱数種を巡回し、成功救援だけを分岐集合へ保存する。"""
    if not stage.teacher_ready:
        raise RuntimeError(f"教師門限未達の段階は収集できない: {stage.name}")
    factory = environment_factory or (
        lambda course: GeneralObstacleEnv(course=course, resample_on_reset=False)
    )
    branches_dir = output_dir / "branches"
    branches_dir.mkdir(parents=True, exist_ok=False)
    episodes: list[CollectionEpisodeResult] = []
    for seed in seeds:
        course = sample_curriculum_course(seed, stage.name, "train")
        environment = factory(course)
        output_path = branches_dir / (
            f"seed_{seed}_x{course.obstacles[0].start_x}_rescued.npz"
        )
        try:
            episode = collect_rescue_episode(
                environment,
                student,
                teacher,
                seed=seed,
                output_path=output_path,
                rescue_config=rescue_config,
                metadata={"stage": stage.name, "split": "train"},
            )
            episodes.append(episode)
        finally:
            environment.close()
    serialized = [episode.as_dict() for episode in episodes]
    return {
        "method": "continuous_interactive_teacher_rescue_collection",
        "controller_mode": "student_with_training_rescue",
        "stage": stage.name,
        "split": "train",
        "student_weights_updated": False,
        "teacher_is_training_only": True,
        "evaluation_result": False,
        "rescue_profile": rescue_profile_name,
        "rescue_config": asdict(rescue_config),
        "episode_count": len(episodes),
        "collection_success_count": sum(
            episode.course_complete for episode in episodes
        ),
        "hard_fall_count": sum(episode.hard_fall for episode in episodes),
        "accepted_branch_count": sum(
            episode.branch_accepted for episode in episodes
        ),
        "teacher_interventions": sum(episode.rescue_count for episode in episodes),
        "teacher_control_steps": sum(
            episode.teacher_control_steps for episode in episodes
        ),
        "student_control_steps": sum(
            episode.student_control_steps for episode in episodes
        ),
        "episodes": serialized,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """正式訓練を行わない救援収集専用引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="循環学生の訪問状態から訓練専用教師救援を収集する。"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", default="hurdle_single")
    parser.add_argument("--seed-manifest", default=str(DEFAULT_SEED_MANIFEST))
    parser.add_argument(
        "--rescue-profile",
        choices=RESCUE_PROFILE_NAMES,
        default=M2_DEFAULT_PROFILE,
    )
    return parser


def main() -> None:
    """凍結訓練種で一回の救援収集を実行し、重みを更新せず終了する。"""
    from sb3_contrib import RecurrentPPO

    from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher

    args = build_argument_parser().parse_args()
    manifest = load_seed_manifest(Path(args.seed_manifest))
    if manifest.stage != args.stage:
        raise ValueError("乱数種目録と収集段階が一致しない。")
    stage = get_curriculum_stage(args.stage)
    output_dir = COLLECTION_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    student = RecurrentPPO.load(Path(args.model), device="cpu")
    teacher = PortfolioHeight1Teacher()
    rescue_config = get_rescue_profile(args.rescue_profile)
    summary = collect_rescue_batch(
        student,
        teacher,
        seeds=manifest.for_split("train"),
        stage=stage,
        output_dir=output_dir,
        rescue_config=rescue_config,
        rescue_profile_name=args.rescue_profile,
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
