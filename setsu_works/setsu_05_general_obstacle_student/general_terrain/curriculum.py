"""障害物技能を分離した段階的ランダムカリキュラムを定義する。"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

from general_terrain.feasibility import assert_course_feasible
from general_terrain.terrain import (
    MAX_GAP_VOXELS,
    MAX_START_RUNWAY_VOXELS,
    MIN_GAP_VOXELS,
    START_RUNWAY_VOXELS,
    CourseSpec,
    build_course,
)


TEACHER_VERIFIED = "verified"
TEACHER_BELOW_GATE = "below_gate"
TEACHER_MISSING = "missing"


@dataclass(frozen=True)
class CurriculumGate:
    """一段階を昇格させるための固定評価条件を保持する。"""

    evaluation_episodes: int = 11
    minimum_successes: int = 9
    maximum_hard_falls: int = 0

    def evaluate(self, metrics: Mapping[str, object]) -> "GateResult":
        """評価集計を門限と比較し、不合格理由も返す。"""
        episodes = int(metrics.get("evaluation_episodes", self.evaluation_episodes))
        successes = int(metrics["success_count"])
        hard_falls = int(metrics["hard_fall_count"])
        reasons = []
        if episodes != self.evaluation_episodes:
            reasons.append(
                f"評価回数が{self.evaluation_episodes}ではなく{episodes}である"
            )
        if successes < self.minimum_successes:
            reasons.append(
                f"完走数が{self.minimum_successes}未満である: {successes}"
            )
        if hard_falls > self.maximum_hard_falls:
            reasons.append(
                f"硬転倒数が{self.maximum_hard_falls}を超える: {hard_falls}"
            )
        return GateResult(passed=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class GateResult:
    """カリキュラム門限の合否と監査可能な理由を保持する。"""

    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CurriculumStage:
    """同一技能内だけをランダム化する一つのカリキュラム段階を表す。"""

    name: str
    order: int
    template_sequences: tuple[tuple[str, ...], ...]
    prerequisite: str | None
    teacher_status: str
    gate: CurriculumGate = CurriculumGate()

    @property
    def obstacle_count(self) -> int:
        """この段階で常に現れる障害物数を返す。"""
        counts = {len(sequence) for sequence in self.template_sequences}
        if len(counts) != 1:
            raise ValueError(f"段階内の障害物数が一致しない: {self.name}")
        return counts.pop()

    @property
    def teacher_ready(self) -> bool:
        """教師が所定門限を実証済みかを返す。"""
        return self.teacher_status == TEACHER_VERIFIED


CURRICULUM_STAGES = (
    CurriculumStage(
        name="hurdle_single",
        order=1,
        template_sequences=(("low_hurdle",),),
        prerequisite=None,
        teacher_status=TEACHER_VERIFIED,
    ),
    CurriculumStage(
        name="hurdle_double",
        order=2,
        template_sequences=(("low_hurdle", "low_hurdle"),),
        prerequisite="hurdle_single",
        teacher_status=TEACHER_BELOW_GATE,
    ),
    CurriculumStage(
        name="platform_single",
        order=3,
        template_sequences=(("low_platform_short",),),
        prerequisite="hurdle_double",
        teacher_status=TEACHER_MISSING,
    ),
    CurriculumStage(
        name="platform_double",
        order=4,
        template_sequences=(("low_platform_short", "low_platform_short"),),
        prerequisite="platform_single",
        teacher_status=TEACHER_MISSING,
    ),
    CurriculumStage(
        name="hurdle_platform_mixed",
        order=5,
        template_sequences=(
            ("low_hurdle", "low_platform_short"),
            ("low_platform_short", "low_hurdle"),
        ),
        prerequisite="platform_double",
        teacher_status=TEACHER_MISSING,
    ),
)

STAGE_BY_NAME = {stage.name: stage for stage in CURRICULUM_STAGES}


def get_curriculum_stage(name: str) -> CurriculumStage:
    """名前からカリキュラム段階を取得し、未知名は明示的に拒否する。"""
    try:
        return STAGE_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"未知のカリキュラム段階: {name}") from error


def sample_curriculum_course(
    seed: int,
    stage_name: str,
    split: str,
) -> CourseSpec:
    """技能種類を混ぜず位置と間隔だけを再現可能に乱数化する。"""
    stage = get_curriculum_stage(stage_name)
    rng = random.Random(seed)
    sequence = rng.choice(stage.template_sequences)
    gaps = [
        rng.randint(MIN_GAP_VOXELS, MAX_GAP_VOXELS)
        for _ in range(len(sequence) - 1)
    ]
    start_runway = rng.randint(START_RUNWAY_VOXELS, MAX_START_RUNWAY_VOXELS)
    course = build_course(
        sequence,
        split=f"{split}_{stage.name}",
        seed=seed,
        difficulty=1,
        gaps=gaps,
        start_runway_voxels=start_runway,
    )
    assert_course_feasible(course)
    return course
