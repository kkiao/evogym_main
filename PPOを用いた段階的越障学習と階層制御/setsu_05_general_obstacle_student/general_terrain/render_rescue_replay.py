"""M2救援回を同一設定で再実行し、代表的なGIFを生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from general_terrain.curriculum import get_curriculum_stage, sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.interactive_collection import collect_rescue_episode
from general_terrain.seed_manifest import DEFAULT_SEED_MANIFEST, load_seed_manifest
from general_terrain.rescue_profiles import (
    M2_DEFAULT_PROFILE,
    RESCUE_PROFILE_NAMES,
    get_rescue_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = PROJECT_ROOT / "artifacts" / "m2_rescue_replays"


def build_argument_parser() -> argparse.ArgumentParser:
    """凍結訓練種の代表回だけを描画する引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="M2救援回を再現し、監査用GIFを生成する。"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed-manifest", default=str(DEFAULT_SEED_MANIFEST))
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument(
        "--rescue-profile",
        choices=RESCUE_PROFILE_NAMES,
        default=M2_DEFAULT_PROFILE,
    )
    return parser


def main() -> None:
    """重みを更新せず一つの救援回を描画して監査要約を保存する。"""
    from sb3_contrib import RecurrentPPO

    from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher

    args = build_argument_parser().parse_args()
    manifest = load_seed_manifest(Path(args.seed_manifest))
    if args.seed not in manifest.for_split("train"):
        raise ValueError("描画乱数種は凍結訓練集合に含まれなければならない。")
    stage = get_curriculum_stage(manifest.stage)
    rescue_config = get_rescue_profile(args.rescue_profile)
    course = sample_curriculum_course(args.seed, stage.name, "train")
    output_dir = REPLAY_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array",
    )
    try:
        student = RecurrentPPO.load(Path(args.model), device="cpu")
        teacher = PortfolioHeight1Teacher()
        result = collect_rescue_episode(
            environment,
            student,
            teacher,
            seed=args.seed,
            output_path=output_dir / f"seed_{args.seed}_rescued.npz",
            output_gif=output_dir / f"seed_{args.seed}_rescue.gif",
            frame_interval=args.frame_interval,
            rescue_config=rescue_config,
            metadata={
                "purpose": "m2_representative_replay",
                "rescue_profile": args.rescue_profile,
            },
        )
    finally:
        environment.close()
    summary = {
        "student_weights_updated": False,
        "teacher_is_training_only": True,
        "rescue_profile": args.rescue_profile,
        "seed_manifest": manifest.as_dict(),
        "student_model": str(Path(args.model).resolve()),
        "episode": result.as_dict(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
