"""保存済み回復方策群を同一条件で比較し教師候補の被覆率を調べる。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from general_terrain.train_recovery_teacher import POSITIONS, evaluate_recovery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "recovery_runs"


def discover_models() -> list[tuple[str, Path, float]]:
    """利用可能な最良モデルと対応する前綴終了率を列挙する。"""
    candidates = []
    for run_dir in sorted(RUNS_ROOT.iterdir()):
        model_path = run_dir / "best_model.zip"
        if not model_path.exists():
            continue
        prefix_fraction = 0.5 if "fraction50" in run_dir.name else 1.0
        candidates.append((run_dir.name, model_path, prefix_fraction))
    return candidates


def main() -> None:
    """各候補を全十一位置で検査し方策集合の合併被覆も保存する。"""
    parser = argparse.ArgumentParser(description="比较全部已保存的落地恢复候选。")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "training_only_teacher" / "generated" / "recovery_portfolio_audit.json"),
    )
    parser.add_argument("--seed", type=int, default=81_000)
    args = parser.parse_args()

    rows = []
    covered: dict[int, list[str]] = {position: [] for position in POSITIONS}
    for name, model_path, prefix_fraction in discover_models():
        model = PPO.load(model_path, device="cpu")
        result = evaluate_recovery(
            model,
            seed=args.seed,
            positions=POSITIONS,
            prefix_fraction=prefix_fraction,
        )
        successful_positions = [
            int(episode["position"])
            for episode in result["episodes"]
            if episode["success"]
        ]
        for position in successful_positions:
            covered[position].append(name)
        row = {
            "run": name,
            "model": str(model_path.resolve()),
            "prefix_fraction": prefix_fraction,
            "success_rate": result["success_rate"],
            "hard_fall_rate": result["hard_fall_rate"],
            "successful_positions": successful_positions,
            "episodes": result["episodes"],
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "run": name,
                    "prefix_fraction": prefix_fraction,
                    "success": successful_positions,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = {
        "positions": list(POSITIONS),
        "models": rows,
        "coverage": {str(key): value for key, value in covered.items()},
        "covered_positions": [key for key, value in covered.items() if value],
        "uncovered_positions": [key for key, value in covered.items() if not value],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["uncovered_positions"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
