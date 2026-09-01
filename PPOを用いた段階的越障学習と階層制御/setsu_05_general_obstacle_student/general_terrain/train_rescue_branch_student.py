"""安全救回分岐を集約し九十五次元の単一学生方策へ保守的に移す。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from general_terrain.train_noisy_height1_teacher import (
    DEFAULT_FLAT_MODEL,
    collect_flat_anchor_dataset,
    evaluate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL_MODEL = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_height1_runs"
    / "noisy_height1_demo_seed7_v3"
    / "best_model.zip"
)
DEFAULT_BRANCH_RUN = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_teacher_search"
    / "noisy_teacher_portfolio_seed7_v1"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "height1_rescue_branch_student"


def load_rescue_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """教師探索で安全完走した全分岐を一つの教師データへ連結する。"""
    observations = []
    actions = []
    rows = []
    for branch_path in sorted((path / "branches").glob("x*_rescued.npz")):
        data = np.load(branch_path)
        branch_observations = np.asarray(data["observations"], dtype=np.float32)
        branch_actions = np.asarray(data["actions"], dtype=np.float32)
        observations.append(branch_observations)
        actions.append(branch_actions)
        rows.append({"file": branch_path.name, "rows": len(branch_observations)})
    if not observations:
        raise RuntimeError(f"安全救回分岐が見つからない：{path}")
    return (
        np.concatenate(observations),
        np.concatenate(actions),
        {"branches": rows, "rows": sum(item["rows"] for item in rows)},
    )


def one_epoch(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    rng: np.random.Generator,
    learning_rate: float,
) -> float:
    """一回だけ順序を混ぜ、方策平均を教師動作へ保守的に近づける。"""
    parameters = list(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-6)
    losses = []
    model.policy.train()
    permutation = rng.permutation(len(observations))
    for start in range(0, len(observations), 256):
        indices = permutation[start : start + 256]
        batch_observations = torch.as_tensor(
            observations[indices],
            dtype=torch.float32,
            device=model.device,
        )
        batch_actions = torch.as_tensor(
            actions[indices],
            dtype=torch.float32,
            device=model.device,
        )
        latent = model.policy.mlp_extractor.forward_actor(batch_observations)
        predictions = model.policy.action_net(latent)
        loss = torch.mean((predictions - batch_actions) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def append_progress(path: Path, row: dict[str, object]) -> None:
    """各模倣時点の高さ一厳格評価をCSVへ保存する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def score(result: dict[str, object]) -> tuple[int, int, int, float]:
    """完走数、安全、越壁数、距離の順に候補を比較する。"""
    return (
        int(result["success_count"]),
        -int(result["hard_fall_count"]),
        int(result["clearance_count"]),
        float(result["mean_max_x"]),
    )


def main() -> None:
    """救回分岐で学生を更新し九成功零側倒の門を自動判定する。"""
    parser = argparse.ArgumentParser(description="用教师救回分支训练统一95维学生。")
    parser.add_argument("--run-name", default="rescue_branch_student_seed7_v1")
    parser.add_argument("--initial-model", default=str(DEFAULT_INITIAL_MODEL))
    parser.add_argument("--flat-model", default=str(DEFAULT_FLAT_MODEL))
    parser.add_argument("--branch-run", default=str(DEFAULT_BRANCH_RUN))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--evaluate-every", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--branch-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    model = PPO.load(Path(args.initial_model), device="cpu")
    flat_model = PPO.load(Path(args.flat_model), device="cpu")
    flat_observations, flat_actions = collect_flat_anchor_dataset(
        flat_model,
        seed=args.seed + 10_000,
    )
    branch_observations, branch_actions, branch_metadata = load_rescue_dataset(
        Path(args.branch_run)
    )
    observations = np.concatenate(
        (
            flat_observations,
            np.repeat(branch_observations, args.branch_repeat, axis=0),
        )
    )
    actions = np.concatenate(
        (
            flat_actions,
            np.repeat(branch_actions, args.branch_repeat, axis=0),
        )
    )
    metadata = {
        "initial_model": str(Path(args.initial_model).resolve()),
        "flat_anchor_rows": len(flat_observations),
        "rescue_dataset": branch_metadata,
        "branch_repeat": args.branch_repeat,
        "training_rows": len(observations),
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rng = np.random.default_rng(args.seed)
    initial = evaluate(
        model,
        seed=args.seed + 70_000,
        noise_std=args.noise_std,
        noise_probability=args.noise_probability,
    )
    best_score = score(initial)
    best_epoch = 0
    best_result = initial
    model.save(output_dir / "best_model")
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = one_epoch(
            model,
            observations,
            actions,
            rng=rng,
            learning_rate=args.learning_rate,
        )
        if epoch % args.evaluate_every != 0 and epoch != args.epochs:
            continue
        result = evaluate(
            model,
            seed=args.seed + 70_000,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        )
        row = {
            "epoch": epoch,
            "loss": loss,
            "success_count": result["success_count"],
            "clearance_count": result["clearance_count"],
            "hard_fall_count": result["hard_fall_count"],
            "mean_max_x": result["mean_max_x"],
        }
        history.append({**row, "evaluation": result})
        append_progress(output_dir / "progress.csv", row)
        model.save(output_dir / "checkpoints" / f"epoch_{epoch:02d}")
        candidate_score = score(result)
        if candidate_score > best_score:
            best_score = candidate_score
            best_epoch = epoch
            best_result = result
            model.save(output_dir / "best_model")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if result["success_count"] >= 9 and result["hard_fall_count"] == 0:
            break
    best_model = PPO.load(output_dir / "best_model.zip", device="cpu")
    held_out = evaluate(
        best_model,
        seed=args.seed + 80_000,
        noise_std=args.noise_std,
        noise_probability=args.noise_probability,
    )
    summary = {
        "method": "conservative_rescue_branch_imitation",
        "observation_dimension": 95,
        "privileged_student_inputs": False,
        "best_epoch": best_epoch,
        "best_score": list(best_score),
        "fixed_gate_evaluation": best_result,
        "held_out_evaluation": held_out,
        "robustness_gate_passed": bool(
            best_result["success_count"] >= 9
            and best_result["hard_fall_count"] == 0
        ),
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
