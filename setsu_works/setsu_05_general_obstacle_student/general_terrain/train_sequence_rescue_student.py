"""安全教師分岐を時系列のまま学習する軽量LSTM学生を訓練する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.train_noisy_height1_teacher import POSITIONS, course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRANCH_RUN = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_teacher_search"
    / "noisy_teacher_portfolio_seed7_v1"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "height1_sequence_rescue_student"


class SequenceStudent(nn.Module):
    """九十五次元観測と内部履歴から六動作を返す学生ネットワーク。"""

    def __init__(self, hidden_size: int = 192) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(95, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        self.memory = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 6), nn.Tanh())

    def forward(self, observations, state=None):
        """一括または一歩観測を処理し動作と次の記憶を返す。"""
        encoded = self.encoder(observations)
        memory, state = self.memory(encoded, state)
        return self.head(memory), state


def load_sequences(path: Path) -> tuple[list[dict[str, np.ndarray]], dict[str, object]]:
    """安全完走した位置別分岐を段階情報付きで読み込む。"""
    sequences = []
    rows = []
    for branch_path in sorted((path / "branches").glob("x*_rescued.npz")):
        data = np.load(branch_path)
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        stages = np.asarray(data["stages"]).astype(str)
        weights = np.asarray(
            [1.0 if stage == "flat" else 3.0 for stage in stages],
            dtype=np.float32,
        )
        sequences.append(
            {
                "observations": observations,
                "actions": actions,
                "weights": weights,
            }
        )
        rows.append(
            {
                "file": branch_path.name,
                "steps": len(observations),
                "weighted_steps": float(np.sum(weights)),
            }
        )
    if not sequences:
        raise RuntimeError(f"時系列救回分岐が見つからない：{path}")
    return sequences, {"branches": rows, "sequence_count": len(sequences)}


def pad_sequences(
    sequences: list[dict[str, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """長さの異なる軌跡を損失マスク付きの一括配列へ変換する。"""
    maximum_steps = max(len(item["observations"]) for item in sequences)
    observations = np.zeros((len(sequences), maximum_steps, 95), dtype=np.float32)
    actions = np.zeros((len(sequences), maximum_steps, 6), dtype=np.float32)
    weights = np.zeros((len(sequences), maximum_steps), dtype=np.float32)
    for index, item in enumerate(sequences):
        steps = len(item["observations"])
        observations[index, :steps] = item["observations"]
        actions[index, :steps] = item["actions"]
        weights[index, :steps] = item["weights"]
    return (
        torch.from_numpy(observations),
        torch.from_numpy(actions),
        torch.from_numpy(weights),
    )


def evaluate(
    model: SequenceStudent,
    *,
    observation_mean: np.ndarray,
    observation_scale: np.ndarray,
    seed: int,
    noise_std: float,
    noise_probability: float,
) -> dict[str, object]:
    """十一位置を独立記憶と固定動作摂動で厳格評価する。"""
    episodes = []
    model.eval()
    for index, position in enumerate(POSITIONS):
        episode_seed = seed + index
        environment = GeneralObstacleEnv(
            course=course(position, episode_seed, "sequence_student_validation"),
            resample_on_reset=False,
        )
        rng = np.random.default_rng(episode_seed + 5_000_000)
        state = None
        disturbance_count = 0
        maximum_angle = 0.0
        upper_contact_steps = 0
        try:
            observation, info = environment.reset(seed=episode_seed)
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                normalized = (observation - observation_mean) / observation_scale
                tensor = torch.as_tensor(
                    normalized[None, None, :],
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    prediction, state = model(tensor, state)
                action = prediction[0, 0].cpu().numpy().astype(np.float32)
                if rng.random() < noise_probability:
                    action = np.clip(
                        action + rng.normal(0.0, noise_std, size=action.shape),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                    disturbance_count += 1
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                maximum_angle = max(maximum_angle, float(info["orientation_error"]))
                upper_contact_steps += int(bool(info["upper_body_grounded"]))
        finally:
            environment.close()
        episodes.append(
            {
                "position": position,
                "steps": steps,
                "disturbance_count": disturbance_count,
                "course_complete": bool(info["course_complete"]),
                "raw_clearances": int(info["raw_clearances"]),
                "recovered_obstacles": int(info["recovered_obstacles"]),
                "hard_fall": bool(info["hard_fall"]),
                "failure_reason": str(info["failure_reason"]),
                "maximum_angle_degrees": math.degrees(maximum_angle),
                "upper_body_contact_steps": upper_contact_steps,
                "maximum_com_x": float(info["max_x_position"]),
            }
        )
    return {
        "episodes": episodes,
        "success_count": sum(item["course_complete"] for item in episodes),
        "clearance_count": sum(item["raw_clearances"] >= 1 for item in episodes),
        "hard_fall_count": sum(item["hard_fall"] for item in episodes),
        "mean_max_x": float(np.mean([item["maximum_com_x"] for item in episodes])),
    }


def score(result: dict[str, object]) -> tuple[int, int, int, float]:
    """完走、安全、越壁、距離の順で最良時点を決める。"""
    return (
        int(result["success_count"]),
        -int(result["hard_fall_count"]),
        int(result["clearance_count"]),
        float(result["mean_max_x"]),
    )


def append_progress(path: Path, row: dict[str, object]) -> None:
    """一定学習回数ごとの評価をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """段階重み付き時系列模倣を行い高さ一門を判定する。"""
    parser = argparse.ArgumentParser(description="训练带时序记忆的救回分支学生。")
    parser.add_argument("--run-name", default="sequence_rescue_student_seed7_v1")
    parser.add_argument("--branch-run", default=str(DEFAULT_BRANCH_RUN))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--evaluate-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    sequences, dataset_metadata = load_sequences(Path(args.branch_run))
    all_observations = np.concatenate([item["observations"] for item in sequences])
    observation_mean = np.mean(all_observations, axis=0).astype(np.float32)
    observation_scale = np.maximum(
        np.std(all_observations, axis=0),
        1e-3,
    ).astype(np.float32)
    observations, actions, weights = pad_sequences(sequences)
    normalized_observations = (
        observations - torch.from_numpy(observation_mean)[None, None, :]
    ) / torch.from_numpy(observation_scale)[None, None, :]
    model = SequenceStudent(hidden_size=args.hidden_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-6,
    )
    best_score = (-1, -11, -1, -1.0)
    best_epoch = 0
    best_result = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        noisy_observations = normalized_observations + 0.003 * torch.randn_like(
            normalized_observations
        )
        predictions, _ = model(noisy_observations)
        squared_error = torch.mean((predictions - actions) ** 2, dim=-1)
        loss = torch.sum(squared_error * weights) / torch.sum(weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch % args.evaluate_every != 0 and epoch != args.epochs:
            continue
        result = evaluate(
            model,
            observation_mean=observation_mean,
            observation_scale=observation_scale,
            seed=args.seed + 70_000,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        )
        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "success_count": result["success_count"],
            "clearance_count": result["clearance_count"],
            "hard_fall_count": result["hard_fall_count"],
            "mean_max_x": result["mean_max_x"],
        }
        history.append({**row, "evaluation": result})
        append_progress(output_dir / "progress.csv", row)
        torch.save(model.state_dict(), output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        candidate_score = score(result)
        if candidate_score > best_score:
            best_score = candidate_score
            best_epoch = epoch
            best_result = result
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if result["success_count"] >= 9 and result["hard_fall_count"] == 0:
            break
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))
    held_out = evaluate(
        model,
        observation_mean=observation_mean,
        observation_scale=observation_scale,
        seed=args.seed + 80_000,
        noise_std=args.noise_std,
        noise_probability=args.noise_probability,
    )
    np.savez_compressed(
        output_dir / "normalization.npz",
        mean=observation_mean,
        scale=observation_scale,
    )
    summary = {
        "method": "weighted_sequence_rescue_imitation",
        "observation_dimension": 95,
        "action_dimension": 6,
        "privileged_student_inputs": False,
        "hidden_size": args.hidden_size,
        "dataset": dataset_metadata,
        "best_epoch": best_epoch,
        "best_score": list(best_score),
        "fixed_gate_evaluation": best_result,
        "held_out_evaluation": held_out,
        "robustness_gate_passed": bool(
            best_result is not None
            and best_result["success_count"] >= 9
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
