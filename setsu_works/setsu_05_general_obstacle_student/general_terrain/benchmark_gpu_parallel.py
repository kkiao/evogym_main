"""EvoGymのCPU並列実行とGPU方策更新を同一条件で計測する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from general_terrain.environment import GeneralObstacleEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "performance_benchmark"
PROTECTED_STUDENT = (
    PROJECT_ROOT
    / "runs"
    / "height1_recurrent_dagger_student"
    / "height1_recurrent_dagger_seed7_v1"
    / "best_model.zip"
)
EXPECTED_PROTECTED_STUDENT_SHA256 = (
    "8ad4c30c52c96aad00b4c0898da785522a5f4acca2443fdfce0afd3adf2f78fd"
)
ROLLOUT_TRANSITIONS = 512


@dataclass(frozen=True)
class BenchmarkConfig:
    """一つの方策学習計測条件を保持する。"""

    name: str
    device: str
    environment_count: int


def sha256_file(path: Path) -> str:
    """ファイルのSHA-256をストリーム読み込みで返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_parameter_sha256(model: PPO | RecurrentPPO) -> str:
    """方策パラメータを固定順で連結してSHA-256を返す。"""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def benchmark_configurations(cuda_available: bool) -> list[BenchmarkConfig]:
    """同一並列数でCPUとGPUを比較できる条件一覧を作る。"""
    configurations = [
        BenchmarkConfig("cpu_n1", "cpu", 1),
        BenchmarkConfig("cpu_n4", "cpu", 4),
        BenchmarkConfig("cpu_n8", "cpu", 8),
    ]
    if cuda_available:
        configurations.extend(
            [
                BenchmarkConfig("cuda_n4", "cuda", 4),
                BenchmarkConfig("cuda_n8", "cuda", 8),
            ]
        )
    return configurations


def steps_per_environment(environment_count: int) -> int:
    """一回の更新が常に512遷移になる各環境の歩数を返す。"""
    if ROLLOUT_TRANSITIONS % environment_count != 0:
        raise ValueError("環境数は512の約数でなければならない。")
    return ROLLOUT_TRANSITIONS // environment_count


def make_environment(seed: int) -> GeneralObstacleEnv:
    """描画を完全に無効化した訓練分布環境を生成する。"""
    return GeneralObstacleEnv(
        split="train",
        difficulty=1,
        obstacle_count=1,
        base_seed=seed,
        resample_on_reset=True,
        render_mode=None,
    )


def make_environment_factory(seed: int) -> Callable[[], GeneralObstacleEnv]:
    """Windowsのspawn方式で直列化できる環境生成関数を返す。"""

    def factory() -> GeneralObstacleEnv:
        """固定された種から一つの独立環境を作る。"""
        return make_environment(seed)

    return factory


def make_vector_environment(environment_count: int, seed: int) -> VecEnv:
    """単一環境またはspawn型の並列環境を構築する。"""
    factories = [
        make_environment_factory(seed + index * 10_000)
        for index in range(environment_count)
    ]
    if environment_count == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def synchronize_device(device: str) -> None:
    """GPU計測区間の前後で非同期処理を同期する。"""
    if device == "cuda":
        torch.cuda.synchronize()


def run_environment_throughput(
    *,
    environment_count: int,
    transition_target: int,
    seed: int,
) -> dict[str, object]:
    """方策更新を除外してEvoGym物理遷移だけを計測する。"""
    setup_started = time.perf_counter()
    environment = make_vector_environment(environment_count, seed)
    setup_seconds = time.perf_counter() - setup_started
    try:
        environment.reset()
        actions = np.zeros(
            (environment_count,) + environment.action_space.shape,
            dtype=np.float32,
        )
        for _ in range(8):
            environment.step(actions)
        vector_steps = max(1, int(np.ceil(transition_target / environment_count)))
        started = time.perf_counter()
        for _ in range(vector_steps):
            environment.step(actions)
        elapsed = time.perf_counter() - started
    finally:
        environment.close()
    transitions = vector_steps * environment_count
    return {
        "environment_count": environment_count,
        "requested_transitions": transition_target,
        "measured_transitions": transitions,
        "setup_seconds": setup_seconds,
        "measurement_seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
    }


def build_model(
    *,
    algorithm: str,
    environment: VecEnv,
    device: str,
    seed: int,
) -> PPO | RecurrentPPO:
    """同じロールアウト量と最適化条件で新規ベンチマーク方策を作る。"""
    common = {
        "env": environment,
        "device": device,
        "seed": seed,
        "n_steps": steps_per_environment(environment.num_envs),
        "batch_size": 128,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "verbose": 0,
    }
    if algorithm == "ppo":
        return PPO(
            "MlpPolicy",
            policy_kwargs={"net_arch": [64, 64]},
            **common,
        )
    if algorithm == "recurrent_ppo":
        return RecurrentPPO(
            "MlpLstmPolicy",
            policy_kwargs={
                "net_arch": [64, 64],
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
            },
            **common,
        )
    raise ValueError(f"未知のアルゴリズム: {algorithm}")


def run_training_throughput(
    *,
    configuration: BenchmarkConfig,
    algorithm: str,
    total_steps: int,
    seed: int,
) -> dict[str, object]:
    """新規方策だけを短く更新して端から端までの速度を測る。"""
    environment = make_vector_environment(configuration.environment_count, seed)
    model = build_model(
        algorithm=algorithm,
        environment=environment,
        device=configuration.device,
        seed=seed,
    )
    try:
        model.learn(
            total_timesteps=ROLLOUT_TRANSITIONS,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        hash_before = policy_parameter_sha256(model)
        if configuration.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize_device(configuration.device)
        started = time.perf_counter()
        model.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        synchronize_device(configuration.device)
        elapsed = time.perf_counter() - started
        hash_after = policy_parameter_sha256(model)
        measured_steps = int(model.num_timesteps) - ROLLOUT_TRANSITIONS
        peak_memory = (
            int(torch.cuda.max_memory_allocated())
            if configuration.device == "cuda"
            else 0
        )
    finally:
        environment.close()
        del model
        if configuration.device == "cuda":
            torch.cuda.empty_cache()
    return {
        **asdict(configuration),
        "algorithm": algorithm,
        "warmup_transitions": ROLLOUT_TRANSITIONS,
        "requested_training_transitions": total_steps,
        "measured_training_transitions": measured_steps,
        "measurement_seconds": elapsed,
        "transitions_per_second": measured_steps / elapsed,
        "policy_hash_before_measurement": hash_before,
        "policy_hash_after_measurement": hash_after,
        "parameters_changed": hash_before != hash_after,
        "peak_cuda_memory_bytes": peak_memory,
        "checkpoint_saved": False,
    }


def run_render_case(
    *,
    name: str,
    target_rps: int | None,
    resolution: tuple[int, int],
    frame_count: int,
    seed: int,
    keep_frames: bool,
) -> tuple[dict[str, object], list[np.ndarray]]:
    """OpenGL描画とCPU読戻しを指定条件で計測する。"""
    environment = GeneralObstacleEnv(
        split="train",
        difficulty=1,
        obstacle_count=1,
        base_seed=seed,
        resample_on_reset=True,
        render_mode="rgb_array",
    )
    frames: list[np.ndarray] = []
    checksum = 0
    try:
        observation, _ = environment.reset(seed=seed)
        del observation
        environment.default_viewer.set_target_rps(target_rps)
        environment.default_viewer.set_resolution(resolution)
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        environment.render()
        started = time.perf_counter()
        for _ in range(frame_count):
            _, _, terminated, truncated, _ = environment.step(action)
            frame = np.asarray(environment.render())
            checksum = (checksum + int(frame[::16, ::16].sum())) % (2**63 - 1)
            if keep_frames:
                frames.append(frame.copy())
            if terminated or truncated:
                environment.reset()
        elapsed = time.perf_counter() - started
    finally:
        environment.close()
    return (
        {
            "name": name,
            "target_rps": target_rps,
            "resolution": list(resolution),
            "frame_count": frame_count,
            "measurement_seconds": elapsed,
            "frames_per_second": frame_count / elapsed,
            "frame_checksum": checksum,
        },
        frames,
    )


def run_render_benchmark(
    *,
    output_dir: Path,
    frame_count: int,
    seed: int,
) -> dict[str, object]:
    """既定描画、無制限描画、低解像度描画とGIF符号化を比較する。"""
    cases = []
    default_case, _ = run_render_case(
        name="default_1200x600_50rps",
        target_rps=50,
        resolution=(1200, 600),
        frame_count=frame_count,
        seed=seed,
        keep_frames=False,
    )
    cases.append(default_case)
    uncapped_case, _ = run_render_case(
        name="uncapped_1200x600",
        target_rps=None,
        resolution=(1200, 600),
        frame_count=frame_count,
        seed=seed,
        keep_frames=False,
    )
    cases.append(uncapped_case)
    low_case, frames = run_render_case(
        name="uncapped_600x300",
        target_rps=None,
        resolution=(600, 300),
        frame_count=frame_count,
        seed=seed,
        keep_frames=True,
    )
    cases.append(low_case)
    gif_path = output_dir / "render_encoding_probe.gif"
    encoding_started = time.perf_counter()
    imageio.mimsave(gif_path, frames, fps=12, loop=0)
    encoding_seconds = time.perf_counter() - encoding_started
    return {
        "cases": cases,
        "gif_encoding": {
            "path": str(gif_path.resolve()),
            "frame_count": len(frames),
            "encoding_seconds": encoding_seconds,
            "frames_per_second": len(frames) / encoding_seconds,
            "file_bytes": gif_path.stat().st_size,
            "encoder": "imageio_cpu_gif",
        },
    }


def add_speedups(summary: dict[str, object]) -> None:
    """単一CPU基準と同一並列数CPU基準の速度比を結果へ追加する。"""
    rows = summary["training_benchmarks"]
    assert isinstance(rows, list)
    by_key = {
        (str(row["algorithm"]), str(row["name"])): row
        for row in rows
    }
    for row in rows:
        baseline = by_key[(str(row["algorithm"]), "cpu_n1")]
        row["speedup_vs_cpu_n1"] = (
            float(row["transitions_per_second"])
            / float(baseline["transitions_per_second"])
        )
        if str(row["device"]) == "cuda":
            cpu_name = f"cpu_n{int(row['environment_count'])}"
            cpu_peer = by_key[(str(row["algorithm"]), cpu_name)]
            row["speedup_vs_same_parallel_cpu"] = (
                float(row["transitions_per_second"])
                / float(cpu_peer["transitions_per_second"])
            )
        else:
            row["speedup_vs_same_parallel_cpu"] = 1.0


def make_recommendation(summary: dict[str, object]) -> dict[str, object]:
    """各アルゴリズムの最速条件とGPU採用可否を判定する。"""
    rows = summary["training_benchmarks"]
    assert isinstance(rows, list)
    algorithms = sorted({str(row["algorithm"]) for row in rows})
    decisions = {}
    for algorithm in algorithms:
        candidates = [row for row in rows if row["algorithm"] == algorithm]
        fastest = max(candidates, key=lambda row: float(row["transitions_per_second"]))
        gpu_candidates = [row for row in candidates if row["device"] == "cuda"]
        best_gpu = (
            max(gpu_candidates, key=lambda row: float(row["transitions_per_second"]))
            if gpu_candidates
            else None
        )
        decisions[algorithm] = {
            "fastest_configuration": fastest["name"],
            "fastest_transitions_per_second": fastest["transitions_per_second"],
            "gpu_measurably_beneficial": bool(
                best_gpu is not None
                and float(best_gpu["speedup_vs_same_parallel_cpu"]) >= 1.05
            ),
            "best_gpu_configuration": best_gpu["name"] if best_gpu else None,
            "best_gpu_speedup_vs_same_parallel_cpu": (
                best_gpu["speedup_vs_same_parallel_cpu"] if best_gpu else None
            ),
        }
    return {
        "minimum_material_speedup": 1.05,
        "algorithm_decisions": decisions,
        "formal_training_should_remain_paused": True,
        "teacher_allowed_in_final_student_test": False,
    }


def hardware_metadata() -> dict[str, object]:
    """再現に必要なPython、PyTorch、CUDA情報を収集する。"""
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_num_threads": torch.get_num_threads(),
    }


def main() -> None:
    """隔離ベンチマークを実行しJSONとCSV互換の要約を保存する。"""
    parser = argparse.ArgumentParser(
        description="EvoGymのGPU・CPU並列性能を隔離条件で計測する。"
    )
    parser.add_argument("--run-name", default="gpu_parallel_benchmark_v1")
    parser.add_argument("--total-steps", type=int, default=4096)
    parser.add_argument("--environment-transitions", type=int, default=4096)
    parser.add_argument("--render-frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("ppo", "recurrent_ppo"),
        default=("ppo", "recurrent_ppo"),
    )
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    if args.total_steps <= 0 or args.total_steps % ROLLOUT_TRANSITIONS != 0:
        raise ValueError("total-stepsは512の正の倍数でなければならない。")
    if not PROTECTED_STUDENT.is_file():
        raise FileNotFoundError(PROTECTED_STUDENT)
    protected_hash_before = sha256_file(PROTECTED_STUDENT)
    if protected_hash_before != EXPECTED_PROTECTED_STUDENT_SHA256:
        raise RuntimeError("保護対象学生の事前SHA-256が凍結値と一致しない。")

    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    configurations = benchmark_configurations(torch.cuda.is_available())
    environment_counts = sorted({item.environment_count for item in configurations})
    summary: dict[str, object] = {
        "method": "isolated_gpu_parallel_performance_benchmark_v1",
        "formal_training": False,
        "benchmark_models_initialized_from_scratch": True,
        "benchmark_checkpoints_saved": False,
        "protected_student_loaded": False,
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
        "validation_split_accesses": 0,
        "holdout_split_accesses": 0,
        "hardware": hardware_metadata(),
        "protected_student": {
            "path": str(PROTECTED_STUDENT.resolve()),
            "expected_sha256": EXPECTED_PROTECTED_STUDENT_SHA256,
            "sha256_before": protected_hash_before,
        },
        "configuration": {
            "total_steps_per_training_case": args.total_steps,
            "warmup_steps_per_training_case": ROLLOUT_TRANSITIONS,
            "rollout_transitions_per_update": ROLLOUT_TRANSITIONS,
            "environment_transition_target": args.environment_transitions,
            "render_frames_per_case": args.render_frames,
            "algorithms": list(args.algorithms),
            "configurations": [asdict(item) for item in configurations],
        },
        "environment_benchmarks": [],
        "training_benchmarks": [],
    }
    for environment_count in environment_counts:
        result = run_environment_throughput(
            environment_count=environment_count,
            transition_target=args.environment_transitions,
            seed=args.seed,
        )
        summary["environment_benchmarks"].append(result)
        print(json.dumps({"environment": result}, ensure_ascii=False), flush=True)
    for algorithm in args.algorithms:
        for configuration in configurations:
            result = run_training_throughput(
                configuration=configuration,
                algorithm=algorithm,
                total_steps=args.total_steps,
                seed=args.seed,
            )
            summary["training_benchmarks"].append(result)
            print(json.dumps({"training": result}, ensure_ascii=False), flush=True)
    add_speedups(summary)
    if not args.skip_render:
        summary["render_benchmark"] = run_render_benchmark(
            output_dir=output_dir,
            frame_count=args.render_frames,
            seed=args.seed,
        )
    protected_hash_after = sha256_file(PROTECTED_STUDENT)
    summary["protected_student"]["sha256_after"] = protected_hash_after
    summary["protected_student"]["unchanged"] = (
        protected_hash_after == protected_hash_before
    )
    if protected_hash_after != EXPECTED_PROTECTED_STUDENT_SHA256:
        raise RuntimeError("保護対象学生の事後SHA-256が凍結値と一致しない。")
    summary["recommendation"] = make_recommendation(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
