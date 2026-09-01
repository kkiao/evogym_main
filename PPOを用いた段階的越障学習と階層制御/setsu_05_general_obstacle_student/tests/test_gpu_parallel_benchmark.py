"""GPU・CPU並列ベンチマークの純粋関数を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.benchmark_gpu_parallel import (
    add_speedups,
    benchmark_configurations,
    make_recommendation,
    steps_per_environment,
)


class GpuParallelBenchmarkTest(unittest.TestCase):
    """ベンチマーク条件と速度比計算の回帰を防ぐ。"""

    def test_configuration_matrix_adds_gpu_only_when_available(self) -> None:
        """CUDA可否に応じて比較対象だけを追加する。"""
        cpu_names = [item.name for item in benchmark_configurations(False)]
        gpu_names = [item.name for item in benchmark_configurations(True)]
        self.assertEqual(cpu_names, ["cpu_n1", "cpu_n4", "cpu_n8"])
        self.assertEqual(
            gpu_names,
            ["cpu_n1", "cpu_n4", "cpu_n8", "cuda_n4", "cuda_n8"],
        )

    def test_rollout_size_is_constant(self) -> None:
        """並列数に依存せず一更新が512遷移になる。"""
        for count in (1, 4, 8):
            self.assertEqual(steps_per_environment(count) * count, 512)

    def test_speedup_and_recommendation_use_same_parallel_cpu(self) -> None:
        """GPU採用判定が同じ環境数のCPU条件を基準にする。"""
        summary = {
            "training_benchmarks": [
                {
                    "algorithm": "ppo",
                    "name": "cpu_n1",
                    "device": "cpu",
                    "environment_count": 1,
                    "transitions_per_second": 100.0,
                },
                {
                    "algorithm": "ppo",
                    "name": "cpu_n4",
                    "device": "cpu",
                    "environment_count": 4,
                    "transitions_per_second": 200.0,
                },
                {
                    "algorithm": "ppo",
                    "name": "cpu_n8",
                    "device": "cpu",
                    "environment_count": 8,
                    "transitions_per_second": 220.0,
                },
                {
                    "algorithm": "ppo",
                    "name": "cuda_n4",
                    "device": "cuda",
                    "environment_count": 4,
                    "transitions_per_second": 230.0,
                },
                {
                    "algorithm": "ppo",
                    "name": "cuda_n8",
                    "device": "cuda",
                    "environment_count": 8,
                    "transitions_per_second": 221.0,
                },
            ]
        }
        add_speedups(summary)
        decision = make_recommendation(summary)["algorithm_decisions"]["ppo"]
        self.assertTrue(decision["gpu_measurably_beneficial"])
        self.assertEqual(decision["best_gpu_configuration"], "cuda_n4")


if __name__ == "__main__":
    unittest.main()
