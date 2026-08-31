import csv
from pathlib import Path
import random

import numpy as np

from src.body import make_climber_seed_body, mutate_body
from src.reinforce import evaluate_body, train_body


# 実験条件。最初は小さくして、最後まで動くか確認する。
TASK_NAME = "Climber-v0"
POPULATION_SIZE = 3
GENERATIONS = 2
SHORT_TRAIN_EPISODES = 10
EVALUATION_EPISODES = 2
MAX_STEPS = 500


def make_initial_population():
    """固定体1個と、その突然変異体を作り、最初の集団を返す。"""
    parent = make_climber_seed_body()
    population = [parent]

    while len(population) < POPULATION_SIZE:
        child = mutate_body(parent)
        population.append(child)

    return population


def choose_best(scored_bodies):
    """(fitness, body)のリストから、fitness最大の一組を返す。"""
    best_fitness, best_body = scored_bodies[0]

    for fitness, body in scored_bodies[1:]:
        if fitness > best_fitness:
            best_fitness = fitness
            best_body = body

    return best_fitness, best_body


def make_next_generation(best_body):
    """最良体を残し、最良体の突然変異体で次世代を満たす。"""
    next_generation = [best_body.copy()]

    while len(next_generation) < POPULATION_SIZE:
        child = mutate_body(best_body)
        next_generation.append(child)

    return next_generation


def main():
    # 1. 同じ実験を再実行しやすくするため、Pythonの乱数の種を固定する。
    random.seed(7)

    # 2. 結果を保存するフォルダと、世代をまたいで使う変数を準備する。
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    history_rows = []
    population = make_initial_population()
    best_fitness_ever = None
    best_body_ever = None

    # 3. 世代ごとに、全個体を学習・評価する。
    for generation in range(GENERATIONS):
        print("=== generation", generation, "===")
        scored_bodies = []

        # 3-a. この世代の各体へ、公平に新しいPolicyを作って短期学習させる。
        for individual, body in enumerate(population):
            print("individual", individual, "を学習・評価します")

            policy, training_returns = train_body(
                TASK_NAME,
                body,
                episodes=SHORT_TRAIN_EPISODES,
                max_steps=MAX_STEPS,
                seed=7,
            )

            # 3-b. 重みを変えず、平均行動で評価した平均報酬をfitnessにする。
            fitness = evaluate_body(
                TASK_NAME,
                body,
                policy,
                episodes=EVALUATION_EPISODES,
                max_steps=MAX_STEPS,
            )

            # 3-c. 後で形と数値を比較できるよう、体とCSVの一行を保存する。
            body_file = results_dir / f"g{generation}_i{individual}_body.npy"
            np.save(body_file, body)
            history_rows.append([
                generation,
                individual,
                fitness,
                training_returns[-1],
                str(body_file),
            ])
            scored_bodies.append((fitness, body))
            print("fitness:", fitness)

            # 3-d. 全世代を通じた最良体も別に覚えておく。
            if best_fitness_ever is None or fitness > best_fitness_ever:
                best_fitness_ever = fitness
                best_body_ever = body.copy()

        # 4. この世代の最良体を選び、最終世代以外では次世代を作る。
        generation_best_fitness, generation_best_body = choose_best(scored_bodies)
        print("この世代の最高fitness:", generation_best_fitness)

        if generation < GENERATIONS - 1:
            population = make_next_generation(generation_best_body)

    # 5. 実験全体で最良だった体と、全個体の結果表を保存する。
    np.save(results_dir / "best_body.npy", best_body_ever)

    with open(results_dir / "evolution_history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "generation",
            "individual",
            "fitness",
            "last_train_return",
            "body_file",
        ])
        writer.writerows(history_rows)

    print("全世代で最も良い体を保存しました: results/best_body.npy")


if __name__ == "__main__":
    main()