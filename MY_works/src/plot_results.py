import csv
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"

def read_training_csv(csv_path):
    """Day 1のCSVから、横軸episodeと縦軸returnの二つのリストを作る。"""
    episodes = []
    returns = []

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            episodes.append(int(row["episode"]))
            returns.append(float(row["return"]))

    return episodes, returns


def make_training_graph(csv_path, output_path, title):
    """CSVの学習結果をPNGの折れ線グラフへ保存する。"""
    episodes, returns = read_training_csv(csv_path)
    average_returns = moving_average(returns, window=20)

    #returnの平均が前半後半でどれくらい変わったかを文字で表示する
    first_average = sum(returns[:20]) / 20
    last_average = sum(returns[-20:]) / 20
    improvement = (last_average - first_average) / first_average * 100

    summary = (
        f"最初の20回の平均 :{first_average:.3f} -"
        f"最後の20回の平均 :{last_average:.3f} -"
        f"約 {improvement:.1f}% の改善"
    )


    plt.figure(figsize=(8, 4))
    # 細い薄い線：各episodeの実際の報酬。探索のため上下する。
    plt.plot(episodes, returns, color="lightgray", label="each episode")
    # 太い青線：直近20回の平均。学習の傾向を見る。
    plt.plot(
    episodes,
    average_returns,
    color="blue",
    linewidth=2.5,
    label="20-episode moving average",
)
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title(title)
    plt.grid()
    plt.legend()
    #文字の表示方法
    plt.figtext(
    0.5,
    0.01,
    summary,
    ha="center",
    fontsize=10,
)
    #違うやつ
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path)
    plt.close()


def main():
    results_dir = Path("results")
    make_training_graph(
        results_dir / "fixed_training.csv",
        results_dir / "fixed_learning_curve.png",
        "Fixed body training in Climber-v0",
    )

    make_evolution_graph(
    results_dir / "evolution_history.csv",
    results_dir / "evolution_curve.png",
)
    make_training_graph(
    results_dir / "best_training.csv",
    results_dir / "best_learning_curve.png",
    "Best evolved body training",
)
    
#平均のreturnを折れ線グラフで表示
def moving_average(values, window=20):
    averages = []

    for index in range(len(values)):
        start = max(0, index - window + 1)
        recent_values = values[start:index + 1]
        averages.append(sum(recent_values) / len(recent_values))

    return averages


def make_evolution_graph(csv_path, output_path):
    #世代ごとの最高fitnessと平均fitnessをPNGへ保存する。
        fitness_by_generation = {}

        with open(csv_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                generation = int(row["generation"])
                fitness = float(row["fitness"])

                if generation not in fitness_by_generation:
                    fitness_by_generation[generation] = []
                fitness_by_generation[generation].append(fitness)

        generations = sorted(fitness_by_generation.keys())
        best_values = [max(fitness_by_generation[generation]) for generation in generations]
        mean_values = [
            sum(fitness_by_generation[generation]) / len(fitness_by_generation[generation])
            for generation in generations
        ]

        plt.figure(figsize=(8, 4))
        plt.plot(generations, best_values, marker="o", label="best fitness")
        plt.plot(generations, mean_values, marker="o", label="mean fitness")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.title("Evolution in Climber-v0")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    main()

