# 固定したWalker用の体で、自作REINFORCEを学習するファイル

import csv
from pathlib import Path

import evogym.envs
import gymnasium as gym
import numpy as np
import torch

from src.body import make_walker_body
from src.policy import Policy
from src.reinforce import calculate_returns, choose_action, update_policy


# Walker-v0：平らな地面を右方向へ進む課題
TASK_NAME = "Walker-v0"

# 「1回の試行→学習更新」を繰り返す回数
EPISODES = 120

# 将来の報酬をどの程度重視するか
GAMMA = 0.99

# 1エピソード内で行動できる最大回数
MAX_STEPS = 500

# 出力ファイルの名前の先頭
OUTPUT_PREFIX = "best"

# Noneなら自分で用意したWalker用固定体を使う。
# 既存の体を使う場合だけ、"results/xxxx_body.npy" のように書く。
BODY_FILE = "results/best_body.npy"


def main():
    # resultsフォルダがなければ作る。
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Noneなら固定体を作る。None以外なら.npyファイルから体を読む。
    if BODY_FILE is None:
        body = make_walker_body()
    else:
        body = np.load(BODY_FILE)

    # 使った体を保存する。
    np.save(results_dir / f"{OUTPUT_PREFIX}_body.npy", body)

    # 体をWalker-v0へ渡して、シミュレーション環境を作る。
    # 学習中は画像を作らないので render_mode=None。
    env = gym.make(TASK_NAME, body=body, render_mode=None)

    # resetは「ロボットを初期位置へ戻し、最初の観測obsを受け取る」処理。
    obs, _ = env.reset(seed=7)

    # 観測の数、筋肉（行動）の数を環境から調べる。
    observation_size = len(obs)
    action_size = env.action_space.shape[0]

    # 観測数と行動数に合う、新しいPolicyを作る。
    policy = Policy(observation_size, action_size)

    # Policyの重みをどう更新するかを担当する道具。
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)

    # 後でCSVへ書く、各エピソードの結果を入れるリスト。
    rows = []

    for episode_number in range(EPISODES):
        # 新しい1回の試行を開始する。
        obs, _ = env.reset()

        rewards = []
        log_probs = []
        episode_return = 0.0

        for step_number in range(MAX_STEPS):
            # Policyに観測を渡し、筋肉への行動と、その行動のlog確率を得る。
            action, log_prob = choose_action(
                policy,
                obs,
                env.action_space.low,
                env.action_space.high,
            )

            # 行動を環境に渡す。
            # next_obs: 次の観測、reward: 今回の点数
            next_obs, reward, terminated, truncated, _ = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)
            episode_return += reward
            obs = next_obs

            # 終了条件に達したら、このエピソードを終える。
            if terminated or truncated:
                break

        # 各時点から見た将来報酬を計算し、
        # log_probsを使ってPolicyの重みを更新する。
        returns = calculate_returns(rewards, GAMMA)
        loss = update_policy(optimizer, log_probs, returns)

        # グラフ用に、今回の結果を保存しておく。
        rows.append([episode_number, episode_return, loss])

        print(
            "episode:", episode_number,
            "return:", episode_return,
            "loss:", loss,
        )

    env.close()

    # 学習結果をCSVへ保存する。
    with open(
        results_dir / f"{OUTPUT_PREFIX}_training.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "return", "loss"])
        writer.writerows(rows)

    # 最後に学習済みPolicyの重みを保存する。
    torch.save(
        policy.state_dict(),
        results_dir / f"{OUTPUT_PREFIX}_policy.pt",
    )

    print("学習結果を保存しました。")


if __name__ == "__main__":
    main()