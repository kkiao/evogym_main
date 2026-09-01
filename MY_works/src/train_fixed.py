# 固定したWalker用の体で、自作REINFORCEを学習するファイル
"""
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
"""

#####################################################追加####################
import csv
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv

# 不要なインポート「from examples.bo import optimizer」を削除しました
from src.body.fixed_body import make_walker_body
from src.policy import Policy
from src.reinforce import calculate_returns, choose_action_batch, update_policy

TASK_NAME = "Walker-v0"
NUM_ENVS = 16  # 並列数：9800X3Dなら16が最適
EPISODES = 16000 # 並列化するので多めに設定
GAMMA = 0.99
MAX_STEPS = 500
OUTPUT_PREFIX = "parallel_fixed"
BODY_FILE = None

def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # GPU設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if BODY_FILE is None:
        body = make_walker_body()
    else:
        # もしファイルがない場合のエラーを防ぐ
        try:
            body = np.load(BODY_FILE)
        except FileNotFoundError:
            print(f"Warning: {BODY_FILE} not found. Making a new body.")
            body = make_walker_body()

    np.save(results_dir / f"{OUTPUT_PREFIX}_body.npy", body)

    # --- 並列環境の作成 ---
    def make_single_env():
        return gym.make(TASK_NAME, body=body, render_mode=None)

    # 16個の環境を裏で起動
    envs = AsyncVectorEnv([make_single_env for _ in range(NUM_ENVS)])
    
    # 初期リセット（全環境分の観測を一気に取得）
    obs, _ = envs.reset(seed=7)
    observation_size = obs.shape[1]
    action_size = envs.single_action_space.shape[0]

    # PolicyをGPUへ
    policy = Policy(observation_size, action_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)

    rows = []

    # 並列化のため、グループ単位で学習を進める
    for episode_group in range(EPISODES // NUM_ENVS):
        obs, _ = envs.reset()

        # 16環境分のデータを貯める箱
        batch_rewards = [[] for _ in range(NUM_ENVS)]
        batch_log_probs = [[] for _ in range(NUM_ENVS)]
        batch_returns = [0.0 for _ in range(NUM_ENVS)]

        for step in range(MAX_STEPS):
            # 16個一気に計算
            actions, log_probs = choose_action_batch(policy, obs, envs, device)

            # 16個一気に1ステップ進む
            next_obs, rewards, terminateds, truncateds, _ = envs.step(actions)

            for i in range(NUM_ENVS):
                batch_rewards[i].append(rewards[i])
                batch_log_probs[i].append(log_probs[i])
                batch_returns[i] += rewards[i]

            obs = next_obs
            # いずれかの環境が終了しても、VectorEnvは自動リセットされるため続行可能ですが、
            # 今回は簡易的に全環境の終了フラグを確認します
            if np.all(terminateds | truncateds):
                break

       # --- 修正後の学習更新部分：16個のロスをまとめて一気に更新 ---
        all_loss = []
        
        for i in range(NUM_ENVS):
            # i番目の環境のリターン（将来報酬）を計算
            rets = calculate_returns(batch_rewards[i], GAMMA)
            
            # テンソルに変換（警告が出ない書き方）
            log_probs_tensor = torch.stack(batch_log_probs[i])
            returns_tensor = torch.as_tensor(rets, dtype=torch.float32, device=device)
            
            # 報酬の正規化
            if len(returns_tensor) > 1:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            
            # 各環境のロスを計算してリストに貯める
            # (注意：ここではまだ .backward() を呼ばない)
            all_loss.append(-(log_probs_tensor * returns_tensor).sum())

        # 16個分のロスの平均をとる
        total_loss = torch.stack(all_loss).mean()
        
        # ここで一気にバックプロパゲーション（重みの更新）
        # これなら1回しか呼ばれないのでエラーにならない
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ログの記録
        avg_return = np.mean(batch_returns)
        avg_loss = total_loss.item()
        rows.append([episode_group, avg_return, avg_loss])

        print(f"Group: {episode_group} | Avg Return: {avg_return:.2f} | Loss: {avg_loss:.4f}")

    envs.close()

    # 学習結果をCSVへ保存
    with open(results_dir / f"{OUTPUT_PREFIX}_training.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "avg_return", "avg_loss"])
        writer.writerows(rows)

    # Policyの重みを保存
    torch.save(policy.state_dict(), results_dir / f"{OUTPUT_PREFIX}_policy.pt")
    print(f"学習完了！結果を {results_dir} に保存しました。")

if __name__ == "__main__":
    main()