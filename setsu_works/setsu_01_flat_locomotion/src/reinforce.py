"""REINFORCE（モンテカルロ方策勾配）の学習・評価ツール。"""

import gymnasium as gym
import evogym.envs  # noqa: F401 - 読み込み時にEvoGym環境をGymnasiumへ登録する
import numpy as np
import torch
from torch.distributions import Normal

from src.policy import Policy


def calculate_returns(rewards, gamma):
    """各時刻からエピソード終了までの割引累積報酬を計算する。"""
    returns = []
    future_return = 0.0
    for reward in reversed(rewards):
        future_return = float(reward) + gamma * future_return
        returns.append(future_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def scale_latent_action(latent_action, action_low, action_high):
    """滑らかなtanh写像で任意の実数行動を環境の有効範囲へ変換する。"""
    low = torch.as_tensor(action_low, dtype=latent_action.dtype, device=latent_action.device)
    high = torch.as_tensor(action_high, dtype=latent_action.dtype, device=latent_action.device)
    center = (low + high) / 2.0
    half_range = (high - low) / 2.0
    return center + half_range * torch.tanh(latent_action)


def choose_action(
    policy,
    obs,
    action_low,
    action_high,
    exploration_std=0.20,
):
    """方策分布から有界行動を標本化し、REINFORCE用の対数確率を返す。"""
    if exploration_std <= 0:
        raise ValueError("exploration_std 必须大于 0。")

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    latent_mean = policy(obs_tensor)
    distribution = Normal(latent_mean, torch.full_like(latent_mean, exploration_std))

    # 非有界の潜在空間で標本化してから、EvoGymの行動範囲へ滑らかに写像する。
    # 環境は切り詰め後の行動を実行する一方、勾配は切り詰め前を参照する不整合を防ぐ。
    latent_action = distribution.sample()
    action = scale_latent_action(latent_action, action_low, action_high)
    log_prob = distribution.log_prob(latent_action).sum()

    return action.squeeze(0).detach().cpu().numpy(), log_prob


def policy_parameters(optimizer):
    """勾配クリッピング用に、オプティマイザが管理する全パラメータを取得する。"""
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def update_policy(optimizer, log_probs, returns, max_grad_norm=1.0):
    """一つ以上の完全なエピソードを用いてREINFORCE更新を一回実行する。"""
    if not log_probs:
        raise ValueError("没有可用于更新的 log_probs。")

    log_probs_tensor = torch.stack(log_probs)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32).reshape(-1)
    if log_probs_tensor.numel() != returns_tensor.numel():
        raise ValueError(
            f"log_probs 数量 {log_probs_tensor.numel()} 与 returns 数量 "
            f"{returns_tensor.numel()} 不一致。"
        )
    if not torch.isfinite(log_probs_tensor).all() or not torch.isfinite(returns_tensor).all():
        raise FloatingPointError("策略更新数据中出现 NaN 或 Inf。")

    # 複数エピソードを同一バッチで正規化し、エピソード間の相対的な優劣を保つ。
    normalized_returns = (
        (returns_tensor - returns_tensor.mean())
        / (returns_tensor.std(unbiased=False) + 1e-6)
    )
    loss = -(log_probs_tensor.reshape(-1) * normalized_returns).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("策略损失出现 NaN 或 Inf。")

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy_parameters(optimizer), max_grad_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad()
        raise FloatingPointError("策略梯度出现 NaN 或 Inf。")
    optimizer.step()
    return float(loss.item())


def make_mean_action(policy, obs, low_array, high_array):
    """評価時はランダム探索を加えず、方策の平均行動を使用する。"""
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        latent_mean = policy(obs_tensor)
        action = scale_latent_action(latent_mean, low_array, high_array)
    return action.squeeze(0).cpu().numpy()


def summarize_rollouts(episode_returns, episode_steps):
    """エピソード別の結果を共通評価指標へ集約する。"""
    returns_array = np.asarray(episode_returns, dtype=float)
    steps_array = np.asarray(episode_steps, dtype=float)
    speeds_array = returns_array / np.maximum(steps_array, 1.0)
    return {
        "mean_return": float(returns_array.mean()),
        "std_return": float(returns_array.std()),
        "min_return": float(returns_array.min()),
        "max_return": float(returns_array.max()),
        "mean_steps": float(steps_array.mean()),
        "mean_speed": float(speeds_array.mean()),
        "episode_returns": [float(value) for value in episode_returns],
        "episode_steps": [int(value) for value in episode_steps],
    }


def evaluate_policy(task_name, body, policy, episodes, max_steps, seed=100):
    """方策を決定論的に評価し、報酬・歩数・平均変位などの統計量を返す。"""
    if episodes <= 0:
        raise ValueError("episodes 必须大于 0。")

    env = gym.make(task_name, body=body, render_mode=None)
    episode_returns = []
    episode_steps = []
    try:
        for episode_number in range(episodes):
            obs, _ = env.reset(seed=seed + episode_number)
            episode_return = 0.0
            steps = 0

            for _ in range(max_steps):
                action = make_mean_action(
                    policy,
                    obs,
                    env.action_space.low,
                    env.action_space.high,
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break

            episode_returns.append(episode_return)
            episode_steps.append(steps)
    finally:
        env.close()

    return summarize_rollouts(episode_returns, episode_steps)


def evaluate_random_actions(task_name, body, episodes, max_steps, seed=2000):
    """一様ランダム行動を評価し、方策が学習できたかを判定する基準にする。"""
    if episodes <= 0:
        raise ValueError("episodes 必须大于 0。")

    env = gym.make(task_name, body=body, render_mode=None)
    episode_returns = []
    episode_steps = []
    try:
        for episode_number in range(episodes):
            episode_seed = seed + episode_number
            obs, _ = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            episode_return = 0.0
            steps = 0
            for _ in range(max_steps):
                obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                episode_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            episode_returns.append(episode_return)
            episode_steps.append(steps)
    finally:
        env.close()
    return summarize_rollouts(episode_returns, episode_steps)


def evaluate_sampled_policy(
    task_name,
    body,
    policy,
    episodes,
    max_steps,
    exploration_std,
    seed=3000,
):
    """ネットワークを更新せず、指定した探索ノイズで学習済み確率方策を評価する。"""
    if episodes <= 0:
        raise ValueError("episodes 必须大于 0。")
    if exploration_std <= 0:
        raise ValueError("exploration_std 必须大于 0。")

    original_torch_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    env = gym.make(task_name, body=body, render_mode=None)
    episode_returns = []
    episode_steps = []
    try:
        for episode_number in range(episodes):
            obs, _ = env.reset(seed=seed + episode_number)
            episode_return = 0.0
            steps = 0
            for _ in range(max_steps):
                with torch.no_grad():
                    action, _ = choose_action(
                        policy,
                        obs,
                        env.action_space.low,
                        env.action_space.high,
                        exploration_std,
                    )
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            episode_returns.append(episode_return)
            episode_steps.append(steps)
    finally:
        env.close()
        torch.set_rng_state(original_torch_rng_state)
    return summarize_rollouts(episode_returns, episode_steps)


def train_body(
    task_name,
    body,
    episodes,
    max_steps,
    seed=7,
    batch_episodes=5,
    exploration_std=0.20,
):
    """形状ごとに新しい方策を学習する。任意の形状進化実験向けに保持する。"""
    if batch_episodes <= 0:
        raise ValueError("batch_episodes 必须大于 0。")

    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make(task_name, body=body, render_mode=None)
    first_obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    policy = Policy(len(first_obs), env.action_space.shape[0])
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    training_returns = []
    batch_log_probs = []
    batch_returns = []

    try:
        for episode_number in range(episodes):
            obs, _ = env.reset(seed=seed + episode_number)
            rewards = []
            episode_log_probs = []
            episode_return = 0.0

            for _ in range(max_steps):
                action, log_prob = choose_action(
                    policy,
                    obs,
                    env.action_space.low,
                    env.action_space.high,
                    exploration_std,
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                episode_log_probs.append(log_prob)
                episode_return += float(reward)
                if terminated or truncated:
                    break

            batch_log_probs.extend(episode_log_probs)
            batch_returns.append(calculate_returns(rewards, gamma=0.99))
            training_returns.append(episode_return)

            batch_is_full = (episode_number + 1) % batch_episodes == 0
            is_last_episode = episode_number + 1 == episodes
            if batch_is_full or is_last_episode:
                loss = update_policy(
                    optimizer,
                    batch_log_probs,
                    torch.cat(batch_returns),
                )
                batch_log_probs = []
                batch_returns = []
                print(
                    f"train episode {episode_number + 1}/{episodes} "
                    f"return={episode_return:.6f} loss={loss:.6f}"
                )
    finally:
        env.close()

    return policy, training_returns


def evaluate_body(task_name, body, policy, episodes, max_steps):
    """旧進化コードとの互換用に、決定論的評価の平均報酬を返す。"""
    metrics = evaluate_policy(task_name, body, policy, episodes, max_steps)
    return metrics["mean_return"]
