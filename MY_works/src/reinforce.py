#強化学習
#行動の評価、方策で状態から行動を選ぶ->その行動を選んだ確率を計算する、方策を更新し良い行動を選ぶ確率を上げる
from src.policy import Policy
import gymnasium as gym
import evogym.envs
import torch
from torch.distributions import Normal

#1エピソードの、その時点以降で得られる報酬の合計を計算する。これにより、どのくらいその行動の価値があったかを評価する。
def calculate_returns(rewards, gamma):
    returns = []
    future_return = 0.0

    # 最後の報酬から逆向きに計算する。
    for reward in reversed(rewards):
        future_return = reward + gamma * future_return
        returns.insert(0, future_return)

    return torch.tensor(returns, dtype=torch.float32)

#状態から行動を一つ選び、その行動を選んだ確率を計算する
def choose_action(policy, obs, action_low, action_high):
    # NumPy配列の観測を、Policyへ渡せるTensorにする。
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mean = policy(obs_tensor)

    # EvoGymが許す行動範囲の中央と半分の幅を作る。
    low = torch.tensor(action_low, dtype=torch.float32)
    high = torch.tensor(action_high, dtype=torch.float32)
    center = (low + high) / 2
    half_range = (high - low) / 2

    # tanhの-1〜1を、EvoGymが許す行動範囲へ変換する。
    mean_in_range = center + half_range * torch.tanh(mean)

    # 学習中は平均の近くからランダムに行動を選び、新しい動きを試す。
    std = torch.ones_like(mean_in_range) * 0.20
    distribution = Normal(mean_in_range, std)
    raw_action = distribution.sample()
    action = torch.maximum(torch.minimum(raw_action, high), low)#rewrite

    # 今回選んだ行動が、方策からどれくらい出やすかったかを記録する。
    log_prob = distribution.log_prob(raw_action).sum()

    # EvoGymへ渡す行動（NumPy配列）と、後で学習に使うlog_probを返す。
    return action.squeeze(0).detach().numpy(), log_prob


#方策を更新する。良い行動と悪い行動に分け、良い行動の確率を上げる
"""def update_policy(optimizer, log_probs, returns):
    # 時刻ごとに別々だったTensorを、一つのTensorにまとめる。
    log_probs_tensor = torch.stack(log_probs)

    # 平均より良い行動を正、悪い行動を負にして、更新を安定させる。
    normalized_returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 0.000001)

    # 良いreturnなら、その行動のlog_probを大きくする方向に更新する損失。
    loss = -(log_probs_tensor * normalized_returns).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
"""
def update_policy(optimizer, log_probs, returns):
    # 1. log_probsをスタックしてテンソルにする
    log_probs_tensor = torch.stack(log_probs)

    # 2. returnsをテンソルにする際、GPUに送るコードを追加
    # log_probs_tensorと同じデバイス（GPU）に合わせるのが一番確実です
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=log_probs_tensor.device)
    # 3. 報酬の正規化（これを行うと学習が安定します）
    if len(returns_tensor) > 1:
        normalized_returns = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
    else:
        normalized_returns = returns_tensor

    # 4. 損失の計算
    # ここで「cuda:0 と cpu」が混ざっていたのがエラーの原因でした
    loss = -(log_probs_tensor * normalized_returns).sum()

    # 5. 重みの更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_body(task_name, body, episodes, max_steps, seed=7):
    """体ごとに新しいPolicyを作り、REINFORCEで指定回数学習する。"""
    torch.manual_seed(seed)
    env = gym.make(task_name, body=body, render_mode=None)
    first_obs, _ = env.reset(seed=seed)

    observation_size = len(first_obs)
    action_size = env.action_space.shape[0]
    policy = Policy(observation_size, action_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    training_returns = []

    for episode_number in range(episodes):
        obs, _ = env.reset()
        rewards = []
        log_probs = []
        episode_return = 0.0

        for step_number in range(max_steps):
            action, log_prob = choose_action(
                policy,
                obs,
                env.action_space.low,
                env.action_space.high,
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)
            episode_return = episode_return + reward
            obs = next_obs

            if terminated or truncated:
                break

        returns = calculate_returns(rewards, gamma=0.99)
        loss = update_policy(optimizer, log_probs, returns)
        training_returns.append(episode_return)
        print("train episode", episode_number, "return", episode_return, "loss", loss)

    env.close()
    return policy, training_returns


def evaluate_body(task_name, body, policy, episodes, max_steps):
    """重みを変えず、平均行動だけで実力を測り、平均報酬を返す。"""
    env = gym.make(task_name, body=body, render_mode=None)
    evaluation_returns = []

    for episode_number in range(episodes):
        obs, _ = env.reset(seed=100 + episode_number)
        episode_return = 0.0

        for step_number in range(max_steps):
            action = make_mean_action(
                policy,
                obs,
                env.action_space.low,
                env.action_space.high,
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_return = episode_return + reward
            obs = next_obs

            if terminated or truncated:
                break

        evaluation_returns.append(episode_return)

    env.close()
    return sum(evaluation_returns) / len(evaluation_returns)

def make_mean_action(policy, obs, low_array, high_array):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean = policy(obs_tensor)
    low = torch.tensor(low_array, dtype=torch.float32)
    high = torch.tensor(high_array, dtype=torch.float32)
    action = (low + high) / 2 + (high - low) / 2 * torch.tanh(mean)
    return action.squeeze(0).numpy()


###################追加16個の学習環境を構築#############
import torch
from torch.distributions import Normal

def choose_action_batch(policy, obs, envs, device):
    obs_tensor = torch.from_numpy(obs).float().to(device)
    
    # 1. Policyを実行
    outputs = policy(obs_tensor)
    
    # 2. 戻り値の数によって処理を分ける
    if isinstance(outputs, tuple) and len(outputs) == 2:
        # mu と sigma が両方返ってくるタイプの場合
        mu, sigma = outputs
    else:
        # mu だけが返ってくるタイプの場合
        mu = outputs
        # ★ 0.1 から 0.5 くらいに上げると、最初は激しく動いて探索しやすくなります
        sigma = torch.full_like(mu, 0.5)
    
    # 3. 分布を作ってサンプリング
    # sigmaが負にならないように softplus や exp をかける必要がある場合があります
    # ここでは単純に正の値であることを想定
    dist = Normal(mu, sigma)
    action = dist.sample()
    
    # log確率を計算
    log_prob = dist.log_prob(action).sum(dim=-1)
    
    action_np = action.cpu().detach().numpy()
    low = envs.single_action_space.low
    high = envs.single_action_space.high
    action_np = action_np.clip(low, high)
    
    return action_np, log_prob