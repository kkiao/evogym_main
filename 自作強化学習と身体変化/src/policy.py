#状態に応じて、筋肉をどう動かすか（どの様に行動するか）を出す/方策と呼ばれる
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()

        # 状態の数字を受け取り、各筋肉の行動の平均値を出す部品。
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
        )

    def forward(self, observation_tensor):
        return self.network(observation_tensor)