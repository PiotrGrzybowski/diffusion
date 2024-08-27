import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, size: int, theta: float = 10000) -> None:
        super().__init__()
        self.size = size
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = math.log(self.theta) / (self.size // 2 - 1)
        embedding = torch.exp(torch.arange(self.size // 2).to(x) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, init_dim: int, time_dim: int, theta: float = 10000) -> None:
        super().__init__()
        self.positional_embedding = SinusoidalPositionalEmbedding(init_dim, theta)
        self.linear1 = nn.Linear(init_dim, time_dim)
        self.linear2 = nn.Linear(time_dim, time_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_embedding(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
