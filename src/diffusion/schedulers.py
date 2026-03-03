import math
from typing import Protocol

import torch


class Scheduler(Protocol):
    def schedule(self) -> torch.Tensor: ...


class LinearScheduler:
    def __init__(self, timesteps: int, start: float, end: float) -> None:
        self.timesteps = timesteps
        self.scale = 1000 / timesteps
        self.start = start * self.scale
        self.end = end * self.scale

    def schedule(self) -> torch.Tensor:
        return torch.linspace(self.start, self.end, self.timesteps)


class CosineScheduler:
    def __init__(self, timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> None:
        self.timesteps = timesteps
        self.s = s
        self.max_beta = max_beta

    def schedule(self) -> torch.Tensor:
        steps = self.timesteps
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        f_t = torch.cos(((t / steps) + self.s) / (1 + self.s) * math.pi / 2) ** 2
        f_t = f_t / f_t[0]
        betas = 1 - (f_t[1:] / f_t[:-1])
        betas = torch.clamp(betas, min=1e-8, max=self.max_beta)
        return betas.to(dtype=torch.float32)
