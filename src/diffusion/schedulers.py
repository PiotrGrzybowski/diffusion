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
