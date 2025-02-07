from typing import Protocol

import torch


class Scheduler(Protocol):
    def schedule(self) -> torch.Tensor: ...


class LinearScheduler:
    def __init__(self, timesteps: int, start: float, end: float) -> None:
        self.timesteps = timesteps
        self.start = start
        self.end = end

    def schedule(self) -> torch.Tensor:
        return torch.linspace(self.start, self.end, self.timesteps)
