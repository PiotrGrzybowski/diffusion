from abc import ABC, abstractmethod

import torch


class TimestepSampler(ABC):
    def __init__(self, timesteps: int) -> None:
        super().__init__()
        self.timesteps = timesteps

    @abstractmethod
    def sample_like(self, x: torch.Tensor, device) -> torch.Tensor:
        pass

    @abstractmethod
    def full_like(self, x: torch.Tensor, timestep: int, device) -> torch.Tensor:
        pass


class UniformTimestepSampler(TimestepSampler):
    def sample_like(self, x: torch.Tensor, device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (x.size(0),)).to(device=device, dtype=torch.long)

    def full_like(self, x: torch.Tensor, timestep: int, device) -> torch.Tensor:
        return torch.full((x.size(0),), timestep, device=device, dtype=torch.long)
