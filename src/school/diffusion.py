import torch
import torch.nn as nn

from school.factors import Factors
from school.schedulers import Scheduler


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int, scheduler: Scheduler, in_channels) -> None:
        super().__init__()
        self.factors = Factors(scheduler.schedule())
        self.timesteps_count = timesteps
        self.in_channels = in_channels

    def q_mean(self, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.factors.gammas[timesteps]) * x_start

    def q_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return 1 - self.factors.gammas[timesteps]

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Implements q(x_t | x_0), samples x_t from x_0 for specified timesteps and sampled noise."""
        return self.q_mean(x_start, timesteps) + torch.sqrt(self.q_variance(timesteps)) * noise
