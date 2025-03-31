from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffusion.diffusion_factors import Factors


@dataclass
class SampleInputs:
    mean: torch.Tensor
    variance: torch.Tensor
    x_start: torch.Tensor
    epsilon: torch.Tensor
    factors: Factors
    timesteps: torch.Tensor


class Sampler(ABC):
    @abstractmethod
    def sample(self, inputs: SampleInputs) -> torch.Tensor:
        raise NotImplementedError


class DDPMSampler(Sampler):
    def sample(self, inputs: SampleInputs) -> torch.Tensor:
        mean = inputs.mean
        variance = inputs.variance
        timesteps = inputs.timesteps

        noise = torch.randn_like(mean).to(device=mean.device)

        mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + torch.sqrt(variance) * noise * mask

        return x_prev


class DDIMSampler(Sampler):
    def __init__(self, eta: float):
        self.eta = eta

    def sample(self, inputs: SampleInputs) -> torch.Tensor:
        x_start = inputs.x_start
        epsilon = inputs.epsilon
        variance = inputs.variance
        factors = inputs.factors
        timesteps = inputs.timesteps
        gammas_prev = factors.gammas_prev[timesteps]

        noise = torch.randn_like(x_start).to(device=x_start.device)
        sigma = self.eta * torch.sqrt(variance)

        mean = x_start * torch.sqrt(gammas_prev) + torch.sqrt(1 - gammas_prev - sigma**2) * epsilon
        mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + mask * sigma * noise

        return x_prev
