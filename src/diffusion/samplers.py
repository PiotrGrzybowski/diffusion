from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffusion.diffusion_factors import Factors
from diffusion.diffusion_terms import DiffusionTerms


@dataclass
class SampleOutputs:
    mean: torch.Tensor
    x_start: torch.Tensor
    epsilon: torch.Tensor
    x_prev: torch.Tensor


class Sampler(ABC):
    @abstractmethod
    def sample(self, outputs: DiffusionTerms, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DDPMSampler(Sampler):
    def sample(self, outputs: DiffusionTerms, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
        mean = outputs.mean
        variance = outputs.variance

        noise = torch.randn_like(mean).to(device=mean.device)

        mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + torch.sqrt(variance) * noise * mask

        return x_prev


class DDIMSampler(Sampler):
    def __init__(self, eta: float):
        self.eta = eta

    def sample(self, outputs: DiffusionTerms, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
        x_start = outputs.x_start
        variance = outputs.variance

        noise = torch.randn_like(x_start).to(device=x_start.device)
        sigma = self.eta * torch.sqrt(variance)

        mean = x_start * torch.sqrt(factors.gammas_prev) + torch.sqrt(1 - factors.gammas_prev - sigma**2) * noise
        mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + mask * sigma * noise

        return x_prev
