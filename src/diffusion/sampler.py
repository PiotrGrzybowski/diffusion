from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffusion.diffusion_factors import Factors
from diffusion.means import MeanOutputs


@dataclass
class SampleOutputs:
    mean: torch.Tensor
    x_start: torch.Tensor
    epsilon: torch.Tensor
    x_prev: torch.Tensor


class Sampler(ABC):
    @abstractmethod
    def sample(self, mean_outputs: MeanOutputs, variance: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> SampleOutputs:
        raise NotImplementedError


class DDPMSampler(Sampler):
    def sample(self, mean_outputs: MeanOutputs, variance: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> SampleOutputs:
        mean = mean_outputs.mean
        x_start = mean_outputs.x_start
        epsilon = mean_outputs.epsilon

        noise = torch.randn_like(mean).to(device=mean.device)

        mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + torch.sqrt(variance) * noise * mask

        return SampleOutputs(mean, x_start, epsilon, x_prev)


class DDIMSampler(Sampler):
    def __init__(self, eta: float):
        self.eta = eta

    def sample(self, mean_outputs: MeanOutputs, variance: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> SampleOutputs:
        x_start = mean_outputs.x_start
        epsilon = mean_outputs.epsilon

        noise = torch.randn_like(x_start).to(device=x_start.device)
        sigma = self.eta * torch.sqrt(variance)

        mean = x_start * torch.sqrt(factors.gammas_prev) + torch.sqrt(1 - factors.gammas_prev - sigma**2) * noise
        mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + mask * sigma * noise

        return SampleOutputs(mean, x_start, epsilon, x_prev)
