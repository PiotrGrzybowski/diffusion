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


class ImageSampler(ABC):
    @abstractmethod
    def sample(self, inputs: SampleInputs) -> torch.Tensor:
        raise NotImplementedError


class DDPMSampler(ImageSampler):
    def sample(self, inputs: SampleInputs) -> torch.Tensor:
        mean = inputs.mean
        variance = inputs.variance
        timesteps = inputs.timesteps

        noise = torch.randn_like(mean).to(device=mean.device)

        mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        x_prev = mean + torch.sqrt(variance) * noise * mask

        return x_prev
