from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffusion.diffusion_factors import Factors


@dataclass
class VarianceInputs:
    factors: Factors
    timesteps: torch.Tensor
    model_output: torch.Tensor


class VarianceStrategy(ABC):
    @abstractmethod
    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        raise NotImplementedError


class FixedSmallVariance(VarianceStrategy):
    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        factors = inputs.factors

        betas = factors.betas[inputs.timesteps]
        gammas = factors.gammas[inputs.timesteps]
        gammas_prev = factors.gammas_prev[inputs.timesteps]

        return betas * (1 - gammas_prev) / (1 - gammas)

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        timesteps = torch.where(inputs.timesteps == 0, torch.tensor(1), inputs.timesteps)
        inputs.timesteps = timesteps

        variance = self.variance(inputs)
        return torch.log(variance)


class FixedLargeVariance(VarianceStrategy):
    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return inputs.factors.betas[inputs.timesteps]

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return torch.log(self.variance(inputs))


class DirectVariance(VarianceStrategy):
    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return torch.exp(inputs.model_output)

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return inputs.model_output


class TrainableRangeVariance(VarianceStrategy):
    def __init__(self, lower_variance: VarianceStrategy, upper_variance: VarianceStrategy) -> None:
        self.lower_variance = lower_variance
        self.uper_variance = upper_variance

    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return torch.exp(self.log_variance(inputs))

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        v = (inputs.model_output + 1) / 2
        return v * self.uper_variance.log_variance(inputs) + (1 - v) * self.lower_variance.log_variance(inputs)
