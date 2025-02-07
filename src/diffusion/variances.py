from dataclasses import dataclass
from typing import Protocol

import torch

from diffusion.diffusion_factors import Factors


@dataclass(frozen=True)
class VarianceInputs:
    timesteps: torch.Tensor
    model_output: torch.Tensor | None = None


class VarianceStrategy(Protocol):
    def variance(self, inputs: VarianceInputs) -> torch.Tensor: ...

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor: ...


class FixedSmallVariance:
    def __init__(self, factors: Factors) -> None:
        self.factors = factors

    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        betas = self.factors.betas[inputs.timesteps]
        gammas = self.factors.gammas[inputs.timesteps]
        gammas_prev = self.factors.gammas_prev[inputs.timesteps]

        return betas * (1 - gammas_prev) / (1 - gammas)

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        timesteps = torch.where(inputs.timesteps == 0, torch.tensor(1), inputs.timesteps)
        inputs = VarianceInputs(timesteps, inputs.model_output)

        variance = self.variance(inputs)
        return torch.log(variance)


class FixedLargeVariance:
    def __init__(self, factors: Factors) -> None:
        self.factors = factors

    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return self.factors.betas[inputs.timesteps]

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return torch.log(self.variance(inputs))


class TrainableVariance:
    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        if inputs.model_output is None:
            raise ValueError("Model output must be provided for trainable variance")
        return torch.exp(inputs.model_output)

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        if inputs.model_output is None:
            raise ValueError("Model output must be provided for trainable variance")
        return inputs.model_output


class TrainableRangeVariance:
    def __init__(self, lower_variance: VarianceStrategy, upper_variance: VarianceStrategy) -> None:
        self.lower_variance = lower_variance
        self.uper_variance = upper_variance

    def variance(self, inputs: VarianceInputs) -> torch.Tensor:
        return torch.exp(self.log_variance(inputs))

    def log_variance(self, inputs: VarianceInputs) -> torch.Tensor:
        if inputs.model_output is None:
            raise ValueError("Model output must be provided for trainable range variance")
        v = (inputs.model_output + 1) / 2
        return v * self.uper_variance.log_variance(inputs) + (1 - v) * self.lower_variance.log_variance(inputs)
