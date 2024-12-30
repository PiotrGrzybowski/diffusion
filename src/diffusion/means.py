from dataclasses import dataclass
from typing import Protocol

import torch

from diffusion.diffusion_factors import Factors


@dataclass(frozen=True)
class MeanInputs:
    timesteps: torch.Tensor
    model_output: torch.Tensor
    x_t: torch.Tensor


@dataclass(frozen=True)
class MeanObjectives:
    mean: torch.Tensor
    x_start: torch.Tensor
    epsilon: torch.Tensor


class MeanStrategy(Protocol):
    def mean(self, inputs: MeanInputs) -> torch.Tensor: ...

    def mean_objective(self, inputs: MeanObjectives) -> torch.Tensor: ...


class DirectMean:
    def mean(self, inputs: MeanInputs) -> torch.Tensor:
        return inputs.model_output

    def mean_objective(self, inputs: MeanObjectives) -> torch.Tensor:
        return inputs.mean


class EpsilonMean:
    def __init__(self, factors: Factors) -> None:
        self.factors = factors

    def mean(self, inputs: MeanInputs) -> torch.Tensor:
        return self._mean_from_epsilon(inputs.timesteps, inputs.x_t, inputs.model_output)

    def mean_objective(self, inputs: MeanObjectives) -> torch.Tensor:
        return inputs.epsilon

    def _mean_from_epsilon(self, timesteps: torch.Tensor, x_t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps
        alphas = self.factors.alphas[timesteps]
        betas = self.factors.betas[timesteps]
        gammas = self.factors.gammas[timesteps]

        return 1 / torch.sqrt(alphas) * (x_t - betas / torch.sqrt(1 - gammas) * epsilon)


class XStartMean:
    def __init__(self, factors: Factors) -> None:
        self.factors = factors

    def mean(self, inputs: MeanInputs) -> torch.Tensor:
        return self._mean_from_x_start(inputs.timesteps, inputs.x_t, inputs.model_output)

    def mean_objective(self, inputs: MeanObjectives) -> torch.Tensor:
        return inputs.x_start

    def _mean_from_x_start(self, timesteps: torch.Tensor, x_t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        gammas = self.factors.gammas[timesteps]
        gammas_prev = self.factors.gammas_prev[timesteps]
        alphas = self.factors.alphas[timesteps]
        betas = self.factors.betas[timesteps]

        x_t_coeff = torch.sqrt(alphas) * (1 - gammas_prev) / (1 - gammas)
        x_0_coeff = torch.sqrt(gammas_prev) * betas / (1 - gammas)

        return x_t_coeff * x_t + x_0_coeff * x_0
