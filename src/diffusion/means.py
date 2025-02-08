from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from diffusion.diffusion_factors import Factors


@dataclass(frozen=True)
class MeanInputs:
    factors: Factors
    timesteps: torch.Tensor
    x_t: torch.Tensor
    mean_objective: torch.Tensor


@dataclass(frozen=True)
class MeanObjectives:
    mean: torch.Tensor
    x_start: torch.Tensor
    epsilon: torch.Tensor


@runtime_checkable
class MeanStrategy(Protocol):
    def mean(self, inputs: MeanInputs) -> torch.Tensor: ...

    def mean_objective(self, inputs: MeanObjectives) -> torch.Tensor: ...


class DirectMean:
    def mean(self, inputs: MeanInputs) -> torch.Tensor:
        return inputs.mean_objective

    def mean_objective(self, inputs: MeanObjectives) -> torch.Tensor:
        return inputs.mean


class XStartMean:
    def mean(self, inputs: MeanInputs) -> torch.Tensor:
        timesteps = inputs.timesteps
        factors = inputs.factors

        gammas = factors.gammas[timesteps]
        gammas_prev = factors.gammas_prev[timesteps]
        alphas = factors.alphas[timesteps]
        betas = factors.betas[timesteps]

        x_0 = inputs.mean_objective
        x_t = inputs.x_t

        x_t_coeff = torch.sqrt(alphas) * (1 - gammas_prev) / (1 - gammas)
        x_0_coeff = torch.sqrt(gammas_prev) * betas / (1 - gammas)

        return x_t_coeff * x_t + x_0_coeff * x_0

    def mean_objective(self, inputs: MeanObjectives) -> torch.Tensor:
        return inputs.x_start


class EpsilonMean:
    def mean(self, inputs: MeanInputs) -> torch.Tensor:
        timesteps = inputs.timesteps
        factors = inputs.factors

        alphas = factors.alphas[timesteps]
        betas = factors.betas[timesteps]
        gammas = factors.gammas[timesteps]

        x_t = inputs.x_t
        epsilon = inputs.mean_objective

        return 1 / torch.sqrt(alphas) * (x_t - betas / torch.sqrt(1 - gammas) * epsilon)

    def mean_objective(self, inputs: MeanObjectives) -> torch.Tensor:
        return inputs.epsilon
