from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from diffusion.diffusion_factors import Factors


class PosteriorMean(ABC):
    @abstractmethod
    def mean(self, x_start: torch.Tensor, x_t: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DDPMMean(PosteriorMean):
    def mean(self, x_start: torch.Tensor, x_t: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
        gammas = factors.gammas[timesteps]
        gammas_prev = factors.gammas_prev[timesteps]
        alphas = factors.alphas[timesteps]
        betas = factors.betas[timesteps]

        x_t_coeff = torch.sqrt(alphas) * (1 - gammas_prev) / (1 - gammas)
        x_start_coeff = torch.sqrt(gammas_prev) * betas / (1 - gammas)

        return x_t_coeff * x_t + x_start_coeff * x_start


class DDIMMean(PosteriorMean):
    def mean(self, x_start: torch.Tensor, x_t: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
        betas = factors.betas[timesteps]
        gammas = factors.gammas[timesteps]
        gammas_prev = factors.gammas_prev[timesteps]

        epsilon = (x_t - torch.sqrt(gammas) * x_start) / torch.sqrt(1 - gammas)
        variance = (1 - gammas_prev) * betas / (1 - gammas)

        return torch.sqrt(gammas_prev) * x_start + torch.sqrt(1 - gammas_prev - variance) * epsilon


def epsilon_from_xstart(x_start: torch.Tensor, x_t: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    gammas = factors.gammas[timesteps]
    return (x_t - torch.sqrt(gammas) * x_start) / torch.sqrt(1 - gammas)


def x_start_from_mean(x_t: torch.Tensor, mean: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    gammas = factors.gammas[timesteps]
    gammas_prev = factors.gammas_prev[timesteps]
    alphas = factors.alphas[timesteps]
    betas = factors.betas[timesteps]

    x_t_coeff = torch.sqrt(alphas) * (1 - gammas_prev) / (1 - gammas)
    x_0_coeff = torch.sqrt(gammas_prev) * betas / (1 - gammas)

    return (mean - x_t_coeff * x_t) / x_0_coeff


def x_start_from_epsilon(x_t: torch.Tensor, epsilon: torch.Tensor, factors: Factors, timesteps: torch.Tensor):
    gammas = factors.gammas[timesteps]
    return (x_t - torch.sqrt(1 - gammas) * epsilon) / torch.sqrt(gammas)
    # return x_t / torch.sqrt(gammas) - epsilon * torch.sqrt(1 / gammas - 1)


def mean_from_xstart(x_start: torch.Tensor, x_t: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    gammas = factors.gammas[timesteps]
    gammas_prev = factors.gammas_prev[timesteps]
    alphas = factors.alphas[timesteps]
    betas = factors.betas[timesteps]

    x_t_coeff = torch.sqrt(alphas) * (1 - gammas_prev) / (1 - gammas)
    x_start_coeff = torch.sqrt(gammas_prev) * betas / (1 - gammas)

    return x_t_coeff * x_t + x_start_coeff * x_start


def mean_from_epsilon(x_t: torch.Tensor, epsilon: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    alphas = factors.alphas[timesteps]
    betas = factors.betas[timesteps]
    gammas = factors.gammas[timesteps]

    return 1 / torch.sqrt(alphas) * (x_t - betas / torch.sqrt(1 - gammas) * epsilon)


@dataclass
class MeanOutputs:
    mean: torch.Tensor
    x_start: torch.Tensor
    epsilon: torch.Tensor


@dataclass
class MeanInputs:
    x_t: torch.Tensor
    model_output: torch.Tensor
    factors: Factors
    timesteps: torch.Tensor


class MeanStrategy(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: MeanInputs) -> MeanOutputs:
        raise NotImplementedError


class DirectMean(MeanStrategy):
    def forward(self, inputs: MeanInputs) -> MeanOutputs:
        x_t = inputs.x_t
        mean = inputs.model_output
        factors = inputs.factors
        timesteps = inputs.timesteps

        x_start = x_start_from_mean(x_t, mean, factors, timesteps)

        if not self.training:
            x_start = x_start.clamp(-1.0, 1.0)

        epsilon = epsilon_from_xstart(x_start, x_t, factors, timesteps)

        return MeanOutputs(mean, x_start, epsilon)


class XStartMean(MeanStrategy):
    def forward(self, inputs: MeanInputs) -> MeanOutputs:
        x_t = inputs.x_t
        x_start = inputs.model_output
        factors = inputs.factors
        timesteps = inputs.timesteps

        if not self.training:
            x_start = x_start.clamp(-1.0, 1.0)

        mean = mean_from_xstart(x_start, x_t, factors, timesteps)
        epsilon = epsilon_from_xstart(x_start, x_t, factors, timesteps)

        return MeanOutputs(mean, x_start, epsilon)


class EpsilonMean(MeanStrategy):
    def forward(self, inputs: MeanInputs) -> MeanOutputs:
        x_t = inputs.x_t
        epsilon = inputs.model_output
        factors = inputs.factors
        timesteps = inputs.timesteps

        x_start = x_start_from_epsilon(x_t, epsilon, factors, timesteps)

        if not self.training:
            x_start = x_start.clamp(-1.0, 1.0)

        mean = mean_from_xstart(x_start, x_t, factors, timesteps)

        return MeanOutputs(mean, x_start, epsilon)
