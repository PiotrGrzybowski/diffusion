from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn.functional import mse_loss

from diffusion.diffusion_factors import Factors
from diffusion.gaussian_utils import discretized_gaussian_log_likelihood, gaussian_kl


@dataclass
class LossInputs:
    timesteps: torch.Tensor
    mean: torch.Tensor
    mean_objective: torch.Tensor
    variance: torch.Tensor
    log_variance: torch.Tensor
    predicted_mean: torch.Tensor
    predicted_mean_objective: torch.Tensor
    predicted_log_variance: torch.Tensor


class DiffusionLoss(Protocol):
    def forward(self, inputs: LossInputs) -> torch.Tensor: ...


class MeanMse:
    def __init__(self, factors: Factors):
        self.factors = factors

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        scale = 1 / 2 * inputs.variance

        return (scale * mse_loss(inputs.predicted_mean, inputs.mean)).mean()


class MeanMseSimple:
    def __init__(self, factors: Factors):
        self.factors = factors

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.predicted_mean, inputs.mean)


class XStartMse:
    def __init__(self, factors: Factors):
        self.factors = factors

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        scale = self.factors.gamma_prev * self.factors.betas**2 / (inputs.variance * (1 - self.factors.gammas) ** 2)
        return (scale * mse_loss(inputs.predicted_mean, inputs.mean)).mean()


class XStartMseSimple:
    def __init__(self, factors: Factors):
        self.factors = factors

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.predicted_mean_objective, inputs.mean_objective)


class NoiseMse:
    def __init__(self, factors: Factors):
        self.factors = factors

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        alphas = self.factors.alphas
        betas = self.factors.betas
        gammas = self.factors.gammas

        scale = betas**2 / (2 * inputs.variance * alphas * (1 - gammas))
        return (scale * mse_loss(inputs.predicted_mean_objective, inputs.mean_objective)).mean()


class NoiseMseSimple:
    def __init__(self, factors: Factors):
        self.factors = factors

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.predicted_mean_objective, inputs.mean_objective)


class VariationalBound:
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        loss = gaussian_kl(inputs.mean, inputs.log_variance, inputs.predicted_mean, inputs.predicted_log_variance)

        predicted_variance = torch.exp(inputs.predicted_log_variance)
        decoder_nnl = -discretized_gaussian_log_likelihood(inputs.mean, inputs.predicted_mean, predicted_variance)
        idx = torch.where(inputs.timesteps == 0)
        loss[idx] = decoder_nnl[idx]

        return loss.mean()


class HybridLoss:
    def __init__(self, factors: Factors, mean_loss: DiffusionLoss, variance_loss: DiffusionLoss, omega: float):
        self.factors = factors
        self.mean_loss = mean_loss
        self.variance_loss = variance_loss
        self.omega = omega

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        frozen_mean = inputs.mean.detach()
        frozen_mean_objective = inputs.mean_objective.detach()
        mean_loss = self.mean_loss.forward(inputs)

        inputs.mean = frozen_mean
        inputs.mean_objective = frozen_mean_objective
        variance_loss = self.variance_loss.forward(inputs)

        return mean_loss + self.omega * variance_loss
