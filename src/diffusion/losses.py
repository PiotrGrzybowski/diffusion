from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch.nn.functional import mse_loss

from diffusion.diffusion_factors import Factors
from diffusion.gaussian_utils import discretized_gaussian_log_likelihood, gaussian_kl


@dataclass
class LossInputs:
    factors: Factors
    timesteps: torch.Tensor
    mean: torch.Tensor
    mean_objective: torch.Tensor
    variance: torch.Tensor
    log_variance: torch.Tensor
    predicted_mean: torch.Tensor
    predicted_mean_objective: torch.Tensor
    predicted_log_variance: torch.Tensor


@runtime_checkable
class DiffusionLoss(Protocol):
    def forward(self, inputs: LossInputs) -> torch.Tensor: ...


class MseMeanDirect:
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        scale = 1 / 2 * inputs.variance

        return (scale * mse_loss(inputs.predicted_mean, inputs.mean)).mean()


class MseMeanDirectSimple:
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.predicted_mean, inputs.mean)


class MseMeanXStart:
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        timesteps = inputs.timesteps
        factors = inputs.factors

        betas = factors.betas[timesteps]
        gammas = factors.gammas[timesteps]
        gammas_prev = factors.gammas_prev[timesteps]

        scale = gammas_prev * betas**2 / (inputs.variance * (1 - gammas) ** 2)
        return (scale * mse_loss(inputs.predicted_mean, inputs.mean)).mean()


class MseMeanXStartSimple:
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.predicted_mean_objective, inputs.mean_objective)


class MseMeanEpsilon:
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        timesteps = inputs.timesteps
        factors = inputs.factors

        alphas = factors.alphas[timesteps]
        betas = factors.betas[timesteps]
        gammas = factors.gammas[timesteps]

        scale = betas**2 / (2 * inputs.variance * alphas * (1 - gammas))
        return (scale * mse_loss(inputs.predicted_mean_objective, inputs.mean_objective)).mean()


class MseMeanEpsilonSimple:
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.predicted_mean_objective, inputs.mean_objective)


class VLB:
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        loss = gaussian_kl(inputs.mean, inputs.log_variance, inputs.predicted_mean, inputs.predicted_log_variance)

        predicted_variance = torch.exp(inputs.predicted_log_variance)
        decoder_nnl = -discretized_gaussian_log_likelihood(inputs.mean, inputs.predicted_mean, predicted_variance)
        idx = torch.where(inputs.timesteps == 0)
        loss[idx] = decoder_nnl[idx]

        return loss.mean()


class Hybrid:
    def __init__(self, mean_loss: DiffusionLoss, variance_loss: DiffusionLoss, omega: float):
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
