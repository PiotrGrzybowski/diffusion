from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import mse_loss

from diffusion.diffusion_factors import Factors
from diffusion.diffusion_terms import DiffusionTerms
from diffusion.gaussian_utils import discretized_gaussian_log_likelihood, gaussian_kl


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the mean of a tensor while flattening it."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


@dataclass
class LossInputs:
    target: DiffusionTerms
    predicted: DiffusionTerms
    factors: Factors
    timesteps: torch.Tensor


class DiffusionLoss(ABC):
    @abstractmethod
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        raise NotImplementedError


class MseMeanDirect(DiffusionLoss):
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        scale = 1 / 2 * inputs.target.variance

        return scale * mse_loss(inputs.target.mean, inputs.predicted.mean)


class MseMeanDirectSimple(DiffusionLoss):
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.target.mean, inputs.predicted.mean)


class MseMeanXStart(DiffusionLoss):
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        timesteps = inputs.timesteps
        factors = inputs.factors

        betas = factors.betas[timesteps]
        gammas = factors.gammas[timesteps]
        gammas_prev = factors.gammas_prev[timesteps]

        scale = gammas_prev * betas**2 / (inputs.target.variance * (1 - gammas) ** 2)
        return (scale * mse_loss(inputs.target.mean, inputs.predicted.mean)).mean()


class MseMeanXStartSimple(DiffusionLoss):
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.target.x_start, inputs.predicted.x_start)


class MseMeanEpsilon(DiffusionLoss):
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        timesteps = inputs.timesteps
        factors = inputs.factors

        alphas = factors.alphas[timesteps]
        betas = factors.betas[timesteps]
        gammas = factors.gammas[timesteps]

        scale = betas**2 / (2 * inputs.target.variance * alphas * (1 - gammas))
        return (scale * mse_loss(inputs.target.epsilon, inputs.predicted.epsilon)).mean()


class MseMeanEpsilonSimple(DiffusionLoss):
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        return mse_loss(inputs.target.epsilon, inputs.predicted.epsilon)


class VLB(DiffusionLoss):
    def forward(self, inputs: LossInputs) -> torch.Tensor:
        kl_loss = gaussian_kl(inputs.target.mean, inputs.target.log_variance, inputs.predicted.mean, inputs.predicted.log_variance)
        decoder_nnl = -discretized_gaussian_log_likelihood(inputs.target.x_start, inputs.predicted.mean, inputs.predicted.log_variance)

        kl_loss = mean_flat(kl_loss) / np.log(2.0)
        decoder_nnl = mean_flat(decoder_nnl) / np.log(2.0)

        idx = torch.where(inputs.timesteps == 0)
        kl_loss[idx] = decoder_nnl[idx]

        return kl_loss.mean()


class Hybrid(DiffusionLoss):
    def __init__(self, mean_loss: DiffusionLoss, variance_loss: DiffusionLoss, omega: float):
        self.mean_loss = mean_loss
        self.variance_loss = variance_loss
        self.omega = omega

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        frozen_mean = inputs.predicted.mean.detach()
        mean_loss = self.mean_loss.forward(inputs)

        inputs.predicted_mean = frozen_mean
        variance_loss = self.variance_loss.forward(inputs)

        return mean_loss + self.omega * variance_loss
