from enum import Enum

import torch
from torch.nn.functional import mse_loss

from diffusion.diffusion_factors import Factors
from diffusion.gaussian_utils import discretized_gaussian_log_likelihood, gaussian_kl


class LossType(Enum):
    MEAN_MSE = "MeanMSE"
    MEAN_SIMPLE_MSE = "MeanSimpleMSE"
    NOISE_MSE = "NoiseMSE"
    NOISE_SIMPLE_MSE = "NoiseSimpleMSE"
    SIMPLE_MSE = "SimpleMSE"
    VARIATIONAL_BOUND = "VariationalBound"
    HYBRID = "Hybrid"


def mse_mean_loss(predicted_mean: torch.Tensor, target_mean: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    posterior_variance = factors.betas * (1 - factors.gammas_prev) / (1 - factors.gammas)
    posterior_variance = torch.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]])[timesteps]

    scale = 1 / 2 * posterior_variance

    return (scale * mse_loss(predicted_mean, target_mean)).mean()


def simple_mse_mean_loss(predicted_mean: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
    return mse_loss(predicted_mean, target_mean)


def mse_noise_loss(predicted_noise: torch.Tensor, target_noise: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    posterior_variance = factors.betas * (1 - factors.gammas_prev) / (1 - factors.gammas)
    posterior_variance = torch.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]])[timesteps]
    alphas = factors.alphas[timesteps]
    betas = factors.betas[timesteps]
    gammas = factors.gammas[timesteps]

    scale = betas**2 / (2 * posterior_variance * alphas * (1 - gammas))
    return (scale * mse_loss(predicted_noise, target_noise)).mean()


def simple_mse_noise_loss(predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    return mse_loss(predicted_noise, target_noise)


def mean_mse_loss(predicted_mean: torch.Tensor, target_mean: torch.Tensor, factors: Factors, timesteps: torch.Tensor) -> torch.Tensor:
    posterior_variance = factors.betas * (1 - factors.gammas_prev) / (1 - factors.gammas)
    posterior_variance = torch.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]])[timesteps]
    alphas = factors.alphas[timesteps]
    betas = factors.betas[timesteps]
    gammas = factors.gammas[timesteps]

    scale = betas**2 / (2 * posterior_variance * alphas * (1 - gammas))
    return (scale * mse_loss(predicted_mean, target_mean)).mean()


def mean_simple_mse_loss(predicted_mean: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
    return mse_loss(predicted_mean, target_mean)


def variational_bound_loss(
    target_mean: torch.Tensor,
    target_log_variance: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_log_variance: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    loss = gaussian_kl(target_mean, target_log_variance, predicted_mean, predicted_log_variance)

    predicted_variance = torch.exp(predicted_log_variance)
    decoder_nnl = -discretized_gaussian_log_likelihood(target_mean, predicted_mean, predicted_variance)
    idx = torch.where(timesteps == 0)
    loss[idx] = decoder_nnl[idx]

    return loss.mean()


def hybrid_loss(
    target_mean: torch.Tensor,
    target_variance: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_variance: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    simple = mean_simple_mse_loss(target_mean, predicted_mean)
    variational = variational_bound_loss(target_mean, target_variance, predicted_mean, predicted_variance, timesteps)
    return simple + 1e-3 * variational


class LossManager:
    def forward(
        self,
        loss_type: LossType,
        timesteps: torch.Tensor,
        factors: Factors,
        target_mean: torch.Tensor,
        target_variance: torch.Tensor,
        target_noise: torch.Tensor,
        predicted_mean: torch.Tensor,
        predicted_variance: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        match loss_type:
            case LossType.MEAN_MSE:
                return mean_mse_loss(predicted_mean, target_mean, factors, timesteps)
            case LossType.MEAN_SIMPLE_MSE:
                return mean_simple_mse_loss(predicted_mean, target_mean)
            case LossType.NOISE_MSE:
                return mse_noise_loss(predicted_noise, target_noise, factors, timesteps)
            case LossType.NOISE_SIMPLE_MSE:
                return simple_mse_noise_loss(predicted_noise, target_noise)
            case LossType.VARIATIONAL_BOUND:
                return variational_bound_loss(target_mean, target_variance, predicted_mean, predicted_variance, timesteps)
            case LossType.HYBRID:
                return hybrid_loss(target_mean, target_variance, predicted_mean, predicted_variance, timesteps)
            case _:
                raise ValueError(f"Unknown loss type: {loss_type}")


#
# if __name__ == "__main__":
#     loss_manager = LossManager()
#
#     # Create dummy data
#     target = torch.randn(4, 3, 64, 64)
#     predicted = torch.randn(4, 3, 64, 64)
#     timesteps = torch.randint(0, 3, (4,))
#     scheduler = LinearScheduler(1000, 0.001, 0.02)
#     factors = Factors(betas=scheduler.schedule())
#
#     # Test VariationalBoundLoss
#     target_mean = torch.randn(4, 3, 64, 64)
#     target_variance = torch.rand(4, 3, 64, 64).abs()
#     predicted_mean = torch.randn(4, 3, 64, 64)
#     predicted_variance = target_variance
#
#     loss = loss_manager.forward(
#         LossType.VARIATIONAL_BOUND, timesteps, factors, target_mean, target_variance, predicted_mean, predicted_variance
#     )
#     print(f"VariationalBoundLoss: {loss}")
#
#     loss = loss_manager.forward(LossType.SIMPLE_MSE, timesteps, factors, target_mean, target_variance, predicted_mean, predicted_variance)
#     print(f"SimpleMSELoss: {loss}")
#
#     loss = loss_manager.forward(LossType.MEAN_MSE, timesteps, factors, target_mean, target_variance, predicted_mean, predicted_variance)
#     print(f"ScaledMSELoss: {loss}")
#
#     loss = loss_manager.forward(LossType.HYBRID, timesteps, factors, target_mean, target_variance, predicted_mean, predicted_variance)
#     print(f"HybridLoss: {loss}")
