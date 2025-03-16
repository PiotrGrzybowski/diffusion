import math

import torch

from diffusion.diffusion_terms import DiffusionTerms
from diffusion.gaussian_utils import discretized_gaussian_log_likelihood, gaussian_kl, prior_kl


def nll(target_terms: DiffusionTerms, predicted_terms: DiffusionTerms, timesteps: torch.Tensor, max_timesteps: int) -> torch.Tensor:
    loss = gaussian_kl(target_terms.mean, target_terms.log_variance, predicted_terms.mean, predicted_terms.log_variance)
    prior_nnl = prior_kl(target_terms.mean, target_terms.log_variance)
    decoder_nnl = -discretized_gaussian_log_likelihood(predicted_terms.x_start, predicted_terms.mean, predicted_terms.variance)

    decoder_idx = torch.where(timesteps == 0)
    prior_idx = torch.where(timesteps == max_timesteps - 1)

    loss[decoder_idx] = decoder_nnl[decoder_idx]
    loss[prior_idx] = prior_nnl[prior_idx]
    loss = loss.mean() / math.log(2.0)

    return loss
