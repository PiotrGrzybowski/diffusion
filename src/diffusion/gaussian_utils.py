import math

import torch

from diffusion.diffusion_factors import Factors
from diffusion.schedulers import LinearScheduler
from diffusion.variances import FixedSmallVariance, VarianceInputs


def gaussian_kl(mean_p: torch.Tensor, log_var_p: torch.Tensor, mean_q: torch.Tensor, log_var_q: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence between two gaussians.
    Args:
        mean_p (torch.Tensor): The mean of the first Gaussian.
        var_p (torch.Tensor): The variance of the first Gaussian.
        mean_q (torch.Tensor): The mean of the second Gaussian.
        var_q (torch.Tensor): The variance of the second Gaussian.

    Returns:
        torch.Tensor: The KL divergence between the two Gaussians.
    """

    return 0.5 * (-1.0 + log_var_q - log_var_p + torch.exp(log_var_p - log_var_q) + ((mean_p - mean_q) ** 2) * torch.exp(-log_var_q))


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """A fast approximation of the cumulative distribution function of the standard normal.
    Args:
        x (torch.Tensor): The input tensor to compute the CDF for.

    Returns:
        torch.Tensor: The CDF of the input tensor.
    """
    return 0.5 * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0) / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x_start: torch.Tensor, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Compute the log-likelihood of a Gaussian distribution discretizing to a given image.
    Args:
        x (torch.Tensor): target images. It is assumed torch.Tensor was uint8 values, rescaled to torch.Tensor range [-1, 1].
        mean (torch.Tensor): Gaussian mean Tensor.
        variance (torch.Tensor): Gaussian variance Tensor.
    Returns:
        torch.Tensor: a tensor like x of log probabilities (in nats).
    """

    std = torch.sqrt(torch.exp(log_variance))
    step = 1.0 / 255.0
    centered_x = x_start - mean
    upper = (centered_x + step) / std
    lower = (centered_x - step) / std

    cdf_upper = approx_standard_normal_cdf(upper)
    cdf_lower = approx_standard_normal_cdf(lower)
    cdf_delta = cdf_upper - cdf_lower

    likelihood = torch.where(x_start < -0.999, cdf_upper, torch.where(x_start > 0.999, 1 - cdf_lower, cdf_delta))
    li = likelihood.mean() / math.log(2.0)

    return torch.log(likelihood.clamp(min=1e-12))


def prior_kl(mean_p: torch.Tensor, log_var_p: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence between given and Normal Gaussians"""
    mean_q = torch.zeros_like(mean_p, device=mean_p.device)
    log_var_q = torch.zeros_like(log_var_p, device=mean_p.device)

    return gaussian_kl(mean_p, log_var_p, mean_q, log_var_q)


if __name__ == "__main__":
    import torch

    # x_start, mean, variance = torch.load("bundle.pt")
    # d = discretized_gaussian_log_likelihood(x_start, mean, variance)

    var = FixedSmallVariance()
    scheduler = LinearScheduler(1000, 0.0001, 0.02)
    factors = Factors(scheduler.schedule())
    out = torch.empty(0)
    inputs = VarianceInputs(factors, torch.tensor([0]), out)

    print(f"variance: {var.variance(inputs)}")
    print(f"log_variance: {var.log_variance(inputs)}")
    log_variance = var.log_variance(inputs)

    x_start = torch.tensor([1])
    mean = torch.tensor([1.05])

    print(discretized_gaussian_log_likelihood(x_start, mean, log_variance))
