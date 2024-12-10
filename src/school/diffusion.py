import torch
import torch.nn as nn

from school.factors import Factors
from school.schedulers import Scheduler


class GaussianDiffusion(nn.Module):
    """
    Implements the Gaussian diffusion process for generative models.

    Args:
        timesteps (int): Number of timesteps in the diffusion process.
        scheduler (Scheduler): Scheduler to handle noise scheduling (e.g., betas, alphas).
        in_channels (int): Number of input channels for the model.
    """

    def __init__(self, timesteps: int, scheduler: Scheduler, in_channels) -> None:
        super().__init__()
        self.factors = Factors(scheduler.schedule())
        self.timesteps_count = timesteps
        self.in_channels = in_channels

    def q_mean(self, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean of the forward diffusion process q(x_t | x_0).

        Args:
            x_start (torch.Tensor): Original input tensor (x_0).
            timesteps (torch.Tensor): Timesteps for which to compute the mean.

        Returns:
            torch.Tensor: Mean of the forward diffusion process at given timesteps.
        """
        return torch.sqrt(self.factors.gammas[timesteps]) * x_start

    def q_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute the variance of the forward diffusion process q(x_t | x_0).

        Args:
            timesteps (torch.Tensor): Timesteps for which to compute the variance.

        Returns:
            torch.Tensor: Variance of the forward diffusion process at given timesteps.
        """
        return 1 - self.factors.gammas[timesteps]

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Sample from the forward diffusion process q(x_t | x_0).

        Args:
            x_start (torch.Tensor): Original input tensor (x_0).
            timesteps (torch.Tensor): Timesteps for which to sample x_t.
            noise (torch.Tensor): Gaussian noise to perturb x_0.

        Returns:
            torch.Tensor: Sampled x_t at the given timesteps.

        Note:
            The `noise` argument is passed explicitly because `q_sample` may be used during
            loss calculation, where it is important to sample x_t for a given reference noise.
            Having direct access to the noise ensures reproducibility and precise control.
        """
        return self.q_mean(x_start, timesteps) + torch.sqrt(self.q_variance(timesteps)) * noise
