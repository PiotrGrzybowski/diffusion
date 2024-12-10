import pytest
import torch
from school.diffusion import GaussianDiffusion
from school.factors import Factors
from school.schedulers import LinearScheduler
from torch import nn


@pytest.fixture
def diffusion():
    """Fixture to initialize GaussianDiffusion with real Factors and Scheduler."""
    timesteps = 1000
    start = 0.0001
    end = 0.02

    scheduler = LinearScheduler(timesteps, start, end)

    return GaussianDiffusion(timesteps=timesteps, scheduler=scheduler, in_channels=3)


def test_gaussian_diffusion_initialization(diffusion: GaussianDiffusion):
    """Test the initialization of the GaussianDiffusion class."""
    assert isinstance(diffusion, nn.Module), "Factors should be initialized correctly."
    assert isinstance(diffusion.factors, Factors), "Factors should be initialized correctly."
    assert diffusion.timesteps_count == 1000, "Timesteps count is incorrect."
    assert diffusion.in_channels == 3, "Input channels value is incorrect."


def test_q_mean(diffusion: GaussianDiffusion):
    """Test the q_mean function."""
    x_start = torch.rand(2, 3, 64, 64)
    timesteps = torch.tensor([0, 999])
    q_mean_result = diffusion.q_mean(x_start, timesteps)

    expected_q_mean = torch.sqrt(diffusion.factors.gammas[timesteps]) * x_start
    assert torch.allclose(q_mean_result, expected_q_mean), "q_mean computation is incorrect."


def test_q_variance(diffusion: GaussianDiffusion):
    """Test the q_variance function."""
    timesteps = torch.tensor([0, 999])
    q_variance_result = diffusion.q_variance(timesteps)

    expected_q_variance = 1 - diffusion.factors.gammas[timesteps]
    assert torch.allclose(q_variance_result, expected_q_variance), "q_variance computation is incorrect."


def test_q_sample(diffusion: GaussianDiffusion):
    """Test the q_sample function."""
    x_start = torch.rand(2, 3, 64, 64)
    timesteps = torch.tensor([0, 999])
    noise = torch.randn_like(x_start)
    q_sample_result = diffusion.q_sample(x_start, timesteps, noise)

    q_mean = torch.sqrt(diffusion.factors.gammas[timesteps]) * x_start
    q_variance = 1 - diffusion.factors.gammas[timesteps]
    expected_q_sample = q_mean + torch.sqrt(q_variance) * noise

    assert torch.allclose(q_sample_result, expected_q_sample), "q_sample computation is incorrect."


def test_q_sample_shape(diffusion: GaussianDiffusion):
    """Test the shape of the q_sample output."""
    x_start = torch.rand(2, 3, 64, 64)
    timesteps = torch.tensor([0, 999])
    noise = torch.randn_like(x_start)
    q_sample_result = diffusion.q_sample(x_start, timesteps, noise)

    assert q_sample_result.shape == x_start.shape, "q_sample output shape is incorrect."
