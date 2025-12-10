import math

import pytest
import torch

from diffusion.diffusion_factors import Factors
from diffusion.schedulers import LinearScheduler


@pytest.fixture
def factors():
    betas = LinearScheduler(1000, 0.0001, 0.02).schedule()
    return Factors(betas)


def test_factors_initialization(factors: Factors):
    """Test the correctness of the Factors class initialization."""
    assert hasattr(factors, "betas"), "Factors class is missing 'betas' attribute."
    assert hasattr(factors, "alphas"), "Factors class is missing 'alphas' attribute."
    assert hasattr(factors, "gammas"), "Factors class is missing 'gammas' attribute."


def test_factors_shapes(factors: Factors):
    """Test the correctness of the shapes of the tensors."""
    expected_shape = (1000, 1, 1, 1)
    assert factors.betas.shape == torch.Size(expected_shape), f"Expected betas shape {expected_shape}, got {factors.betas.shape}."
    assert factors.alphas.shape == torch.Size(expected_shape), f"Expected alphas shape {expected_shape}, got {factors.alphas.shape}."
    assert factors.gammas.shape == torch.Size(expected_shape), f"Expected gammas shape {expected_shape}, got {factors.gammas.shape}."


def test_factors(factors: Factors):
    assert factors.alphas.shape == (1000, 1, 1, 1)
    assert factors.betas.shape == (1000, 1, 1, 1)
    assert factors.gammas.shape == (1000, 1, 1, 1)
    assert factors.gammas_prev.shape == (1000, 1, 1, 1)
    assert math.isclose(factors.gammas_prev[0].item(), 1.0)

    assert torch.allclose(factors.alphas, 1 - factors.betas)
    assert torch.allclose(factors.gammas_prev[1:], factors.gammas[:-1])
    assert torch.allclose(factors.gammas, torch.cumprod(factors.alphas, dim=0))
