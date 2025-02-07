import pytest
import torch
from diffusion.diffusion_factors import Factors
from diffusion.schedulers import LinearScheduler


@pytest.fixture
def factors():
    return Factors(LinearScheduler(1000, 0.0001, 0.02))


def test_factors(factors: Factors):
    assert factors.alphas.shape == (1000, 1, 1, 1)
    assert factors.betas.shape == (1000, 1, 1, 1)
    assert factors.gammas.shape == (1000, 1, 1, 1)
    assert factors.gammas_prev.shape == (1000, 1, 1, 1)
    assert factors.gammas_prev[0].squeeze() == 1.0

    assert torch.allclose(factors.alphas, 1 - factors.betas)
    assert torch.allclose(factors.gammas_prev[1:], factors.gammas[:-1])
    assert torch.allclose(factors.gammas, torch.cumprod(factors.alphas, dim=0))
