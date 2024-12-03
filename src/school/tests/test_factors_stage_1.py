import pytest
import torch
from school.factors import Factors


@pytest.fixture
def factors():
    return Factors(torch.linspace(0.0001, 0.02, 1000))


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


def test_factors_calculations():
    """Test the correctness of alphas and gammas calculations."""
    betas = torch.linspace(0.0001, 0.02, 1000)
    factors = Factors(betas)

    betas = betas.view(-1, 1, 1, 1)
    alphas = 1 - betas
    gammas = torch.cumprod(alphas, dim=0)

    assert torch.allclose(factors.alphas, alphas), "Calculated alphas are incorrect."
    assert torch.allclose(factors.gammas, gammas), "Calculated gammas are incorrect."
