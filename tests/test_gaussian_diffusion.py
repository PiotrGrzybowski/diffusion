from unittest.mock import Mock

import pytest
import torch

from diffusion.diffusion_factors import Factors
from diffusion.gaussian_diffusion import GaussianDiffusion


torch.set_printoptions(precision=10, sci_mode=False)


@pytest.fixture
def diffusion() -> GaussianDiffusion:
    factors = Factors(betas=torch.tensor([0.2, 0.15, 0.1]))
    mock = Mock()
    mock.factors = factors

    model = Mock()
    model.in_channels = 1

    mock.model = model

    return mock


def test_q_mean(diffusion: GaussianDiffusion):
    diffusion.q_mean = GaussianDiffusion.q_mean.__get__(diffusion)

    x_start = torch.tensor([[[[1.0]]]])
    timesteps = torch.tensor([0, 1, 2])

    target_mean = torch.tensor([[[[0.89442719]]], [[[0.82462112]]], [[[0.78230428]]]])
    mean = diffusion.q_mean(timesteps, x_start)

    assert torch.allclose(mean, target_mean)


def test_q_variance(diffusion: GaussianDiffusion):
    diffusion.q_variance = GaussianDiffusion.q_variance.__get__(diffusion)

    timesteps = torch.tensor([0, 1, 2])

    target_variance = torch.tensor([[[[0.2]]], [[[0.32]]], [[[0.388]]]])
    variance = diffusion.q_variance(timesteps)

    assert torch.allclose(variance, target_variance)


def test_q_sample(diffusion: GaussianDiffusion):
    diffusion.q_mean = GaussianDiffusion.q_mean.__get__(diffusion)
    diffusion.q_variance = GaussianDiffusion.q_variance.__get__(diffusion)
    diffusion.q_sample = GaussianDiffusion.q_sample.__get__(diffusion)

    timesteps = torch.tensor([0, 1, 2])
    x_start = torch.tensor([[[[1.0]]]])
    epsilon = torch.tensor([[[[-0.25]]]])

    target_q_sample = torch.tensor([[[[0.7826237679]]], [[[0.6831997633]]], [[[0.6265801787]]]])
    q_sample = diffusion.q_sample(timesteps, x_start, epsilon)

    assert torch.allclose(q_sample, target_q_sample)


def test_posterior_mean(diffusion: GaussianDiffusion):
    diffusion.posterior_mean = GaussianDiffusion.posterior_mean.__get__(diffusion)

    x_start = torch.tensor([[[[1.0]]]])
    x_t = torch.tensor([[[[0.8]]]])
    timesteps = torch.tensor([0, 1, 2])

    posterior_mean = diffusion.posterior_mean(x_start, x_t, timesteps)
    target_posterior_mean = torch.tensor([[[[1.0]]], [[[0.8802399635]]], [[[0.8384665251]]]])

    assert torch.allclose(posterior_mean, target_posterior_mean)


def test_posterior_variance(diffusion: GaussianDiffusion):
    diffusion.posterior_variance = GaussianDiffusion.posterior_variance.__get__(diffusion)

    timesteps = torch.tensor([0, 1, 2])

    target_posterior_variance = torch.tensor([[[[0.0]]], [[[0.0937500000]]], [[[0.0824742317]]]])
    posterior_variance = diffusion.posterior_variance(timesteps)

    assert torch.allclose(posterior_variance, target_posterior_variance)


def test_posterior_log_variance(diffusion: GaussianDiffusion):
    diffusion.posterior_variance = GaussianDiffusion.posterior_variance.__get__(diffusion)
    diffusion.posterior_log_variance = GaussianDiffusion.posterior_log_variance.__get__(diffusion)

    timesteps = torch.tensor([0, 1, 2])

    target_posterior_log_variance = torch.tensor([[[[-2.3671236038]]], [[[-2.3671236038]]], [[[-2.4952692986]]]])
    posterior_log_variance = diffusion.posterior_log_variance(timesteps)

    assert torch.allclose(posterior_log_variance, target_posterior_log_variance)


def test_prediction_split_fixed_variance(diffusion: GaussianDiffusion):
    diffusion.mean_prediction = GaussianDiffusion.mean_prediction.__get__(diffusion)
    diffusion.variance_prediction = GaussianDiffusion.variance_prediction.__get__(diffusion)

    prediction = torch.rand(3, 1, 28, 28)

    mean_prediction = diffusion.mean_prediction(prediction)
    variance_prediction = diffusion.variance_prediction(prediction)

    assert mean_prediction.shape == (3, 1, 28, 28)
    assert variance_prediction.shape == (3, 0, 28, 28)


def test_prediction_split_trainable_variance(diffusion: GaussianDiffusion):
    diffusion.mean_prediction = GaussianDiffusion.mean_prediction.__get__(diffusion)
    diffusion.variance_prediction = GaussianDiffusion.variance_prediction.__get__(diffusion)

    prediction = torch.rand(3, 2, 28, 28)

    mean_prediction = diffusion.mean_prediction(prediction)
    variance_prediction = diffusion.variance_prediction(prediction)

    assert mean_prediction.shape == (3, 1, 28, 28)
    assert variance_prediction.shape == (3, 1, 28, 28)
