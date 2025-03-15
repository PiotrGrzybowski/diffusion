import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict

from diffusion.diffusion_terms import DiffusionTerms
from diffusion.gaussian_diffusion import GaussianDiffusion


@pytest.fixture
def diffusion():
    config_dir = "../configs/diffusion/"

    with initialize(version_base="1.3", config_path=config_dir):
        cfg = compose(config_name="gaussian_diffusion.yaml")

    with open_dict(cfg):
        cfg.timesteps = 1000
        cfg.sample_timesteps = 1000
        cfg.in_channels = 3
        cfg.model.out_channels = 3

    diffusion = instantiate(cfg)

    return diffusion


def test_diffusion_q_sample(diffusion):
    timesteps = torch.tensor([8, 900], dtype=torch.long)
    x_start = (torch.rand(2, 3, 32, 32) - 0.5) * 2
    noise = torch.randn_like(x_start)

    q_sample = diffusion.q_sample(timesteps, x_start, noise)

    assert q_sample.shape == x_start.shape


def test_model_step(diffusion: GaussianDiffusion):
    x_start = (torch.rand(2, 3, 32, 32) - 0.5) * 2
    timesteps = torch.tensor([8, 900], dtype=torch.long)

    output = diffusion.model_step(x_start, timesteps)

    assert isinstance(output, DiffusionTerms)


def test_posterior_step(diffusion: GaussianDiffusion):
    x_start = (torch.rand(2, 3, 32, 32) - 0.5) * 2
    timesteps = torch.tensor([8, 900], dtype=torch.long)

    output = diffusion.posterior_step(x_start, timesteps)

    assert isinstance(output, DiffusionTerms)


def test_training_step(diffusion: GaussianDiffusion):
    x_start = (torch.rand(2, 3, 32, 32) - 0.5) * 2
    batch = (x_start, torch.empty())

    output = diffusion.training_step(batch)

    assert isinstance(output, torch.Tensor)
