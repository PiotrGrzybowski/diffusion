import pytest
import torch.nn as nn
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict

from diffusion.diffusion_factors import Factors
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.losses import DiffusionLoss
from diffusion.means import MeanStrategy
from diffusion.variances import VarianceStrategy


@pytest.fixture
def config_path():
    return "../configs/diffusion/"


def test_gaussian_diffusion_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="gaussian_diffusion.yaml")

    with open_dict(cfg):
        cfg.timesteps = 1000
        cfg.in_channels = 3

    diffusion = instantiate(cfg)

    assert isinstance(diffusion, GaussianDiffusion)
    assert isinstance(diffusion.posterior_mean, MeanStrategy)
    assert isinstance(diffusion.posterior_variance, VarianceStrategy)
    assert isinstance(diffusion.model_mean, MeanStrategy)
    assert isinstance(diffusion.model_variance, VarianceStrategy)
    assert isinstance(diffusion.loss, DiffusionLoss)
    assert isinstance(diffusion.model, nn.Module)
    assert isinstance(diffusion.factors, Factors)

    assert diffusion.in_channels == 3
