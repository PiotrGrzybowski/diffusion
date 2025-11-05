from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from rootutils import find_root

from diffusion.scripts.train import train


@pytest.fixture
def cfg_train(configs_dir: Path) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_dir)):
        cfg = compose(config_name="train.yaml", return_hydra_config=True)

        with open_dict(cfg):
            cfg.paths.root_dir = str(find_root(indicator="pyproject.toml"))
            cfg.trainer.fast_dev_run = True
            cfg.trainer.accelerator = "cpu"
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
            cfg.callbacks = None
            cfg.test = False

    return cfg


losses = ["mean_mse", "noise_mse", "mean_simple_mse", "noise_simple_mse", "variational_bound"]
means = ["epsilon", "xstart", "direct"]
models = ["small_unet"]


import logging

import pytest


losses = ["vlb", "mse_mean_epsilon_simple"]
models = ["unet", "small_unet"]


@pytest.mark.parametrize("mean, loss", product(means, losses))
def testo_train_fast_dev_run(configs_dir: Path, mean: str, loss: str):
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory() as output_dir:
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_dir)):
            overrides = [
                "diffusion/model=small_unet",
                f"diffusion/mean_strategy={mean}",
            ]
            cfg = compose(config_name="train.yaml", overrides=overrides, return_hydra_config=True)

        with open_dict(cfg):
            cfg.paths.output_dir = output_dir
            cfg.trainer.fast_dev_run = True
            cfg.trainer.enable_progress_bar = False
            cfg.trainer.logger = False
            cfg.trainer.enable_model_summary = False
            cfg.trainer.accelerator = "cpu"
            cfg.logger = None
            cfg.callbacks = None
            cfg.timesteps = 10
            cfg.sample_timesteps = 10
        HydraConfig().set_config(cfg)
        train(cfg)
