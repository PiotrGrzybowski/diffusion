import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from diffusion.scripts.train import train


@pytest.mark.slow
@pytest.mark.smoke
def test_train_smoke_epsilon_simple(configs_dir: Path):
    """Smoke test: Train 1 batch with epsilon mean + simple MSE loss.

    This test validates that the complete training pipeline works with
    a representative configuration. It uses:
    - Data: MNIST (downloads if needed, small dataset)
    - Model: small_unet (fast initialization)
    - Mean strategy: epsilon (most common)
    - Loss: mse_epsilon_simple (simple, fast)
    - Trainer: fast_dev_run=True (1 train + 1 val batch)

    Expected runtime: ~5-7 seconds (after data is downloaded)
    """
    logging.disable(logging.CRITICAL)

    with TemporaryDirectory() as output_dir:
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_dir)):
            overrides = [
                "diffusion/model=small_unet",
                "diffusion/mean_strategy=epsilon",
                "diffusion/loss=mse_epsilon_simple",
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
