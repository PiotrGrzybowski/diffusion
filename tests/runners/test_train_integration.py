import logging
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from diffusion.scripts.train import train


# Test parameters
MEAN_STRATEGIES = ["epsilon", "xstart", "direct"]
LOSSES = ["vlb", "mse_epsilon_simple"]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("mean, loss", product(MEAN_STRATEGIES, LOSSES))
def test_train_integration(configs_dir: Path, mean: str, loss: str):
    """Integration test: Train with various mean strategies and losses.

    This test runs the full training pipeline with:
    - Real datamodule (downloads MNIST if needed)
    - Real model (small_unet)
    - Real PyTorch Lightning trainer (fast_dev_run=True)

    It validates that the complete training loop works for all
    combinations of mean strategies and loss functions.

    Expected runtime: ~20 seconds for 6 parameter combinations (cached data)
    """
    logging.disable(logging.CRITICAL)

    with TemporaryDirectory() as output_dir:
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_dir)):
            overrides = [
                "diffusion/model=small_unet",
                f"diffusion/mean_strategy={mean}",
                f"diffusion/loss={loss}",
                "callbacks=none",
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
            cfg.sample_timesteps = 1
            cfg.trainer.limit_val_batches = 0
            cfg.trainer.num_sanity_val_steps = 0

        HydraConfig().set_config(cfg)
        train(cfg)
