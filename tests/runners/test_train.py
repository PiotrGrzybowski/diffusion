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
models = ["small_unet"]


# @pytest.mark.parametrize("loss, model", product(losses, models))
def test_train_fast_dev_run(cfg_train: DictConfig):
    # with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
    # diffusion = compose(config_name=f"model/{model}")
    # loss_config = compose(config_name=f"loss/{loss}")
    # with open_dict(cfg_train):
    #     cfg_train.model = diffusion.model
    #     cfg_train.loss = loss_config.loss

    HydraConfig().set_config(cfg_train)
    with TemporaryDirectory() as output_dir:
        with open_dict(cfg_train):
            cfg_train.trainer.fast_dev_run = True
            cfg_train.trainer.accelerator = "cpu"
            cfg_train.logger = None
            cfg_train.callbacks = None
            cfg_train.train = True
            cfg_train.validate = False
            cfg_train.paths.output_dir = output_dir
            cfg_train.run_name = "test"
            cfg_train.task_name = "test"
            cfg_train.trainer.check_val_every_n_epoch = 10
            cfg_train.sample_timesteps = 2
            train(cfg_train)
