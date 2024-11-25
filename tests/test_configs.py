from pathlib import Path

from diffusion.variances import VarianceType
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig


def test_variance_instantiate(configs_path: Path) -> None:
    configs = ["fixed_small", "fixed_large"]
    variance_types = [VarianceType.FIXED_SMALL, VarianceType.FIXED_LARGE]

    for config_name, variance_type in zip(configs, variance_types):
        with initialize_config_dir(config_dir=str(configs_path), version_base="1.3"):
            config = compose(config_name=f"variance/{config_name}")
        variance = instantiate(config)["variance"]
        assert variance == variance_type


def test_scheduler_instance(configs_path: Path) -> None:
    with initialize_config_dir(config_dir=str(configs_path), version_base="1.3"):
        config = compose(config_name="scheduler/linear")
    config.scheduler.timesteps = 1000
    scheduler = instantiate(config)["scheduler"]
    assert scheduler


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    instantiate(cfg_train.data)
    instantiate(cfg_train.model)
    instantiate(cfg_train.trainer)
