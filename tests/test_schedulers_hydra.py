import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict
from rootutils import find_root

from diffusion.schedulers import CosineScheduler, LinearScheduler


@pytest.fixture
def config_dir() -> str:
    return str(find_root(__file__, indicator="pyproject.toml") / "configs" / "diffusion" / "scheduler")


def test_linear_scheduler_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="linear.yaml")

    with open_dict(cfg):
        cfg.timesteps = 1000

    scheduler = instantiate(cfg)
    assert isinstance(scheduler, LinearScheduler)


def test_cosine_scheduler_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="cosine.yaml")

    with open_dict(cfg):
        cfg.timesteps = 1000

    scheduler = instantiate(cfg)
    assert isinstance(scheduler, CosineScheduler)
