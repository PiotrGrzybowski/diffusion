import pytest
from diffusion.schedulers import LinearScheduler
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import open_dict


@pytest.fixture
def config_path():
    return "../configs/diffusion/scheduler"


def test_linear_scheduler_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="linear.yaml")

    with open_dict(cfg):
        cfg.timesteps = 1000

    scheduler = instantiate(cfg)
    assert isinstance(scheduler, LinearScheduler)
