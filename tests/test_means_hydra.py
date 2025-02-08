import pytest
from diffusion.means import DirectMean, EpsilonMean, XStartMean
from hydra import compose, initialize
from hydra.utils import instantiate


@pytest.fixture
def config_path():
    return "../configs/diffusion/mean"


def test_direct_mean_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="direct.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, DirectMean)


def test_xstart_mean_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="xstart.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, XStartMean)


def test_epsilon_mean_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="epsilon.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, EpsilonMean)
