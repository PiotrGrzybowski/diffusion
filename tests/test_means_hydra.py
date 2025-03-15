import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from rootutils import find_root

from diffusion.means import DirectMean, EpsilonMean, XStartMean


@pytest.fixture
def config_dir() -> str:
    return str(find_root(__file__, indicator="pyproject.toml") / "configs" / "diffusion" / "mean_strategy")


def test_direct_mean_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="direct.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, DirectMean)


def test_xstart_mean_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="xstart.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, XStartMean)


def test_epsilon_mean_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="epsilon.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, EpsilonMean)
