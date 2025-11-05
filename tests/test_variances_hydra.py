import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from rootutils import find_root

from diffusion.variances import DirectVariance, FixedLargeVariance, FixedSmallVariance, TrainableRangeVariance


@pytest.fixture
def config_dir() -> str:
    return str(find_root(__file__, indicator="pyproject.toml") / "configs" / "diffusion" / "variance_strategy")


def test_direct_variance_variance_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="direct.yaml")

    variance_strategy = instantiate(cfg)
    assert isinstance(variance_strategy, DirectVariance)


def test_fixed_small_variance_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="fixed_small.yaml")

    variance_strategy = instantiate(cfg)
    assert isinstance(variance_strategy, FixedSmallVariance)


def test_fixed_large_variance_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="fixed_large.yaml")

    variance_strategy = instantiate(cfg)
    assert isinstance(variance_strategy, FixedLargeVariance)


def test_trainable_range_variance_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="trainable_range.yaml")

    variance_strategy = instantiate(cfg)
    assert isinstance(variance_strategy, TrainableRangeVariance)


def test_direct_variance_is_trainable(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="direct.yaml")

    variance_strategy = instantiate(cfg)
    assert variance_strategy.trainable is True


def test_fixed_small_variance_is_not_trainable(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="fixed_small.yaml")

    variance_strategy = instantiate(cfg)
    assert variance_strategy.trainable is False


def test_fixed_large_variance_is_not_trainable(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="fixed_large.yaml")

    variance_strategy = instantiate(cfg)
    assert variance_strategy.trainable is False


def test_trainable_range_variance_is_trainable(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="trainable_range.yaml")

    variance_strategy = instantiate(cfg)
    assert variance_strategy.trainable is True
