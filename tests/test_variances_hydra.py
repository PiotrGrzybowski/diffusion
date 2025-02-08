import pytest
from diffusion.variances import DirectVariance, FixedLargeVariance, FixedSmallVariance, TrainableRangeVariance
from hydra import compose, initialize
from hydra.utils import instantiate


@pytest.fixture
def config_path():
    return "../configs/variance"


def test_direct_variance_variance_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="direct.yaml")

    variance_strategy = instantiate(cfg)
    assert isinstance(variance_strategy, DirectVariance)


def test_fixed_small_variance_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="fixed_small.yaml")

    variance_strategy = instantiate(cfg)
    assert isinstance(variance_strategy, FixedSmallVariance)


def test_fixed_large_variance_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="fixed_large.yaml")

    variance_strategy = instantiate(cfg)
    assert isinstance(variance_strategy, FixedLargeVariance)


def test_trainable_range_variance_instantiate(config_path: str):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="trainable_range.yaml")

    variance_strategy = instantiate(cfg)
    assert isinstance(variance_strategy, TrainableRangeVariance)
