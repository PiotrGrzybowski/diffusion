import pytest
from diffusion.losses import (
    VLB,
    Hybrid,
    MseMeanDirect,
    MseMeanDirectSimple,
    MseMeanEpsilon,
    MseMeanEpsilonSimple,
    MseMeanXStart,
    MseMeanXStartSimple,
)
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from rootutils import find_root


@pytest.fixture
def config_dir() -> str:
    return str(find_root(__file__, indicator="pyproject.toml") / "configs" / "diffusion" / "loss")


def test_mse_mean_direct_loss_instantiate(config_dir: str):
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="mse_mean_direct.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, MseMeanDirect)


def test_mse_mean_direct_loss_simple_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="mse_mean_direct_simple.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, MseMeanDirectSimple)


def test_mse_mean_xstart_loss_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="mse_mean_xstart.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, MseMeanXStart)


def test_mse_mean_xstart_loss_simple_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="mse_mean_xstart_simple.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, MseMeanXStartSimple)


def test_mse_mean_epsilon_loss_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="mse_mean_epsilon.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, MseMeanEpsilon)


def test_mse_mean_epsilon_loss_simple_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="mse_mean_epsilon_simple.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, MseMeanEpsilonSimple)


def test_vlb_loss_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="vlb.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, VLB)


def test_hybrid_loss_instantiate(config_dir: str):
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="hybrid.yaml")

    mean_strategy = instantiate(cfg)
    assert isinstance(mean_strategy, Hybrid)
