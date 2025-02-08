from pathlib import Path

import pytest
from diffusion.data.mnist_datamodule import MNISTDataModule
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_hydra_default(configs_dir: Path) -> None:
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        cfg = compose(config_name="data/mnist")

        module = instantiate(cfg)
        assert module is not None


@pytest.mark.parametrize("dataset_name", ["mnist", "fashion"])
def test_hydra_various_datasets_default(configs_dir: Path, dataset_name: str) -> None:
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.3"):
        cfg = compose(
            config_name="data/mnist",
            overrides=[f"data.dataset_name={dataset_name}", "data.batch_size=32"],
        )

        datamodule: MNISTDataModule = instantiate(cfg.data)
        assert datamodule is not None

        datamodule.prepare_data()
        datamodule.setup()

        batch = next(iter(datamodule.train_dataloader()))
        x, y = batch
        assert len(x) == 32
        assert len(y) == 32
