from pathlib import Path

import pytest
import torch
from diffusion.data.mnist_datamodule import MNISTDataModule
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


@pytest.mark.parametrize("dataset_name", ["mnist", "fashion", "kmnist"])
def test_mnist_data_download(data_path: Path, dataset_name: str) -> None:
    """Tests if MNIST data is downloaded and the correct directory structure is created."""
    datamodule = MNISTDataModule(path=data_path, dataset_name=dataset_name, batch_size=32)
    datamodule.prepare_data()

    assert Path(data_path, datamodule.dataset_class.__name__).exists()
    assert Path(data_path, datamodule.dataset_class.__name__, "raw").exists()


@pytest.mark.parametrize("batch_size", [32, 128, 256])
def test_mnist_data_splits(data_path: Path, batch_size: int) -> None:
    """Tests if data splits (train, val, test) are created after calling setup()."""
    datamodule = MNISTDataModule(path=data_path, batch_size=batch_size)
    datamodule.setup()

    assert datamodule.data_train is not None
    assert datamodule.data_val is not None
    assert datamodule.data_test is not None


@pytest.mark.parametrize("batch_size", [32, 128, 256])
def test_mnist_dataloaders_creation(data_path: Path, batch_size: int) -> None:
    """Tests if dataloaders for train, validation, and test sets are correctly created."""
    datamodule = MNISTDataModule(path=data_path, batch_size=batch_size)
    datamodule.setup()

    assert datamodule.train_dataloader() is not None
    assert datamodule.val_dataloader() is not None
    assert datamodule.test_dataloader() is not None


@pytest.mark.parametrize("batch_size", [32, 128, 256])
def test_mnist_dataloader_batch(data_path: Path, batch_size: int) -> None:
    """Tests if the batch size, data type, and shapes of a batch are correct."""
    datamodule = MNISTDataModule(path=data_path, batch_size=batch_size)
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128, 256])
def test_mnist_filtered(data_path: Path, batch_size: int) -> None:
    datamodule = MNISTDataModule(path=data_path, batch_size=batch_size, val_split=0.2, labels=[0, 1], samples_per_label=1280)
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size

    assert len(datamodule.train_dataloader()) == int(1280 * 0.8) * 2 // batch_size
    assert len(datamodule.val_dataloader()) == int(1280 * 0.2) * 2 // batch_size


def test_hydra_default(configs_path: Path) -> None:
    with initialize_config_dir(config_dir=str(configs_path), version_base="1.3"):
        cfg = compose(config_name="data/mnist")

        module = instantiate(cfg)
        assert module is not None


@pytest.mark.parametrize("dataset_name", ["mnist", "fashion"])
def test_hydra_various_datasets_default(configs_path: Path, dataset_name: str) -> None:
    with initialize_config_dir(config_dir=str(configs_path), version_base="1.3"):
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
