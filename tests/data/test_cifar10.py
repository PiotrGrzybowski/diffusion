from pathlib import Path

import pytest
import torch
from rootutils import find_root

from diffusion.data.cifar_datamodule import CIFAR10DataModule


@pytest.fixture(scope="package")
def diffusion_configs_path() -> Path:
    """A pytest fixture for path to the configs directory.

    Returns:
        Path: A Path object pointing to the configs directory.
    """
    return find_root("pyproject.toml") / "configs" / "diffusion"


def test_cifar10_data_download(data_path: Path) -> None:
    """Tests if CIFAR10 data is downloaded and the correct directory structure is created."""
    datamodule = CIFAR10DataModule(path=data_path, batch_size=32)
    datamodule.prepare_data()

    assert Path(data_path, "cifar-10-python.tar.gz").exists()


@pytest.mark.parametrize("batch_size", [32, 128, 256])
def test_cifar10_data_splits(data_path: Path, batch_size: int) -> None:
    """Tests if data splits (train, val, test) are created after calling setup()."""
    datamodule = CIFAR10DataModule(path=data_path, batch_size=batch_size)
    datamodule.setup()

    assert datamodule.data_train is not None
    assert datamodule.data_val is not None
    assert datamodule.data_val is not None


@pytest.mark.parametrize("batch_size", [32, 128, 256])
def test_cifar10_dataloaders_creation(data_path: Path, batch_size: int) -> None:
    """Tests if dataloaders for train, validation, and test sets are correctly created."""
    datamodule = CIFAR10DataModule(path=data_path, batch_size=batch_size)
    datamodule.setup()

    assert datamodule.train_dataloader() is not None
    assert datamodule.val_dataloader() is not None


@pytest.mark.parametrize("batch_size", [32, 128, 256])
def test_cifar10_dataloader_batch(data_path: Path, batch_size: int) -> None:
    """Tests if the batch size, data type, and shapes of a batch are correct."""
    datamodule = CIFAR10DataModule(path=data_path, batch_size=batch_size)
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_cifar10_filtered(data_path: Path, batch_size: int) -> None:
    datamodule = CIFAR10DataModule(
        path=data_path, batch_size=batch_size, labels=[0, 1], train_samples_per_label=1280, val_samples_per_label=320
    )
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    train_batch = next(iter(train_dataloader))
    x, y = train_batch
    assert len(x) == batch_size
    assert len(y) == batch_size

    assert len(train_dataloader) == (1280 * 2) // batch_size

    val_dataloader, sample_dataloader = datamodule.val_dataloader()
    val_batch = next(iter(val_dataloader))
    x_val, y_val = val_batch
    assert len(x_val) == batch_size
    assert len(y_val) == batch_size

    sample_batch = next(iter(sample_dataloader))
    x_sample, _ = sample_batch
    assert len(x_sample) == datamodule.predict_samples
