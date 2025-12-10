import tempfile
from pathlib import Path

import pytest
import torch

from diffusion.data.cifar_datamodule import CIFAR10DataModule


BATCH_SIZE = 32
DATA_PATH = Path(tempfile.gettempdir()) / "data"


@pytest.fixture(scope="session")
def datamodule() -> CIFAR10DataModule:
    """A pytest fixture for CIFAR10DataModule.

    Args:
        data_path (Path): Path to the data directory.

    Returns:
        LightningDataModule: An instance of CIFAR10DataModule.
    """
    module = CIFAR10DataModule(path=DATA_PATH, batch_size=BATCH_SIZE)
    module.setup()
    return module


def test_cifar10_data_download(datamodule: CIFAR10DataModule) -> None:
    """Tests if CIFAR10 data is downloaded and the correct directory structure is created."""
    assert datamodule is not None


def test_cifar10_data_splits(datamodule: CIFAR10DataModule) -> None:
    """Tests if data splits (train, val, test) are created after calling setup()."""

    assert datamodule.data_train is not None
    assert datamodule.data_val is not None
    assert datamodule.data_val is not None


def test_cifar10_dataloaders_creation(datamodule: CIFAR10DataModule) -> None:
    """Tests if dataloaders for train, validation, and test sets are correctly created."""

    assert datamodule.train_dataloader() is not None
    assert datamodule.val_dataloader() is not None


def test_cifar10_dataloader_batch(datamodule: CIFAR10DataModule) -> None:
    """Tests if the batch size, data type, and shapes of a batch are correct."""

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert len(x) == BATCH_SIZE
    assert len(y) == BATCH_SIZE
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_cifar10_filtered() -> None:
    """Tests if the batch size and number of batches are correct for filtered datasets."""
    datamodule = CIFAR10DataModule(
        path=DATA_PATH, batch_size=BATCH_SIZE, labels=[0, 1], train_samples_per_label=1280, val_samples_per_label=320
    )
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    train_batch = next(iter(train_dataloader))
    x, y = train_batch
    assert len(x) == BATCH_SIZE
    assert len(y) == BATCH_SIZE

    assert len(train_dataloader) == (1280 * 2) // BATCH_SIZE

    val_dataloader, sample_dataloader = datamodule.val_dataloader()
    val_batch = next(iter(val_dataloader))
    x_val, y_val = val_batch
    assert len(x_val) == BATCH_SIZE
    assert len(y_val) == BATCH_SIZE

    sample_batch = next(iter(sample_dataloader))
    x_sample, _ = sample_batch
    assert len(x_sample) == datamodule.predict_samples
