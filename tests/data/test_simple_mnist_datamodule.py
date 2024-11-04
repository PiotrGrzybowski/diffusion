from pathlib import Path

import pytest
import torch
from diffusion.data.simple_mnist_datamodule import SimpleMNISTDataModule


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_simple_mnist_datamodule(batch_size: int) -> None:
    data_dir = Path("/tmp/data")

    datamodule = SimpleMNISTDataModule(path=data_dir, batch_size=batch_size, labels=[1], train_samples=16, val_samples=16, test_samples=16)
    datamodule.prepare_data()

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    datamodule.setup()
    assert datamodule.data_train and datamodule.data_val and datamodule.data_test
    assert datamodule.train_dataloader() and datamodule.val_dataloader() and datamodule.test_dataloader()

    assert len(datamodule.data_train) == 16
    assert len(datamodule.data_val) == 16
    assert len(datamodule.data_test) == 16

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
