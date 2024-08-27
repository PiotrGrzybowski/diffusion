from pathlib import Path

import pytest
import torch

from diffusion.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128, 256])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = Path("/tmp/data")

    datamodule = MNISTDataModule(path=data_dir, batch_size=batch_size)
    datamodule.prepare_data()

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    datamodule.setup()
    assert datamodule.data_train and datamodule.data_val and datamodule.data_test
    assert datamodule.train_dataloader() and datamodule.val_dataloader() and datamodule.test_dataloader()

    num_datapoints = len(datamodule.data_train) + len(datamodule.data_val) + len(datamodule.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
