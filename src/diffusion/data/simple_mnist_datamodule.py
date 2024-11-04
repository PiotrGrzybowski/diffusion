import ssl
from pathlib import Path

import torch
from torch.utils.data import random_split

from diffusion.data.mnist_datamodule import GaussianDataset, MNISTDataModule
from diffusion.data.utils import create_filtered_dataset


ssl._create_default_https_context = ssl._create_unverified_context


class SimpleMNISTDataModule(MNISTDataModule):
    def __init__(
        self,
        path: Path = Path("/tmp/data"),
        val_split: float = 0.9,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        predict_size: int = 1,
        labels: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        train_samples: int = 100,
        val_samples: int = 10,
        test_samples: int = 10,
    ) -> None:
        super().__init__(path, val_split, batch_size, num_workers, pin_memory, predict_size)
        self.labels = labels
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

    def setup(self, stage: str | None = None) -> None:
        self._check_batch_size_compatibility()
        if not self._is_setup():
            generator = torch.Generator().manual_seed(42)
            train_set, test_set = self._load()
            train_length, val_length = self._train_val_lengths(train_set, self.val_split)
            train_set, val_set = random_split(train_set, lengths=[train_length, val_length], generator=generator)

            self.data_train = create_filtered_dataset(train_set, self.labels, self.train_samples)
            self.data_val = create_filtered_dataset(val_set, self.labels, self.val_samples)
            self.data_test = create_filtered_dataset(test_set, self.labels, self.test_samples)
            self.data_predict = GaussianDataset((self.predict_size, 1, 28, 28))


if __name__ == "__main__":
    mnist = SimpleMNISTDataModule(labels=[1], train_samples=10, val_samples=2, test_samples=2)
    mnist.prepare_data()
    mnist.setup()

    train_loader = mnist.train_dataloader()
    val_loader = mnist.val_dataloader()
    test_loader = mnist.test_dataloader()

    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print(train_loader.dataset[0][0].shape)
    print(val_loader.dataset[0][0].shape)
    print(test_loader.dataset[0][0].shape)

    print(next(iter(train_loader))[0].shape)
