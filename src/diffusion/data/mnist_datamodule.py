import ssl
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


ssl._create_default_https_context = ssl._create_unverified_context


class GaussianDataset(Dataset):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.data = torch.randn(shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        path: Path = Path("/tmp/data"),
        val_split: float = 0.9,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        predict_size: int = 1,
    ) -> None:
        super().__init__()

        self.path = path
        self.batch_size = batch_size
        self.predict_size = predict_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # [0,1] to [-1,1]

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.data_predict: Dataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        MNIST(str(self.path), train=True, download=True)
        MNIST(str(self.path), train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        self._check_batch_size_compatibility()
        if not self._is_setup():
            generator = torch.Generator().manual_seed(42)
            train_set, test_set = self._load()
            train_length, val_length = self._train_val_lengths(train_set, self.val_split)
            train_set, val_set = random_split(train_set, lengths=[train_length, val_length], generator=generator)

            self.data_train = train_set
            self.data_val = val_set
            self.data_test = test_set
            self.data_predict = GaussianDataset((self.predict_size, 1, 28, 28))

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self.data_train:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )
        else:
            raise RuntimeError("The training dataset is not loaded.")

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self.data_val:
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        else:
            raise RuntimeError("The validation dataset is not loaded.")

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        self.data_predict = GaussianDataset((self.predict_size, 1, 28, 28))
        if self.data_test:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        else:
            raise RuntimeError("The test dataset is not loaded.")

    def predict_dataloader(self) -> DataLoader[torch.Tensor]:
        if self.data_predict:
            return DataLoader(
                dataset=self.data_predict,
                batch_size=self.predict_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        else:
            raise RuntimeError("The prediction dataset is not loaded.")

    def _check_batch_size_compatibility(self) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}).")

    def _is_setup(self) -> bool:
        return self.data_train is not None and self.data_val is not None and self.data_test is not None

    def _train_val_lengths(self, dataset: MNIST, split: float) -> tuple[int, int]:
        length = len(dataset)
        train_length = int(length * split)
        val_length = length - train_length
        return train_length, val_length

    def _load(self) -> tuple[MNIST, MNIST]:
        train_set = MNIST(str(self.path), train=True, transform=self.transforms)
        test_set = MNIST(str(self.path), train=False, transform=self.transforms)
        return train_set, test_set


if __name__ == "__main__":
    mnist = MNISTDataModule()
    mnist.prepare_data()
    mnist.setup()

    train_loader = mnist.train_dataloader()
    val_loader = mnist.val_dataloader()
    test_loader = mnist.test_dataloader()

    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print(train_loader.dataset[0][0].shape)
    print(val_loader.dataset[0][0].shape)
    print(test_loader.dataset[0][0].shape)

    print(next(iter(train_loader)).shape)
