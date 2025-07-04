import ssl
from pathlib import Path
from tempfile import gettempdir

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms

from diffusion.data.dataset_map import DatasetMap
from diffusion.data.filtered_mnist import FilteredDataset


ssl._create_default_https_context = ssl._create_unverified_context


class LazyGaussianDataset(Dataset):
    def __init__(self, num_samples: int, shape: tuple):
        """
        A dataset that lazily generates Gaussian noise samples.

        Args:
            num_samples (int): Number of samples to generate.
            shape (tuple): Shape of each sample (e.g., image shape).
            mean (float): Mean of the Gaussian distribution.
            std (float): Standard deviation of the Gaussian distribution.
        """
        self.num_samples = num_samples
        self.shape = shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(self.shape)


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "mnist",
        path: Path = Path(gettempdir()),
        val_split: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        labels: list[int] | None = None,
        train_samples_per_label: int | None = None,
        val_samples_per_label: int | None = None,
        predict_samples: int = 4,
    ) -> None:
        super().__init__()

        self.path = path
        self.dataset_class = DatasetMap.get_dataset_class(dataset_name.lower())
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.labels = labels
        self.train_samples_per_label = train_samples_per_label
        self.val_samples_per_label = val_samples_per_label

        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        # self.data_val: Dataset | None = None

        self.batch_size_per_device = batch_size
        self.generator = torch.Generator().manual_seed(42)

        self.predict_samples = predict_samples

        self.prepare_data()
        self.setup()

    def prepare_data(self) -> None:
        self.dataset_class(self.path, train=True, download=True)
        self.dataset_class(self.path, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        self._check_batch_size_compatibility()
        if not self._is_setup():
            train_dataset = FilteredDataset(
                self.dataset_class(self.path, train=True, transform=self.transforms), self.labels, self.train_samples_per_label
            )
            test_dataset = FilteredDataset(
                self.dataset_class(self.path, train=False, transform=self.transforms), self.labels, self.val_samples_per_label
            )

            self.data_train = train_dataset
            self.data_val = test_dataset

    def _random_split(self, dataset: VisionDataset) -> list:
        val_length = int(len(dataset) * self.val_split)
        train_length = len(dataset) - val_length

        return random_split(dataset, lengths=[train_length, val_length], generator=self.generator)

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

    # def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    #     if self.data_val:
    #         return DataLoader(
    #             dataset=self.data_val,
    #             batch_size=self.batch_size_per_device,
    #             num_workers=self.num_workers,
    #             pin_memory=self.pin_memory,
    #             shuffle=False,
    #         )
    #     else:
    #         raise RuntimeError("The test dataset is not loaded.")

    def predict_dataloader(self):
        return DataLoader(
            dataset=LazyGaussianDataset(num_samples=self.predict_samples, shape=(1, 28, 28)),
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def _check_batch_size_compatibility(self) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}).")

    def _is_setup(self) -> bool:
        return self.data_train is not None and self.data_val is not None and self.data_val is not None
