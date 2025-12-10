import ssl
from pathlib import Path
from tempfile import gettempdir

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from diffusion.data.dataset_map import DatasetMap
from diffusion.data.filtered_mnist import FilteredDataset


ssl._create_default_https_context = ssl._create_unverified_context


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "mnist",
        path: Path = Path(gettempdir()),
        val_split: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        predict_samples: int = 4,
        labels: list[int] | None = None,
        train_samples_per_label: int | None = None,
        val_samples_per_label: int | None = None,
    ) -> None:
        super().__init__()

        self.path = path
        self.batch_size = batch_size
        self.dataset_class = DatasetMap.get_dataset_class(dataset_name.lower())
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        self.labels = labels
        self.train_samples_per_label = train_samples_per_label
        self.val_samples_per_label = val_samples_per_label
        self.predict_samples = predict_samples

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_predict: Dataset | None = None

        self.batch_size_per_device = batch_size
        self.shape = (1, 28, 28)

    @property
    def channels(self) -> int:
        return self.shape[0]

    def prepare_data(self) -> None:
        self.dataset_class(self.path, train=True, download=True)
        self.dataset_class(self.path, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        self._check_batch_size_compatibility()
        if not self._is_setup():
            train_dataset = FilteredDataset(
                self.dataset_class(self.path, train=True, transform=self.transforms), self.labels, self.train_samples_per_label
            )
            val_dataset = FilteredDataset(
                self.dataset_class(self.path, train=False, transform=self.transforms), self.labels, self.val_samples_per_label
            )
            sample_dataset = FilteredDataset(self.dataset_class(self.path, train=False, transform=self.transforms), None, 10)

            self.data_train = train_dataset
            self.data_val = val_dataset
            self.data_sample = sample_dataset

            self.shape = train_dataset[0][0].shape

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

    def val_dataloader(self) -> list[DataLoader[torch.Tensor]]:
        if self.data_val:
            validation_dataloader = DataLoader(
                dataset=self.data_val,
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
            sample_dataloader = DataLoader(
                dataset=self.data_sample,
                batch_size=self.predict_samples,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
            return [validation_dataloader, sample_dataloader]
        else:
            raise RuntimeError("The validation dataset is not loaded.")

    def _check_batch_size_compatibility(self) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size}).")

    def _is_setup(self) -> bool:
        return self.data_train is not None and self.data_val is not None and self.data_val is not None
