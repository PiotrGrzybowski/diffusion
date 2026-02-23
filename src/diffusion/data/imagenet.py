import torch
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageNetDataModule(LightningDataModule):
    """LightningDataModule for ImageNet with train/val/test splits."""

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        augment_train: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.setup_transforms()

    def setup_transforms(self):
        base_transforms = [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

        train_transforms = []
        if self.augment_train:
            train_transforms.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                ]
            )
        train_transforms.extend(base_transforms)

        self.train_transform = transforms.Compose(train_transforms)
        self.val_transform = transforms.Compose(base_transforms)

    def setup(self, stage: str | None = None):
        """Setup datasets for each stage."""
        dataset_dict = load_dataset(self.data_path)

        if stage == "fit" or stage is None:
            self.train_dataset = dataset_dict["train"]
            self.val_dataset = dataset_dict["validation"]

        if stage == "test" or stage is None:
            self.test_dataset = dataset_dict["test"]

    def _build_dataloader(self, dataset, transform, shuffle=False):
        """Builder for creating dataloaders with consistent settings."""
        if dataset is None:
            return None

        def collate_fn(batch):
            images = torch.stack([transform(item["image"]) for item in batch])
            labels = torch.tensor([item["label"] for item in batch])
            return images, labels

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, self.train_transform, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset, self.val_transform, shuffle=False)

    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset, self.val_transform, shuffle=False)
