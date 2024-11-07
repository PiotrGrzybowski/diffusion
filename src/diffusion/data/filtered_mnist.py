from collections import defaultdict

import torch
from torchvision.datasets import MNIST, VisionDataset


class FilteredMNIST(VisionDataset):
    def __init__(self, dataset: MNIST, labels: list[int] | None = None, samples_per_label: int | None = None) -> None:
        """A filtered MNIST dataset that includes only specific labels and a limited number of samples per class.

        Args:
            dataset: (MNIST) An instance of the MNIST dataset to be filtered.
            labels: (list[int]) List of labels to include in the subset.
            samples_per_label: (int) Maximum number of samples per class.

        Attributes:
            dataset: (MNIST) An instance of the MNIST dataset to be filtered.
            labels: (list[int]) List of labels to include in the subset.
            samples_per_class: (int) Maximum number of samples per class.
            indices: (list[int]) List of indices of the filtered dataset.
        """

        self.dataset = dataset
        self.labels = labels
        self.samples_per_class = samples_per_label
        self.indices = self._filter_indices()

    def _filter_indices(self) -> list[int]:
        label_counts = defaultdict(int)
        indices = []

        for idx in range(len(self.dataset)):
            label = int(self.dataset.targets[idx])
            if self._is_label_valid(label) and not self._is_label_finished(label, label_counts):
                indices.append(idx)
                label_counts[label] += 1

                if self.samples_per_class and all(count >= self.samples_per_class for count in label_counts.values()):
                    break

        return indices

    def _is_label_valid(self, label: int) -> bool:
        return self.labels is None or label in self.labels

    def _is_label_finished(self, label: int, counts: dict[int, int]) -> bool:
        return self.samples_per_class is not None and counts[label] >= self.samples_per_class

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_index = self.indices[index]
        return self.dataset[real_index]

    def __len__(self) -> int:
        return len(self.indices)
