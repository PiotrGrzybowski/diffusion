from collections import defaultdict
from math import ceil

import torch
from torch.utils.data import Subset
from torchvision.datasets import VisionDataset


class FilteredDataset(VisionDataset):
    def __init__(self, dataset: VisionDataset, labels: list[int] | None = None, samples_per_label: int | None = None) -> None:
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


def build_balanced_subset(
    dataset: VisionDataset,
    subset_size: int | None,
    labels: list[int] | None = None,
) -> VisionDataset:
    if subset_size is None:
        return FilteredDataset(dataset, labels)

    label_count = len(labels) if labels is not None else len(dataset.classes)
    samples_per_label = ceil(subset_size / label_count)
    filtered_dataset = FilteredDataset(dataset, labels, samples_per_label)

    if len(filtered_dataset) < subset_size:
        raise ValueError(f"Unable to build a validation subset of size {subset_size} from {len(filtered_dataset)} filtered samples.")

    return Subset(filtered_dataset, range(subset_size))
