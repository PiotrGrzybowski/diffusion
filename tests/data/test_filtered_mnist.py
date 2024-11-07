from pathlib import Path

import pytest
from diffusion.data.filtered_mnist import FilteredMNIST
from torchvision.datasets import MNIST


@pytest.fixture
def labels() -> list[int]:
    return [0, 1]


@pytest.fixture
def samples_per_label() -> int:
    return 10


def test_filtered_mnist_instantiation(data_path: Path, labels: list[int], samples_per_label: int) -> None:
    mnist = MNIST(data_path, train=True, download=True)
    FilteredMNIST(mnist, labels=labels, samples_per_label=samples_per_label)


def test_filtered_mnist_length(data_path: Path, labels: list[int], samples_per_label: int) -> None:
    mnist = MNIST(data_path, train=True, download=True)
    filtered_mnist = FilteredMNIST(mnist, labels=labels, samples_per_label=samples_per_label)

    assert len(filtered_mnist) == 20


def test_filtered_mnist_only_valid_labels(data_path: Path, labels: list[int], samples_per_label: int) -> None:
    mnist = MNIST(data_path, train=True, download=True)
    filtered_mnist = FilteredMNIST(mnist, labels=labels, samples_per_label=samples_per_label)

    for i in range(len(filtered_mnist)):
        _, label = filtered_mnist[i]
        assert label in labels


def test_filtered_mnist_valid_sample_per_label(data_path: Path, labels: list[int], samples_per_label: int) -> None:
    mnist = MNIST(data_path, train=True, download=True)
    filtered_mnist = FilteredMNIST(mnist, labels=labels, samples_per_label=samples_per_label)

    counts = {label: 0 for label in labels}
    for i in range(len(filtered_mnist)):
        _, label = filtered_mnist[i]
        counts[int(label)] += 1

    assert all(count == samples_per_label for count in counts.values())


def test_filtered_mnist_all_labels_all_samples(data_path: Path) -> None:
    mnist = MNIST(data_path, train=True, download=True)
    filtered_mnist = FilteredMNIST(mnist)

    assert len(filtered_mnist) == 60000


def test_filtered_mnist_all_labels(data_path: Path) -> None:
    mnist = MNIST(data_path, train=True, download=True)
    filtered_mnist = FilteredMNIST(mnist, samples_per_label=1000)

    counts = {label: 0 for label in range(10)}
    for i in range(len(filtered_mnist)):
        _, label = filtered_mnist[i]
        counts[int(label)] += 1

    assert all(count == 1000 for count in counts.values())
