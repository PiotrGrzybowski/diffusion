from torchvision import datasets


class DatasetMap:
    dataset_map = {
        "mnist": datasets.MNIST,
        "kmnist": datasets.KMNIST,
        "fashion": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }

    @staticmethod
    def get_dataset_class(name: str) -> type:
        """
        Returns the dataset class from torchvision.datasets based on the given name.

        Args:
            name (str): The name of the dataset (e.g., "mnist", "cifar10").

        Returns:
            type: The dataset class from torchvision.datasets.

        Raises:
            ValueError: If the dataset name is not in the mapping.
        """
        name = name.lower()
        if name not in DatasetMap.dataset_map:
            raise ValueError(f"Dataset '{name}' is not supported. Available options are: {list(DatasetMap.dataset_map.keys())}")
        return DatasetMap.dataset_map[name]
