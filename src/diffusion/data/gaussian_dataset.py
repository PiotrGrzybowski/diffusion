import torch
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.data = torch.randn(shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
