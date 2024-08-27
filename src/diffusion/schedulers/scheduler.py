from abc import ABC, abstractmethod

import torch


class Scheduler(ABC):
    @abstractmethod
    def schedule(self) -> torch.Tensor:
        pass
