import torch


class Scheduler:
    def schedule(self, timesteps: int) -> torch.Tensor:
        pass


class LinearScheduler:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end

    def schedule(self, timesteps: int) -> torch.Tensor:
        return torch.linspace(self.start, self.end, timesteps)
