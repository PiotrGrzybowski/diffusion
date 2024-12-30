import torch
import torch.nn as nn

from diffusion.schedulers.scheduler import Scheduler


class Factors(nn.Module):
    def __init__(self, scheduler: Scheduler) -> None:
        super().__init__()
        betas = scheduler.schedule()
        alphas = 1 - betas
        gammas = torch.cumprod(alphas, dim=0)
        gammas_prev = torch.cat([torch.tensor([1.0]), gammas[:-1]])

        self._register("betas", betas)
        self._register("alphas", alphas)
        self._register("gammas", gammas)
        self._register("gammas_prev", gammas_prev)

    def _register(self, name: str, tensor: torch.Tensor) -> None:
        self.register_buffer(name, tensor.view(-1, 1, 1, 1))
