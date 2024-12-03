import torch
import torch.nn as nn


class Factors(nn.Module):
    def __init__(self, betas: torch.Tensor) -> None:
        super().__init__()
        alphas = 1 - betas
        gammas = torch.cumprod(alphas, dim=0)

        self._register("betas", betas)
        self._register("alphas", alphas)
        self._register("gammas", gammas)

    def _register(self, name: str, tensor: torch.Tensor) -> None:
        self.register_buffer(name, tensor.view(-1, 1, 1, 1))
