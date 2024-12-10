import torch
import torch.nn as nn


class Factors(nn.Module):
    def __init__(self, betas: torch.Tensor) -> None:
        super().__init__()
        self.betas = betas.view(-1, 1, 1, 1)
        self.alphas = 1 - self.betas
        self.gammas = torch.cumprod(self.alphas, dim=0)
