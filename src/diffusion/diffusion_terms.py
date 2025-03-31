from dataclasses import dataclass

import torch


@dataclass
class DiffusionTerms:
    mean: torch.Tensor
    x_start: torch.Tensor
    epsilon: torch.Tensor
    variance: torch.Tensor
    log_variance: torch.Tensor
