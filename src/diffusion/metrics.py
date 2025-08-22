import math

import torch
from torchmetrics.metric import Metric

from diffusion.diffusion_terms import DiffusionTerms
from diffusion.gaussian_utils import discretized_gaussian_log_likelihood, gaussian_kl


def vlb(target_terms: DiffusionTerms, predicted_terms: DiffusionTerms, timesteps: torch.Tensor) -> torch.Tensor:
    loss = gaussian_kl(target_terms.mean, target_terms.log_variance, predicted_terms.mean, predicted_terms.log_variance)
    decoder_nnl = -discretized_gaussian_log_likelihood(target_terms.x_start, predicted_terms.mean, predicted_terms.log_variance)

    decoder_idx = torch.where(timesteps == 0)

    loss[decoder_idx] = decoder_nnl[decoder_idx]
    loss = loss.mean() / math.log(2.0)

    return loss


class ScalarAverage(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor):
        self.sum += value
        self.total += 1

    def compute(self):
        return self.sum / self.total


class VarianceKL(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, target_log_variance: torch.Tensor, predicted_log_variance: torch.Tensor):
        loss = gaussian_kl(
            torch.zeros_like(target_log_variance), target_log_variance, torch.zeros_like(predicted_log_variance), predicted_log_variance
        ).mean()
        self.sum += loss
        self.total += 1

    def compute(self):
        return self.sum / self.total


if __name__ == "__main__":
    pass
    metric = ScalarAverage()
    metric = VarianceKL()
    metric.update(torch.tensor(1.0), torch.tensor(0.5))
    metric.update(torch.tensor(2.0), torch.tensor(1.0))
    print(metric.compute())
