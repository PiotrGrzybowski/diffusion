import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from rich.console import Console
from torchmetrics import MeanSquaredError

from diffusion.diffusion_factors import Factors
from diffusion.diffusion_terms import DiffusionTerms
from diffusion.losses import DiffusionLoss, LossInputs
from diffusion.means import MeanInputs, MeanStrategy
from diffusion.metrics import ScalarAverage, vlb
from diffusion.samplers import SampleInputs, Sampler
from diffusion.schedulers import Scheduler
from diffusion.variances import VarianceInputs, VarianceStrategy


class TimestepSampler:
    def __init__(self, timesteps: int) -> None:
        super().__init__()
        self.timesteps = timesteps

    def sample_like(self, x: torch.Tensor, device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (x.size(0),)).to(device=device, dtype=torch.long)

    def full_like(self, x: torch.Tensor, timestep: int, device) -> torch.Tensor:
        return torch.full((x.size(0),), timestep, device=device, dtype=torch.long)


class GaussianDiffusion(LightningModule):
    def __init__(
        self,
        timesteps: int,
        sample_timesteps: int,
        scheduler: Scheduler,
        mean_strategy: MeanStrategy,
        variance_strategy: VarianceStrategy,
        loss: DiffusionLoss,
        model: nn.Module,
        sampler: Sampler,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.factors = Factors(scheduler.schedule())

        self.timesteps = timesteps
        self.sample_timesteps = sample_timesteps
        self.in_channels = in_channels

        self.mean_strategy = mean_strategy
        self.variance_strategy = variance_strategy

        self.loss = loss
        self.sampler = sampler

        self.timestep_sampler = TimestepSampler(timesteps)
        self.mean_mse = MeanSquaredError()
        self.nll_metric = ScalarAverage()

        self.register_components()

    def q_mean(self, timesteps: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.factors.gammas[timesteps]) * x_start

    def q_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return 1 - self.factors.gammas[timesteps]

    def q_sample(self, timesteps: torch.Tensor, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        mean = self.q_mean(timesteps, x_start)
        variance = self.q_variance(timesteps)
        return mean + torch.sqrt(variance) * noise

    def posterior_mean(self, x_start: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        gammas = self.factors.gammas[timesteps]
        gammas_prev = self.factors.gammas_prev[timesteps]
        alphas = self.factors.alphas[timesteps]
        betas = self.factors.betas[timesteps]

        x_t_coeff = torch.sqrt(alphas) * (1 - gammas_prev) / (1 - gammas)
        x_start_coeff = torch.sqrt(gammas_prev) * betas / (1 - gammas)

        return x_t_coeff * x_t + x_start_coeff * x_start

    def posterior_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        betas = self.factors.betas[timesteps]
        gammas = self.factors.gammas[timesteps]
        gammas_prev = self.factors.gammas_prev[timesteps]

        return betas * (1 - gammas_prev) / (1 - gammas)

    def posterior_log_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = torch.where(timesteps == 0, torch.tensor(1), timesteps)
        variance = self.posterior_variance(timesteps)
        return torch.log(variance)

    def posterior_step(self, x_start: torch.Tensor, x_t: torch.Tensor, epsilon: torch.Tensor, timesteps: torch.Tensor) -> DiffusionTerms:
        mean = self.posterior_mean(x_start, x_t, timesteps)
        variance = self.posterior_variance(timesteps)
        log_variance = self.posterior_log_variance(timesteps)

        return DiffusionTerms(mean, x_start, epsilon, variance, log_variance)

    def model_step(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> DiffusionTerms:
        prediction = self.model(x_t, timesteps)
        mean_prediction = self.mean_prediction(prediction)
        variance_prediction = self.variance_prediction(prediction)

        mean_outputs = self.mean_strategy.forward(MeanInputs(x_t, mean_prediction, self.factors, timesteps))
        variance_outputs = self.variance_strategy.forward(VarianceInputs(self.factors, timesteps, variance_prediction))

        return DiffusionTerms(
            mean_outputs.mean, mean_outputs.x_start, mean_outputs.epsilon, variance_outputs.variance, variance_outputs.log_variance
        )

    def diffusion_step(self, x_start: torch.Tensor, timesteps: torch.Tensor) -> tuple[DiffusionTerms, DiffusionTerms]:
        epsilon = torch.randn_like(x_start, device=self.device)
        x_t = self.q_sample(timesteps, x_start, epsilon)
        target_terms = self.posterior_step(x_start, x_t, epsilon, timesteps)
        predicted_terms = self.model_step(x_t, timesteps)

        return target_terms, predicted_terms

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch
        timesteps = self.timestep_sampler.sample_like(x_start, self.device)

        target_terms, predicted_terms = self.diffusion_step(x_start, timesteps)

        loss = self.loss.forward(LossInputs(target_terms, predicted_terms, self.factors, timesteps))
        mean_mse = self.mean_mse(predicted_terms.mean, target_terms.mean)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("mean_mse", mean_mse, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x_start, _ = batch
        console = Console()
        nll = torch.tensor(0.0, device=self.device)

        for timestep in list(range(self.sample_timesteps))[::-1]:
            timesteps = self.timestep_sampler.full_like(x_start, timestep, self.device)
            target_terms, predicted_terms = self.diffusion_step(x_start, timesteps)
            nll += vlb(target_terms, predicted_terms, timesteps)
            console.print(f"Val sampling: {self.sample_timesteps - timestep}/{self.sample_timesteps}, nll={nll}", end="\r")
        self.nll_metric.update(nll)
        self.log("nll", self.nll_metric, on_epoch=True, prog_bar=True)

    def on_after_backward(self) -> None:
        norm = 0.0
        for p in self.model.parameters():
            norm += p.grad.data.norm(2).item() ** 2
        norm = norm**0.5
        self.log("gradient_norm", norm, on_step=True, on_epoch=True, prog_bar=True)

    @torch.inference_mode()
    def sample(self, batch: torch.Tensor, timesteps: int):
        x_t = batch

        times = np.linspace(0, timesteps - 1, timesteps, dtype=int).tolist()[::-1]

        original_factors = self.factors
        self.factors = self.build_sample_factors(len(times))
        for timestep in times:
            timestep = torch.full((x_t.size(0),), timestep, device=x_t.device, dtype=torch.long)
            predicted_terms = self.model_step(x_t, timestep)
            predicted_x_start = predicted_terms.x_start
            inputs = SampleInputs(
                predicted_terms.mean, predicted_terms.variance, predicted_x_start, predicted_terms.epsilon, self.factors, timestep
            )
            x_t = self.sampler.sample(inputs)
            x_prev = ((x_t + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            yield x_prev
        self.factors = original_factors
        yield x_prev

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-4)

    def register_components(self):
        components = {
            "mean": self.mean_strategy.__class__.__name__,
            "variance": self.variance_strategy.__class__.__name__,
            "loss": self.loss.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
            "model": self.model.__class__.__name__,
        }
        self.save_hyperparameters(components)

    def mean_prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:, : self.in_channels].contiguous()

    def variance_prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:, self.in_channels :]

    def build_sample_factors(self, steps: int) -> Factors:
        indexes = np.linspace(0, self.timesteps - 1, steps, dtype=int).tolist()[::-1]
        last_gamma = 1.0
        betas = []
        for i, gamma in enumerate(self.factors.gammas):
            if i in indexes:
                betas.append(1 - gamma / last_gamma)
                last_gamma = gamma
        return Factors(torch.tensor(betas)).to(self.device)
