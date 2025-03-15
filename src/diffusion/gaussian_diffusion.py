import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from rich.console import Console
from torch.nn.functional import mse_loss

from diffusion.diffusion_factors import Factors
from diffusion.diffusion_terms import DiffusionTerms
from diffusion.losses import DiffusionLoss, LossInputs
from diffusion.means import MeanInputs, MeanStrategy
from diffusion.samplers import Sampler
from diffusion.schedulers import Scheduler
from diffusion.variances import VarianceInputs, VarianceStrategy


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

        self.register_components()

    def q_sample(self, timesteps: torch.Tensor, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        mean = torch.sqrt(self.factors.gammas[timesteps]) * x_start
        variance = 1 - self.factors.gammas[timesteps]
        return mean + variance * noise

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

        mean = mean_outputs.mean
        x_start = mean_outputs.x_start
        epsilon = mean_outputs.epsilon

        variance = variance_outputs.variance
        log_variance = variance_outputs.log_variance

        return DiffusionTerms(mean, x_start, epsilon, variance, log_variance)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch
        timesteps = torch.randint(0, self.timesteps, (x_start.size(0),)).to(device=x_start.device, dtype=torch.long)

        epsilon = torch.randn_like(x_start)
        x_t = self.q_sample(timesteps, x_start, epsilon)

        target_terms = self.posterior_step(x_start, x_t, epsilon, timesteps)
        predicted_terms = self.model_step(x_t, timesteps)

        loss = self.loss.forward(LossInputs(target_terms, predicted_terms, self.factors, timesteps))

        mean_mse = mse_loss(predicted_terms.mean, target_terms.mean)
        mean_epsilon_mse = mse_loss(predicted_terms.epsilon, target_terms.epsilon)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_mse", mean_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_epsilon_mse", mean_epsilon_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch

        timesteps = torch.randint(0, self.timesteps, (x_start.size(0),)).to(device=x_start.device, dtype=torch.long)
        epsilon = torch.randn_like(x_start)
        x_t = self.q_sample(timesteps, x_start, epsilon)

        target_terms = self.posterior_step(x_start, x_t, epsilon, timesteps)
        predicted_terms = self.model_step(x_t, timesteps)

        loss = self.loss.forward(LossInputs(target_terms, predicted_terms, self.factors, timesteps))
        mean_mse = mse_loss(predicted_terms.mean, target_terms.mean)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mean_mse", mean_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def build_sample_factors(self, steps: int) -> Factors:
        indexes = np.linspace(0, self.timesteps - 1, steps, dtype=int).tolist()[::-1]
        last_gamma = 1.0
        betas = []
        for i, gamma in enumerate(self.factors.gammas):
            if i in indexes:
                betas.append(1 - gamma / last_gamma)
                last_gamma = gamma
        return Factors(torch.tensor(betas)).to(self.device)

    @torch.inference_mode()
    def sample(self, batch: torch.Tensor, steps: int) -> torch.Tensor:
        # original_factors = self.factors
        # self.factors = self.build_sample_factors(steps)

        x_t = batch
        console = Console()
        indexes = list(range(steps))[::-1]
        for index in indexes:
            console.print(f"Samplingen... {indexes[0] - index}/{len(indexes)}", end="\r")

            timesteps = torch.full((batch.shape[0],), index, device=self.device, dtype=torch.long)
            predicted_terms = self.model_step(x_t, timesteps)
            # posterior_terms = self.posterior_step(x_t, timesteps)

            x_prev = self.sampler.sample(predicted_terms, self.factors, timesteps)

            x_t = x_prev

        x_t = ((x_t + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        # self.factors = original_factors
        return x_t

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.sample(batch, self.sample_timesteps)

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
        return model_output[:, : self.in_channels]

    def variance_prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.shape[1] > self.in_channels:
            return model_output[:, self.in_channels :]
        else:
            return torch.empty(0)
