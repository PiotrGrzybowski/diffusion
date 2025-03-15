import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from rich.console import Console

from diffusion.diffusion_factors import Factors
from diffusion.losses import DiffusionLoss, LossInputs
from diffusion.means import MeanStrategy
from diffusion.samplers import SampleOutputs, Sampler
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

    def posterior_mean(self, x_start, x_t, timesteps) -> torch.Tensor:
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
        variance = self.variance(timesteps)
        return torch.log(variance)

    def model_step(self, x_start: torch.Tensor) -> LossInputs:
        timesteps = torch.randint(0, self.timesteps, (x_start.size(0),)).to(device=x_start.device, dtype=torch.long)

        target_epsilon = torch.randn_like(x_start)
        x_t = self.q_sample(timesteps, x_start, target_epsilon)

        posterior_mean = self.posterior_mean(x_start, x_t, timesteps)
        posterior_variance = self.posterior_variance(timesteps)
        posterior_log_variance = self.posterior_log_variance(timesteps)

        prediction = self.model(x_t, timesteps)
        mean_prediction = self.mean_prediction(prediction)
        variance_prediction = self.variance_prediction(prediction)

        mean_outputs = self.mean_strategy.forward(x_t, mean_prediction, self.factors, timesteps)

        variance_inputs = VarianceInputs(self.factors, timesteps, variance_prediction)
        predicted_log_variance = self.variance_strategy.log_variance(variance_inputs)
        predicted_variance = self.variance_strategy.variance(variance_inputs)

        outputs = LossInputs(
            target_x_start=x_start,
            target_epsilon=target_epsilon,
            target_mean=posterior_mean,
            target_variance=posterior_variance,
            target_log_variance=posterior_log_variance,
            predicted_mean=mean_outputs.mean,
            predicted_x_start=mean_outputs.x_start,
            predicted_epsilon=mean_outputs.epsilon,
            predicted_variance=predicted_variance,
            predicted_log_variance=predicted_log_variance,
            factors=self.factors,
            timesteps=timesteps,
        )

        return outputs

    def sample_step(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> SampleOutputs:
        prediction = self.model(x_t, timesteps)
        mean_prediction = self.mean_prediction(prediction)
        variance_prediction = self.variance_prediction(prediction)

        mean_outputs = self.mean_strategy.forward(x_t, mean_prediction, self.factors, timesteps)
        predicted_variance = self.variance_strategy.variance(VarianceInputs(self.factors, timesteps, variance_prediction))

        return self.sampler.sample(mean_outputs, predicted_variance, self.factors, timesteps)

    @torch.inference_mode()
    def sample(self, batch: torch.Tensor, steps: int) -> torch.Tensor:
        # original_factors = self.factors
        # self.factors = self.build_sample_factors(steps)

        x_t = batch
        console = Console()
        indexes = list(range(steps))[::-1]
        for index in indexes:
            console.print(f"Sampling... {indexes[0] - index}/{len(indexes)}", end="\r")
            timesteps = torch.full((batch.shape[0],), index, device=self.device, dtype=torch.long)

            outputs = self.sample_step(x_t, timesteps)
            x_t = outputs.x_prev

        x_t = ((x_t + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        # self.factors = original_factors
        return x_t

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch
        outputs = self.model_step(x_start)

        loss = self.loss.forward(outputs)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log("train_mean_mse", mean_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch
        loss, mean_mse = self.model_step(x_start)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mean_mse", mean_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

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
