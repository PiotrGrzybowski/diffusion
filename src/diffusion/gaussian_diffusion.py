import torch
import torch.nn as nn
from lightning import LightningModule
from rich.console import Console
from torch.nn.functional import mse_loss

from diffusion.diffusion_factors import Factors
from diffusion.losses import DiffusionLoss, LossInputs
from diffusion.means import MeanInputs, MeanObjectives, MeanStrategy
from diffusion.schedulers import Scheduler
from diffusion.variances import VarianceInputs, VarianceStrategy


class GaussianDiffusion(LightningModule):
    def __init__(
        self,
        timesteps: int,
        scheduler: Scheduler,
        posterior_mean: MeanStrategy,
        posterior_variance: VarianceStrategy,
        model_mean: MeanStrategy,
        model_variance: VarianceStrategy,
        loss: DiffusionLoss,
        model: nn.Module,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.factors = Factors(scheduler)

        self.timesteps = timesteps
        self.in_channels = in_channels

        self.model_mean = model_mean
        self.model_variance = model_variance

        self.posterior_variance = posterior_variance
        self.posterior_mean = posterior_mean

        self.loss = loss

        self.register_components()

    def q_mean(self, timesteps: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.factors.gammas[timesteps]) * x_start

    def q_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return 1 - self.factors.gammas[timesteps]

    def q_posterior_mean(self, timesteps: torch.Tensor, x_t: torch.Tensor, x_start: torch.Tensor) -> torch.Tensor:
        inputs = MeanInputs(self.factors, timesteps, x_t, x_start)
        return self.posterior_mean.mean(inputs)

    def q_posterior_mean_objective(self, mean: torch.Tensor, x_start: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """Tricky, so in order to idendify whether mean, x_start or epsilone is right mean_objective we need to evaluate it from model strategy not the q_posterior strategy. The later one is set to XStartMean so its objective always gives x_start."""
        inputs = MeanObjectives(mean, x_start, epsilon)
        return self.model_mean.mean_objective(inputs)

    def q_posterior_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        inputs = VarianceInputs(self.factors, timesteps)
        return self.posterior_variance.variance(inputs)

    def q_posterior_log_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        inputs = VarianceInputs(self.factors, timesteps)
        return self.posterior_variance.log_variance(inputs)

    def q_posterior_variance_objective(self, timesteps: torch.Tensor) -> torch.Tensor:
        inputs = VarianceInputs(self.factors, timesteps)
        return self.posterior_variance.log_variance(inputs)

    def p_mean(self, timesteps: torch.Tensor, x_t: torch.Tensor, mean_objective: torch.Tensor) -> torch.Tensor:
        inputs = MeanInputs(self.factors, timesteps, x_t, mean_objective)
        return self.model_mean.mean(inputs)

    def p_mean_objective(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:, : self.in_channels]

    def p_variance(self, timesteps: torch.Tensor, variance_objective: torch.Tensor) -> torch.Tensor:
        return self.model_variance.variance(VarianceInputs(self.factors, timesteps, variance_objective))

    def p_log_variance(self, timesteps: torch.Tensor, variance_objective: torch.Tensor) -> torch.Tensor:
        return self.model_variance.log_variance(VarianceInputs(self.factors, timesteps, variance_objective))

    def p_variance_objective(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.shape[1] > self.in_channels:
            return model_output[:, self.in_channels :]
        else:
            return torch.empty(0)

    def q_sample(self, timesteps: torch.Tensor, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.q_mean(timesteps, x_start) + torch.sqrt(self.q_variance(timesteps)) * noise

    def model_step(self, x_start: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = torch.randint(0, self.timesteps, (x_start.size(0),)).to(device=x_start.device, dtype=torch.long)
        target_noise = torch.randn_like(x_start)

        x_t = self.q_sample(timesteps, x_start, target_noise)

        mean = self.q_posterior_mean(timesteps, x_t, x_start)
        mean_objective = self.q_posterior_mean_objective(mean, x_start, target_noise)
        variance = self.q_posterior_variance(timesteps)
        log_variance = self.q_posterior_log_variance(timesteps)

        prediction = self.model(x_t, timesteps)
        predicted_mean_objective = self.p_mean_objective(prediction)
        predicted_mean = self.p_mean(timesteps, x_t, predicted_mean_objective)

        predicted_variance_objective = self.p_variance_objective(prediction)
        predicted_log_variance = self.p_log_variance(timesteps, predicted_variance_objective)

        loss_inputs = LossInputs(
            self.factors,
            timesteps,
            mean,
            mean_objective,
            variance,
            log_variance,
            predicted_mean,
            predicted_mean_objective,
            predicted_log_variance,
        )

        mean_mse = mse_loss(predicted_mean, mean)
        loss = self.loss.forward(loss_inputs)

        return loss.mean(), mean_mse

    @torch.inference_mode()
    def sample(self, shape: tuple[int, ...], verbose: bool = False) -> torch.Tensor:
        indexes = list(range(self.timesteps))[::-1]
        x_t = torch.randn(shape, device=self.device)

        console = Console()

        for index in indexes:
            console.print(f"Sampling... {indexes[0] - index}/{len(indexes)}", end="\r")
            timesteps = torch.full((shape[0],), index, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x_t).to(device=self.device)

            output = self.model(x_t, timesteps)
            mean_objective = self.p_mean_objective(output)
            predicted_mean = self.p_mean(timesteps, x_t, mean_objective)

            variance_objective = self.p_variance_objective(output)
            predicted_variance = self.p_variance(timesteps, variance_objective)

            mask = (timesteps != 0).float().view(-1, 1, 1, 1)
            # TODO: use external sampler Manager as loss and variance.
            x_t = predicted_mean + torch.sqrt(predicted_variance) * noise * mask
        x_t = ((x_t + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return x_t

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch
        loss, mean_mse = self.model_step(x_start)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mean_mse", mean_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch
        loss, mean_mse = self.model_step(x_start)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mean_mse", mean_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.sample(batch.shape)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-4)

    def register_components(self):
        components = {
            "mean": self.model_mean.__class__.__name__,
            "variance": self.model_variance.__class__.__name__,
            "loss": self.loss.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
            "model": self.model.__class__.__name__,
        }
        self.save_hyperparameters(components)
