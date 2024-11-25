import torch
import torch.nn as nn
from lightning import LightningModule
from tqdm import tqdm

from diffusion.diffusion_factors import Factors
from diffusion.losses import LossManager, LossType
from diffusion.schedulers.scheduler import Scheduler
from diffusion.variances import VarianceManager, VarianceType


class GaussianDiffusion(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        loss_type: LossType,
        variance_type: VarianceType,
        scheduler: Scheduler,
    ) -> None:
        super().__init__()
        self.model = model
        self.factors = Factors(scheduler.schedule())
        self.variance_type = variance_type
        self.variance_manager = VarianceManager()
        self.loss_type = loss_type
        self.loss_manager = LossManager()

        self.timesteps_count = timesteps

    def q_mean(self, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.factors.gammas[timesteps]) * x_start

    def q_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return 1 - self.factors.gammas[timesteps]

    def q_posterior_mean(self, x_t: torch.Tensor, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        gammas = self.factors.gammas[timesteps]
        gammas_prev = self.factors.gammas_prev[timesteps]
        alphas = self.factors.alphas[timesteps]
        betas = self.factors.betas[timesteps]

        x_t_coeff = torch.sqrt(alphas) * (1 - gammas_prev) / (1 - gammas)
        x_0_coeff = torch.sqrt(gammas_prev) * betas / (1 - gammas)

        return x_t_coeff * x_t + x_0_coeff * x_start

    def q_posterior_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.variance_manager.variance(VarianceType.FIXED_SMALL, timesteps, self.factors)

    def q_posterior_log_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.variance_manager.log_variance(VarianceType.FIXED_SMALL, timesteps, self.factors)

    def p_mean(self, x_t: torch.Tensor, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.q_posterior_mean(x_t, x_start, timesteps)

    def p_variance(self, timesteps: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return self.variance_manager.variance(self.variance_type, timesteps, self.factors, prediction)

    def p_log_variance(self, timesteps: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return self.variance_manager.log_variance(self.variance_type, timesteps, self.factors, prediction)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Samples x_t from x_start for specified timesteps."""
        if noise is None:
            noise = torch.randn_like(x_start)

        return self.q_mean(x_start, timesteps) + torch.sqrt(self.q_variance(timesteps)) * noise

    def model_step(self, x_start: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(0, self.timesteps_count, (x_start.size(0),)).to(device=x_start.device, dtype=torch.long)
        target_noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, timesteps, target_noise)

        target_mean = self.q_posterior_mean(x_t, x_start, timesteps)
        # target_variance = self.q_posterior_variance(timesteps)
        predicted_noise, model_variance = self._model_mean_variance(x_t, timesteps)

        predicted_x_start = self._x_start_from_noise(x_t, predicted_noise, timesteps)
        predicted_mean = self.p_mean(x_t, predicted_x_start, timesteps)
        # predicted_variance = self.p_variance(timesteps, model_variance)

        target_log_variance = self.q_posterior_log_variance(timesteps)
        predicted_log_variance = self.p_log_variance(timesteps, model_variance)

        loss = self.loss_manager.forward(
            self.loss_type,
            timesteps,
            self.factors,
            target_mean,
            target_log_variance,
            target_noise,
            predicted_mean,
            predicted_log_variance,
            predicted_noise,
        )

        return loss

    @torch.inference_mode()
    def sample(self, shape: tuple[int, ...], verbose: bool = False) -> torch.Tensor:
        indexes = list(range(self.timesteps_count))[::-1]
        x_t = torch.randn(shape, device=self.device)

        if verbose:
            indexes = tqdm(indexes, desc="Sampling")

        for index in indexes:
            timesteps = torch.full((shape[0],), index, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x_t).to(device=self.device)

            predicted_noise, model_variance = self._model_mean_variance(x_t, timesteps)
            predicted_x_start = self._x_start_from_noise(x_t, predicted_noise, timesteps)

            predicted_mean = self.p_mean(x_t, predicted_x_start, timesteps)
            predicted_variance = self.p_variance(timesteps, model_variance)

            mask = (timesteps != 0).float().view(-1, 1, 1, 1)
            x_t = predicted_mean + torch.sqrt(predicted_variance) * noise * mask
        x_t = ((x_t + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return x_t

    def _x_start_from_noise(self, x_t: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        gammas = self.factors.gammas[timesteps]
        x_start = 1 / torch.sqrt(gammas) * (x_t - torch.sqrt(1 - gammas) * noise)
        x_start = torch.clamp(x_start, -1, 1)
        return x_start

    def _model_mean_variance(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(x_t, timesteps)
        if self.variance_type in {VarianceType.TRAINABLE, VarianceType.TRAINABLE_RANGE}:
            predicted_noise = prediction[:, :3]
            predicted_variance = prediction[:, 3:]
        else:
            predicted_noise = prediction
            predicted_variance = torch.empty(0)
        return predicted_noise, predicted_variance

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch
        loss = self.model_step(x_start)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_start, _ = batch
        loss = self.model_step(x_start)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-4)
