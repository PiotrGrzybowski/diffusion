import torch
import torch.nn as nn
from lightning import LightningModule
from tqdm import tqdm

from diffusion.diffusion_factors import Factors
from diffusion.losses import LossInputs, MeanMseSimple
from diffusion.means import EpsilonMean, MeanInputs, MeanObjectives, MeanStrategy, XStartMean
from diffusion.variances2 import FixedSmallVariance, VarianceInputs, VarianceStrategy


class GaussianDiffusion(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        factors: Factors,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.factors = factors

        self.timesteps_count = timesteps
        self.in_channels = in_channels

        self.mean_strategy: MeanStrategy = EpsilonMean(self.factors)
        self.variance_strategy: VarianceStrategy = FixedSmallVariance(self.factors)

        self.posterior_variance_strategy: VarianceStrategy = FixedSmallVariance(self.factors)
        self.posterior_mean_strategy: MeanStrategy = XStartMean(self.factors)

        self.loss = MeanMseSimple(self.factors)

    def q_mean(self, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.factors.gammas[timesteps]) * x_start

    def q_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return 1 - self.factors.gammas[timesteps]

    def q_posterior_mean(self, x_t: torch.Tensor, x_start: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.posterior_mean_strategy.mean(MeanInputs(timesteps, x_t, x_start))

    def q_posterior_mean_objective(self, mean: torch.Tensor, x_start: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return self.posterior_mean_strategy.mean_objective(MeanObjectives(mean, x_start, epsilon))

    def q_posterior_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.posterior_variance_strategy.variance(VarianceInputs(timesteps, torch.empty()))

    def q_posterior_log_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.posterior_variance_strategy.log_variance(VarianceInputs(timesteps, torch.empty()))

    def p_mean(self, x_t: torch.Tensor, timesteps: torch.Tensor, mean_objective: torch.Tensor) -> torch.Tensor:
        return self.mean_strategy.mean(MeanInputs(timesteps, x_t, mean_objective))

    def p_mean_objective(self, output: torch.Tensor) -> torch.Tensor:
        return output[:, : self.in_channels]

    def p_variance_objective(self, output: torch.Tensor) -> torch.Tensor:
        if output.size(1) > self.in_channels:
            return output[:, self.in_channels :]
        else:
            return torch.empty(0)

    def p_variance(self, timesteps: torch.Tensor, variance_objective: torch.Tensor) -> torch.Tensor:
        return self.variance_strategy.variance(VarianceInputs(timesteps, variance_objective))

    def p_log_variance(self, timesteps: torch.Tensor, variance_objective: torch.Tensor) -> torch.Tensor:
        return self.variance_strategy.log_variance(VarianceInputs(timesteps, variance_objective))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        return self.q_mean(x_start, timesteps) + torch.sqrt(self.q_variance(timesteps)) * noise

    def model_step(self, x_start: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(0, self.timesteps_count, (x_start.size(0),)).to(device=x_start.device, dtype=torch.long)
        target_noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, timesteps, target_noise)

        mean = self.q_posterior_mean(x_t, x_start, timesteps)
        mean_objective = self.q_posterior_mean_objective(mean, x_start, target_noise)
        variance = self.q_posterior_variance(timesteps)
        log_variance = self.q_posterior_log_variance(timesteps)

        prediction = self.model(x_t, timesteps)
        predicted_mean_objective = self.p_mean_objective(prediction)
        predicted_mean = self.p_mean(x_t, timesteps, predicted_mean_objective)

        predicted_variance_objective = self.p_variance_objective(prediction)
        predicted_log_variance = self.p_log_variance(timesteps, predicted_variance_objective)

        loss_inputs = LossInputs(
            timesteps, mean, mean_objective, variance, log_variance, predicted_mean, predicted_mean_objective, predicted_log_variance
        )
        loss = self.loss(loss_inputs)

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

            output = self.model(x_t, timesteps)
            mean_objective = self.p_mean_objective(output)
            predicted_mean = self.p_mean(x_t, timesteps, mean_objective)

            variance_objective = self.p_variance_objective(output)
            predicted_variance = self.p_variance(timesteps, variance_objective)

            mask = (timesteps != 0).float().view(-1, 1, 1, 1)
            # TODO: use external sampler Manager as loss and variance.
            x_t = predicted_mean + torch.sqrt(predicted_variance) * noise * mask
        x_t = ((x_t + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return x_t

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
