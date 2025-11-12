import numpy as np
import torch
from lightning import LightningModule
from rich.console import Console
from torchmetrics import MeanSquaredError

from diffusion.diffusion_factors import Factors
from diffusion.diffusion_terms import DiffusionTerms
from diffusion.image_samplers import ImageSampler, SampleInputs
from diffusion.losses import DiffusionLoss, LossInputs
from diffusion.means import MeanInputs, MeanStrategy
from diffusion.metrics import ScalarAverage, VarianceKL, vlb
from diffusion.schedulers import Scheduler
from diffusion.timestep_samplers import TimestepSampler
from diffusion.variances import VarianceInputs, VarianceStrategy


class GaussianDiffusion(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        mean_strategy: MeanStrategy,
        variance_strategy: VarianceStrategy,
        loss: DiffusionLoss,
        scheduler: Scheduler,
        image_sampler: ImageSampler,
        timestep_sampler: TimestepSampler,
        timesteps: int,
        sample_timesteps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.mean_strategy = mean_strategy
        self.variance_strategy = variance_strategy
        self.loss = loss
        self.scheduler = scheduler
        self.image_sampler = image_sampler
        self.timestep_sampler = timestep_sampler

        self.factors = Factors(self.scheduler.schedule())

        self.timesteps = timesteps
        self.sample_timesteps = sample_timesteps

        self.train_mean_mse = MeanSquaredError()
        self.val_mean_mse = MeanSquaredError()
        self.train_variance_kl = VarianceKL()
        self.val_variance_kl = VarianceKL()
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
        mean_mse = self.train_mean_mse(predicted_terms.mean, target_terms.mean)
        variance_kl = self.train_variance_kl(target_terms.log_variance, predicted_terms.log_variance)

        self.log_dict(
            {
                "train/mean_mse": mean_mse,
                "train/variance_kl": variance_kl,
                "train/loss": loss,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int) -> None:
        if dataloader_idx == 0:
            x_start, _ = batch
            timesteps = self.timestep_sampler.sample_like(x_start, self.device)

            target_terms, predicted_terms = self.diffusion_step(x_start, timesteps)

            loss = self.loss.forward(LossInputs(target_terms, predicted_terms, self.factors, timesteps))
            mean_mse = self.val_mean_mse(predicted_terms.mean, target_terms.mean)
            variance_kl = self.val_variance_kl(target_terms.log_variance, predicted_terms.log_variance)

            self.log_dict(
                {
                    "val/mean_mse": mean_mse,
                    "val/variance_kl": variance_kl,
                    "val/loss": loss,
                },
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
        else:
            x_start, _ = batch
            console = Console()
            nll = torch.tensor(0.0, device=self.device)

            for timestep in list(range(self.sample_timesteps))[::-1]:
                timesteps = self.timestep_sampler.full_like(x_start, timestep, self.device)
                target_terms, predicted_terms = self.diffusion_step(x_start, timesteps)
                nll += vlb(target_terms, predicted_terms, timesteps)
                console.print(f"Val sampling: {self.sample_timesteps - timestep}/{self.sample_timesteps}, nll={nll}", end="\r")
            self.nll_metric.update(nll)
            self.log("nll", self.nll_metric, on_epoch=True, prog_bar=True, add_dataloader_idx=False, sync_dist=True)

    def on_after_backward(self) -> None:
        norm = 0.0
        for p in self.model.parameters():
            assert p.grad is not None
            norm += p.grad.data.norm(2).item() ** 2
        norm = norm**0.5
        self.log("gradient_norm", norm, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    @torch.inference_mode()
    def sample(self, batch: torch.Tensor, timesteps: int):
        x_t = batch
        x_prev = torch.empty_like(x_t)

        times = np.linspace(0, timesteps - 1, timesteps, dtype=int).tolist()[::-1]

        original_factors = self.factors
        self.factors = self.build_sample_factors(len(times))
        for timestep in times:
            timestep = torch.full((x_t.size(0),), timestep, device=x_t.device, dtype=torch.long)
            predicted_terms = self.model_step(x_t, timestep)
            inputs = SampleInputs(
                predicted_terms.mean,
                predicted_terms.variance,
                predicted_terms.x_start,
                predicted_terms.epsilon,
                self.factors,
                timestep,
            )
            x_t = self.image_sampler.sample(inputs)
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
        }
        self.save_hyperparameters(components)

    def mean_prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:, : self.model.in_channels].contiguous()

    def variance_prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:, self.model.in_channels :]

    def build_sample_factors(self, steps: int) -> Factors:
        indexes = np.linspace(0, self.timesteps - 1, steps, dtype=int)
        last_gamma = 1.0
        betas = []
        for i, gamma in enumerate(self.factors.gammas):
            if i in indexes:
                betas.append(1 - gamma / last_gamma)
                last_gamma = gamma
        return Factors(torch.tensor(betas)).to(self.device)
