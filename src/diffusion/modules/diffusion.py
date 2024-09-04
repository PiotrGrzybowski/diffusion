import torch
import torch.nn as nn
from lightning import LightningModule

from diffusion.schedulers.scheduler import Scheduler


class Diffusion(nn.Module):
    def __init__(self, timesteps: int, scheduler: Scheduler) -> None:
        super().__init__()
        self.timesteps = timesteps

        betas = scheduler.schedule()
        alphas = 1 - betas
        gammas = torch.cumprod(alphas, dim=0)
        gammas_prev = torch.cat([torch.tensor([1.0]), gammas[:-1]])

        sqrt_gammas = torch.sqrt(gammas)
        sqrt_one_minus_gammas = torch.sqrt(1 - gammas)
        sqrt_recip_alphas = torch.sqrt(1 / alphas)
        posterior_variance = betas * (1 - gammas_prev) / (1 - gammas)

        self.register_buffer("alphas", alphas.view(-1, 1, 1, 1))
        self.register_buffer("sqrt_gammas", sqrt_gammas.view(-1, 1, 1, 1))
        self.register_buffer("sqrt_one_minus_gammas", sqrt_one_minus_gammas.view(-1, 1, 1, 1))
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas.view(-1, 1, 1, 1))
        self.register_buffer("posterior_variance", posterior_variance.view(-1, 1, 1, 1))

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_gamma_t = self.sqrt_gammas[timesteps]
        sqrt_one_minus_gamma_t = self.sqrt_one_minus_gammas[timesteps]

        return sqrt_gamma_t * x_start + sqrt_one_minus_gamma_t * noise

    def reverse_diffusion_mean(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_gamma_t = self.sqrt_one_minus_gammas[t]
        alpha_t = self.alphas[t]

        mean = sqrt_recip_alpha_t * (x_t - ((1 - alpha_t) / sqrt_one_minus_gamma_t) * predicted_noise)

        return mean

    def reverse_diffusion_std(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.posterior_variance[t])


class DiffusionModule(LightningModule):
    def __init__(self, diffusion, model):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.loss_fn = nn.MSELoss()

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, _ = batch
        timesteps = self.diffusion.sample_timesteps(x.size(0)).to(device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)

        x_t = self.diffusion.q_sample(x, timesteps, noise)
        predicted_noise = self.model(x_t, timesteps)
        loss = self.loss_fn(predicted_noise, noise)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss = self.model_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss = self.model_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def predict_step(self, x_t: torch.Tensor) -> torch.Tensor:
        for i in range(self.diffusion.timesteps - 1, -1, -1):
            t = torch.tensor([i for _ in range(len(x_t))]).to(x_t.device)
            predicted_noise = self.model(x_t, t)

            noise = torch.randn_like(x_t).to(x_t.device)
            mean = self.diffusion.reverse_diffusion_mean(x_t, t, predicted_noise)
            std = self.diffusion.reverse_diffusion_std(t)

            x_t = mean + std * noise

        x_t = (x_t + 1.0) / 2.0

        return x_t

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=4e-5)
