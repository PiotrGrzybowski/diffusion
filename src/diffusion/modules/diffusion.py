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

        self.register_buffer("sqrt_gammas", sqrt_gammas)
        self.register_buffer("sqrt_one_minus_gammas", sqrt_one_minus_gammas)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,))

    def q_sample(self, s_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(s_start)
        sqrt_gamma_t = self.sqrt_gammas[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_gamma_t = self.sqrt_one_minus_gammas[timesteps].view(-1, 1, 1, 1)

        return sqrt_gamma_t * s_start + sqrt_one_minus_gamma_t * noise


class DiffusionModule(LightningModule):
    def __init__(self, diffusion, model):
        super().__init__()
        self.diffusion = diffusion
        self.model = model

    def training_step(self, batch, batch_idx):
        x, _ = batch
        timesteps = self.diffusion.sample_timesteps(x.size(0)).to(device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)

        target_noise = self.diffusion.q_sample(x, timesteps, noise)
        predicted_noise = self.model(x, timesteps)
        # loss = nn.functional.mse_loss(predicted_noise, target_noise)
        loss = nn.functional.smooth_l1_loss(predicted_noise, target_noise)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        timesteps = self.diffusion.sample_timesteps(x.size(0)).to(device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)

        target_noise = self.diffusion.q_sample(x, timesteps, noise)
        predicted_noise = self.model(x, timesteps)
        # loss = nn.functional.mse_loss(predicted_noise, target_noise)
        loss = nn.functional.smooth_l1_loss(predicted_noise, target_noise)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)
