import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from diffusion.schedulers import LinearScheduler, Scheduler
from diffusion.unet import Unet


class Diffusion(nn.Module):
    def __init__(self, timesteps: int, scheduler: Scheduler) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.betas = scheduler.schedule(self.timesteps)
        self.alphas = 1 - self.betas
        self.gammas = torch.cumprod(self.alphas, axis=0)
        self.gammas_prev = torch.cat([torch.ones(1), self.gammas[:-1]])
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        sqrt_gammas = torch.sqrt(self.gammas)
        sqrt_one_minus_gammas = torch.sqrt(1.0 - self.gammas)

        self.posterior_variance = self.betas * (1.0 - self.gammas_prev) / (1.0 - self.gammas)

        self.register_buffer("sqrt_gammas", sqrt_gammas)
        self.register_buffer("sqrt_one_minus_gammas", sqrt_one_minus_gammas)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        batch_size = len(x_start)
        noise = torch.randn_like(x_start)
        sqrt_gamma_t = self.sqrt_gammas[timesteps].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_gamma_t = self.sqrt_one_minus_gammas[timesteps].reshape(batch_size, 1, 1, 1)
        return sqrt_gamma_t * x_start + sqrt_one_minus_gamma_t * noise


class DiffusionModule(LightningModule):
    def __init__(self, diffusion: Diffusion, model: nn.Module) -> None:
        super().__init__()
        self.diffusion = diffusion.cuda()
        self.model = model.cuda()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        timesteps = torch.randint(0, self.diffusion.timesteps, (len(x),))
        noise = torch.randn_like(x)
        x_noisy = self.diffusion.q_sample(x, timesteps, noise)
        # out = torch.rand_like(x)
        out = self.model(x_noisy, timesteps)
        loss = F.mse_loss(out, x_noisy)
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


if __name__ == "__main__":
    timesteps = 300
    beta_start = 0.0001
    beta_end = 0.02
    scheduler = LinearScheduler(beta_start, beta_end)
    diffusion = Diffusion(timesteps, scheduler)

    batch_size = 4
    t = torch.randint(0, timesteps, (batch_size,))
    images = torch.rand((batch_size, 1, 64, 64)) * 2 - 1

    q_sampled = diffusion.q_sample(images, t)
    print(q_sampled.shape)

    model = Unet(1, 128, [128, 128, 256, 512], heads=4, head_dim=32)
    module = DiffusionModule(diffusion, model)
    module.training_step(images, 0)
