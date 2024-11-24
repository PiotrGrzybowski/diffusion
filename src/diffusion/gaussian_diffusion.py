import torch
import torch.nn as nn
from lightning import LightningModule

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

    # def q_posterior_mean(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    #     sqrt_recip_alphas = torch.sqrt(1 / self.factors.alphas[timesteps])
    #     sqrt_one_minus_gammas = torch.sqrt(1 - self.factors.gammas[timesteps])
    #     betas = self.factors.betas[timesteps]
    #
    #     return sqrt_recip_alphas * (x_t - (betas / sqrt_one_minus_gammas) * noise)

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
    def sample(self, shape: tuple[int, ...]):
        indexes = list(range(self.timesteps_count))[::-1]
        x_t = torch.randn(shape).to(device=self.device)
        from tqdm import tqdm

        for index in tqdm(indexes):
            timesteps = torch.full((shape[0],), index)
            noise = torch.randn_like(x_t)

            predicted_noise, model_variance = self._model_mean_variance(x_t, timesteps)
            predicted_x_start = self._x_start_from_noise(x_t, predicted_noise, timesteps)

            predicted_mean = self.p_mean(x_t, predicted_x_start, timesteps)
            predicted_variance = self.p_variance(timesteps, model_variance)

            mask = (timesteps != 0).float().view(-1, 1, 1, 1)
            x_t = predicted_mean + predicted_variance * noise * mask
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


#
# class DummyModule(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
#         return torch.rand_like(x)
#
#
# if __name__ == "__main__":
#     bs = 4
#     x = (torch.rand((bs, 1, 5, 5)) - 0.5) * 2
#     y = torch.randint(0, 10, (bs,))
#     t = torch.randint(0, 10, (bs,))
#     from diffusion.schedulers.linear_scheduler import LinearScheduler
#
#     timesteps = 1000
#     model = DummyModule()
#     scheduler = LinearScheduler(timesteps, 0.0001, 0.02)
#     variance_type = VarianceType.FIXED_SMALL
#     loss_type = LossType.MeanMSE
#
#     module = GaussianDiffusion(model, timesteps, loss_type, variance_type, scheduler)
#     timesteps = torch.randint(0, 4, (4,)).to(dtype=torch.long)
#     noise = torch.randn_like(x)
#
#     q_mean = module.q_mean(x, timesteps)
#     print(f"q_mean: {q_mean.squeeze()}")
#
#     q_variance = module.q_variance(timesteps)
#     print(f"q_variance: {q_variance.squeeze()}")
#
#     target_mean = module.q_posterior_mean(x, timesteps, noise)
#     print(f"target_mean: {target_mean.squeeze()}")
#
#     target_variance = module.q_posterior_variance(timesteps)
#     print(f"target_variance: {target_variance.squeeze()}")
#
#     target_log_variance = module.variance_manager.log_variance(variance_type, timesteps, module.factors)
#     print(f"target_log_variance: {target_log_variance.squeeze()}")
#     print(f"target_log_variance: {torch.log(target_variance).squeeze()}")
#
#     output = model(x, t)
#     model_mean, model_variance = module._model_mean_variance(x, t)
#
#     predicted_mean = module.p_mean(x, t, model_mean)
#     print(f"predicted_mean: {predicted_mean.squeeze()}")
#
#     predicted_variance = module.p_variance(t, model_variance)
#     print(f"predicted_variance: {predicted_variance.squeeze()}")
#
#     loss = module.loss_manager.forward(
#         LossType.MeanMSE, t, module.factors, target_mean, target_variance, predicted_mean, predicted_variance
#     )
#     print(f"Mse loss: {loss}")
#
#     loss = module.loss_manager.forward(
#         LossType.SIMPLE_MSE, t, module.factors, target_mean, target_variance, predicted_mean, predicted_variance
#     )
#     print(f"SimpleMse loss: {loss}")
#
#     loss = module.loss_manager.forward(
#         LossType.VARIATIONAL_BOUND, t, module.factors, target_mean, target_variance, predicted_mean, predicted_variance
#     )
#     print(f"VariationalBound loss: {loss}")
#
#     loss = module.model_step(x)
#     print(f"Model step loss: {loss}")


# class Diffusion(nn.Module):
#     def __init__(self, timesteps: int, scheduler: Scheduler) -> None:
#         super().__init__()
#         self.timesteps = timesteps
#         self.factors = Factors(scheduler.schedule())
#         self.variance_strategy = FixedSmallVariance()
#         self.variance_manager = VarianceManager()
#
#         betas = scheduler.schedule()
#         alphas = 1 - betas
#         gammas = torch.cumprod(alphas, dim=0)
#         gammas_prev = torch.cat([torch.tensor([1.0]), gammas[:-1]])
#
#         sqrt_gammas = torch.sqrt(gammas)
#         sqrt_one_minus_gammas = torch.sqrt(1 - gammas)
#         sqrt_recip_alphas = torch.sqrt(1 / alphas)
#         posterior_variance = betas * (1 - gammas_prev) / (1 - gammas)
#
#         self.register_buffer("alphas", alphas.view(-1, 1, 1, 1))
#         self.register_buffer("sqrt_gammas", sqrt_gammas.view(-1, 1, 1, 1))
#         self.register_buffer("sqrt_one_minus_gammas", sqrt_one_minus_gammas.view(-1, 1, 1, 1))
#         self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas.view(-1, 1, 1, 1))
#         self.register_buffer("posterior_variance", posterior_variance.view(-1, 1, 1, 1))
#
#     def sample_timesteps(self, batch_size: int) -> torch.Tensor:
#         return torch.randint(0, self.timesteps, (batch_size,))
#
#     def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         sqrt_gamma_t = self.sqrt_gammas[timesteps]
#         sqrt_one_minus_gamma_t = self.sqrt_one_minus_gammas[timesteps]
#
#         return sqrt_gamma_t * x_start + sqrt_one_minus_gamma_t * noise
#
#     def reverse_diffusion_mean(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
#         sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
#         sqrt_one_minus_gamma_t = self.sqrt_one_minus_gammas[t]
#         alpha_t = self.alphas[t]
#
#         mean = sqrt_recip_alpha_t * (x_t - ((1 - alpha_t) / sqrt_one_minus_gamma_t) * predicted_noise)
#
#         return mean
#
#     def reverse_diffusion_std(self, t: torch.Tensor) -> torch.Tensor:
#         return torch.sqrt(self.posterior_variance[t])
#
#
# class DiffusionModule(LightningModule):
#     def __init__(self, diffusion, model):
#         super().__init__()
#         self.diffusion = diffusion
#         self.model = model
#         self.loss_fn = nn.MSELoss()
#         self.factors = Factors(scheduler.schedule())
#         self.variance_strategy = FixedSmallVariance()
#         self.variance_manager = VarianceManager()
#         self.loss = SimpleMseLoss()
#         self.loss_manager = LossManager()
#
#     def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         x_0, _ = batch
#         timesteps = self.diffusion.sample_timesteps(x_0.size(0)).to(device=x_0.device, dtype=torch.long)
#         noise = torch.randn_like(x_0)
#
#         x_t = self.diffusion.q_sample(x_0, timesteps, noise)
#         prediction = self.model(x_t, timesteps)
#
#         if prediction.shape[1] == 12:
#             predicted_noise = prediction[:, :3]
#             predicted_variance = prediction[:, 3:]
#         else:
#             predicted_noise = prediction
#             predicted_variance = torch.empty(0)
#         variance = self.variance_manager.variance(self.variance_strategy, timesteps, self.factors, predicted_variance)
#         target_variance = self.factor.posterior_variance[timesteps]
#         loss = self.loss_manager.forward(
#             self.loss, timesteps, self.factors, target_mean, target_variance, predicted_mean, predicted_variance
#         )
#         # TODO: include noise / posterior mean in loss Manager signature?
#         loss = self.loss_fn(predicted_noise, noise)
#         return loss
#
#     def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         loss = self.model_step(batch)
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
#         return loss
#
#     def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         loss = self.model_step(batch)
#         self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
#         return loss
#
#     # def on_train_epoch_end(self) -> None:
#     #     # lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
#     #     # self.log("learning_rate", lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#
#     def forward(self, x_t: torch.Tensor) -> torch.Tensor:
#         for i in range(self.diffusion.timesteps - 1, -1, -1):
#             t = torch.tensor([i for _ in range(len(x_t))]).to(x_t.device)
#             predicted_noise = self.model(x_t, t)
#
#             noise = torch.randn_like(x_t).to(x_t.device)
#             mean = self.diffusion.reverse_diffusion_mean(x_t, t, predicted_noise)
#             std = self.diffusion.reverse_diffusion_std(t)
#
#             x_t = mean + std * noise
#
#         x_t = (x_t + 1.0) / 2.0
#
#         return x_t
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
#         # steps_per_epoch = len(self.trainer.datamodule.train_dataloader())  # Replace with actual steps per epoch
#         # total_steps = self.trainer.max_epochs * steps_per_epoch
#
#         return optimizer
#         # return {
#         #     "optimizer": optimizer,
#         #     "lr_scheduler": {
#         #         "scheduler": scheduler,
#         #         "interval": "step",  # Update the LR after each training step (batch)
#         #         "frequency": 1,  # How often to call the scheduler
#         #     },
#         # }
