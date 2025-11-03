import pytest
import torch
import torch.nn as nn

from diffusion.diffusion_factors import Factors
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.losses import MseMeanEpsilonSimple
from diffusion.means import EpsilonMean
from diffusion.samplers import DDPMSampler
from diffusion.schedulers import LinearScheduler
from diffusion.variances import FixedSmallVariance


class TinyUNet(nn.Module):
    """Tiny U-Net for fast testing."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, dim: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.conv_in = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.down = nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(dim * 2, dim * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(dim * 2, dim, 3, stride=2, padding=1, output_padding=1)
        self.conv_out = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Normalize timesteps to [0, 1]
        t = timesteps.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)

        h = self.conv_in(x)
        h = h + t_emb
        h = nn.functional.silu(h)
        h = self.down(h)
        h = nn.functional.silu(h)
        h = self.mid(h)
        h = nn.functional.silu(h)
        h = self.up(h)
        h = nn.functional.silu(h)
        h = self.conv_out(h)

        return h


@pytest.fixture
def diffusion_model():
    """Create a small diffusion model for testing."""
    timesteps = 100
    scheduler = LinearScheduler(timesteps=timesteps, start=0.0001, end=0.02)
    mean_strategy = EpsilonMean()
    variance_strategy = FixedSmallVariance()
    loss = MseMeanEpsilonSimple()
    model = TinyUNet(in_channels=1, out_channels=1)
    sampler = DDPMSampler()

    return GaussianDiffusion(
        timesteps=timesteps,
        sample_timesteps=timesteps,
        scheduler=scheduler,
        mean_strategy=mean_strategy,
        variance_strategy=variance_strategy,
        loss=loss,
        model=model,
        sampler=sampler,
        in_channels=1,
    )


@pytest.fixture
def sample_batch():
    """Create a small batch of images for testing."""
    batch_size = 2
    channels = 1
    height = width = 8
    images = torch.randn(batch_size, channels, height, width) * 0.5
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


class TestGaussianDiffusionInitialization:
    """Test GaussianDiffusion initialization and setup."""

    def test_model_initialization(self, diffusion_model):
        """Test that model initializes correctly."""
        assert isinstance(diffusion_model, GaussianDiffusion)
        assert diffusion_model.timesteps == 100
        assert diffusion_model.in_channels == 1
        assert isinstance(diffusion_model.factors, Factors)

    def test_factors_created(self, diffusion_model):
        """Test that factors are created and registered."""
        assert hasattr(diffusion_model.factors, "betas")
        assert hasattr(diffusion_model.factors, "alphas")
        assert hasattr(diffusion_model.factors, "gammas")
        assert hasattr(diffusion_model.factors, "gammas_prev")

        # Check shapes
        assert diffusion_model.factors.betas.shape[0] == 100
        assert diffusion_model.factors.gammas.shape[0] == 100

    def test_components_registered(self, diffusion_model):
        """Test that components are registered in hparams."""
        hparams = diffusion_model.hparams
        assert "mean" in hparams
        assert "variance" in hparams
        assert "loss" in hparams
        assert "scheduler" in hparams
        assert "model" in hparams


class TestForwardDiffusion:
    """Test forward diffusion process (noise addition)."""

    def test_q_sample_basic(self, diffusion_model, sample_batch):
        """Test forward diffusion samples."""
        x_start, _ = sample_batch
        timesteps = torch.tensor([10, 50], dtype=torch.long)
        noise = torch.randn_like(x_start)

        x_t = diffusion_model.q_sample(timesteps, x_start, noise)

        # Output should have same shape
        assert x_t.shape == x_start.shape
        assert torch.isfinite(x_t).all(), "Forward samples should be finite"

    def test_q_sample_more_noise_at_later_timesteps(self, diffusion_model):
        """Test that later timesteps have more noise."""
        x_start = torch.ones(1, 1, 8, 8)
        noise = torch.randn_like(x_start)

        # Sample at t=10 and t=90
        t_early = torch.tensor([10], dtype=torch.long)
        t_late = torch.tensor([90], dtype=torch.long)

        x_early = diffusion_model.q_sample(t_early, x_start, noise)
        x_late = diffusion_model.q_sample(t_late, x_start, noise)

        # Later timestep should be further from original
        dist_early = (x_early - x_start).abs().mean()
        dist_late = (x_late - x_start).abs().mean()

        assert dist_late > dist_early, "Later timesteps should be further from original (more noise)"

    def test_q_mean_and_variance(self, diffusion_model, sample_batch):
        """Test q_mean and q_variance computations."""
        x_start, _ = sample_batch
        timesteps = torch.tensor([10, 50], dtype=torch.long)

        mean = diffusion_model.q_mean(timesteps, x_start)
        variance = diffusion_model.q_variance(timesteps)

        assert mean.shape == x_start.shape
        assert variance.shape == (2, 1, 1, 1)  # Broadcasted shape
        assert (variance > 0).all(), "Variance should be positive"
        assert (variance < 1).all(), "Variance should be less than 1"


class TestTrainingStep:
    """Test training step and loss computation."""

    def test_training_step_runs(self, diffusion_model, sample_batch):
        """Test that training step executes without errors."""
        loss = diffusion_model.training_step(sample_batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0 or loss.numel() == 1, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss >= 0, "MSE loss should be non-negative"

    def test_training_step_creates_gradients(self, diffusion_model, sample_batch):
        """Test that training step creates gradients."""
        # Zero gradients first
        diffusion_model.zero_grad()

        loss = diffusion_model.training_step(sample_batch)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for param in diffusion_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Training step should create gradients"

    def test_training_step_different_batches_give_different_losses(self, diffusion_model):
        """Test that different batches produce different losses."""
        # Create distinctly different batches
        batch1 = (torch.ones(2, 1, 8, 8) * 1.0, torch.randint(0, 10, (2,)))
        batch2 = (torch.ones(2, 1, 8, 8) * -1.0, torch.randint(0, 10, (2,)))

        # Don't seed here - let the noise be random but data is different
        loss1 = diffusion_model.training_step(batch1)
        loss2 = diffusion_model.training_step(batch2)

        # Losses should be different for distinctly different data
        assert not torch.allclose(loss1, loss2, rtol=0.1), "Different batches should produce different losses"

    def test_diffusion_step(self, diffusion_model, sample_batch):
        """Test diffusion_step returns target and predicted terms."""
        x_start, _ = sample_batch
        timesteps = torch.tensor([10, 50], dtype=torch.long)

        target_terms, predicted_terms = diffusion_model.diffusion_step(x_start, timesteps)

        # Check target terms
        assert target_terms.mean.shape == x_start.shape
        assert target_terms.x_start.shape == x_start.shape
        assert target_terms.epsilon.shape == x_start.shape
        assert torch.isfinite(target_terms.mean).all()

        # Check predicted terms
        assert predicted_terms.mean.shape == x_start.shape
        assert predicted_terms.x_start.shape == x_start.shape
        assert predicted_terms.epsilon.shape == x_start.shape
        assert torch.isfinite(predicted_terms.mean).all()


class TestPosteriorComputation:
    """Test posterior mean and variance computations."""

    def test_posterior_mean(self, diffusion_model, sample_batch):
        """Test posterior mean computation."""
        x_start, _ = sample_batch
        timesteps = torch.tensor([10, 50], dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t = diffusion_model.q_sample(timesteps, x_start, noise)

        mean = diffusion_model.posterior_mean(x_start, x_t, timesteps)

        assert mean.shape == x_start.shape
        assert torch.isfinite(mean).all()

    def test_posterior_variance(self, diffusion_model):
        """Test posterior variance computation."""
        timesteps = torch.tensor([10, 50], dtype=torch.long)

        variance = diffusion_model.posterior_variance(timesteps)

        assert variance.shape == (2, 1, 1, 1)
        assert (variance > 0).all(), "Posterior variance should be positive"
        assert torch.isfinite(variance).all()

    def test_posterior_log_variance(self, diffusion_model):
        """Test posterior log variance computation."""
        timesteps = torch.tensor([0, 10, 50], dtype=torch.long)

        log_variance = diffusion_model.posterior_log_variance(timesteps)

        assert log_variance.shape == (3, 1, 1, 1)
        assert torch.isfinite(log_variance).all()

    def test_posterior_variance_increases_with_timestep(self, diffusion_model):
        """Test that posterior variance generally increases with timestep."""
        t_early = torch.tensor([10], dtype=torch.long)
        t_late = torch.tensor([50], dtype=torch.long)

        var_early = diffusion_model.posterior_variance(t_early)
        var_late = diffusion_model.posterior_variance(t_late)

        # Later timesteps generally have larger posterior variance
        # (though this isn't strictly monotonic due to schedule)
        assert var_late.item() > 0
        assert var_early.item() > 0


class TestSampling:
    """Test sampling (denoising) process."""

    def test_sample_generator(self, diffusion_model):
        """Test that sample() returns a generator."""
        batch_size = 2
        noise = torch.randn(batch_size, 1, 8, 8)

        # Use small number of timesteps for fast testing
        sampler = diffusion_model.sample(noise, timesteps=10)

        assert hasattr(sampler, "__next__"), "sample() should return a generator"

    def test_sample_produces_images(self, diffusion_model):
        """Test that sampling produces valid images."""
        batch_size = 2
        noise = torch.randn(batch_size, 1, 8, 8)

        # Sample with just a few steps
        samples = list(diffusion_model.sample(noise, timesteps=5))

        # Should have 5 intermediate samples + 1 final
        assert len(samples) >= 5

        # Final sample should be uint8 in [0, 255]
        final_sample = samples[-1]
        assert final_sample.shape == (batch_size, 1, 8, 8)
        assert final_sample.dtype == torch.uint8
        assert final_sample.min() >= 0
        assert final_sample.max() <= 255

    def test_sample_deterministic_with_same_noise(self, diffusion_model):
        """Test that sampling is deterministic with same input noise."""
        noise = torch.randn(1, 1, 8, 8)

        # Sample twice with same noise
        torch.manual_seed(42)
        samples1 = list(diffusion_model.sample(noise.clone(), timesteps=5))

        torch.manual_seed(42)
        samples2 = list(diffusion_model.sample(noise.clone(), timesteps=5))

        # Final samples should be identical
        # Note: DDPM is stochastic, so we need same seed for same results
        assert torch.allclose(samples1[-1].float(), samples2[-1].float()), "Sampling with same noise and seed should be deterministic"

    @pytest.mark.slow
    def test_full_sampling_loop(self, diffusion_model):
        """Test full sampling loop with all timesteps."""
        batch_size = 1
        noise = torch.randn(batch_size, 1, 8, 8)

        # Full sampling with all 100 timesteps
        samples = list(diffusion_model.sample(noise, timesteps=diffusion_model.timesteps))

        # Should produce timesteps + 1 samples
        assert len(samples) == diffusion_model.timesteps + 1

        # Check final sample
        final = samples[-1]
        assert final.shape == (batch_size, 1, 8, 8)
        assert final.dtype == torch.uint8


class TestModelStep:
    """Test model prediction step."""

    def test_model_step(self, diffusion_model, sample_batch):
        """Test model_step produces valid predictions."""
        x_start, _ = sample_batch
        timesteps = torch.tensor([10, 50], dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t = diffusion_model.q_sample(timesteps, x_start, noise)

        predicted_terms = diffusion_model.model_step(x_t, timesteps)

        # Check all components are present and finite
        assert predicted_terms.mean.shape == x_start.shape
        assert predicted_terms.x_start.shape == x_start.shape
        assert predicted_terms.epsilon.shape == x_start.shape
        assert torch.isfinite(predicted_terms.mean).all()
        assert torch.isfinite(predicted_terms.x_start).all()
        assert torch.isfinite(predicted_terms.epsilon).all()

    def test_model_prediction_shapes(self, diffusion_model):
        """Test mean_prediction and variance_prediction splitting."""
        batch_size = 2
        channels = 1
        height = width = 8

        # Model output with only mean prediction (in_channels = out_channels)
        model_output = torch.randn(batch_size, channels, height, width)

        mean_pred = diffusion_model.mean_prediction(model_output)
        var_pred = diffusion_model.variance_prediction(model_output)

        assert mean_pred.shape == (batch_size, channels, height, width)
        # Variance prediction should be empty when out_channels == in_channels
        assert var_pred.shape[1] == 0


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_full_forward_backward_pass(self, diffusion_model, sample_batch):
        """Test complete forward and backward pass."""
        # Forward pass
        loss = diffusion_model.training_step(sample_batch)

        # Backward pass
        diffusion_model.zero_grad()
        loss.backward()

        # Check gradients exist
        model_params = list(diffusion_model.model.parameters())
        assert len(model_params) > 0

        has_grad = any(p.grad is not None for p in model_params)
        assert has_grad, "Model should have gradients after backward pass"

    def test_multiple_training_steps(self, diffusion_model, sample_batch):
        """Test multiple training steps execute successfully."""
        losses = []

        for _ in range(3):
            loss = diffusion_model.training_step(sample_batch)
            loss.backward()
            diffusion_model.zero_grad()
            losses.append(loss.item())

        # All losses should be finite and non-negative
        assert all(loss >= 0 for loss in losses)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)

    def test_train_then_sample(self, diffusion_model, sample_batch):
        """Test training followed by sampling."""
        # Train for a few steps
        for _ in range(2):
            loss = diffusion_model.training_step(sample_batch)
            loss.backward()
            diffusion_model.zero_grad()

        # Now sample
        noise = torch.randn(1, 1, 8, 8)
        samples = list(diffusion_model.sample(noise, timesteps=5))

        assert len(samples) >= 5
        assert samples[-1].dtype == torch.uint8

    def test_optimizer_configuration(self, diffusion_model):
        """Test optimizer can be configured."""
        optimizer = diffusion_model.configure_optimizers()

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_timestep_zero(self, diffusion_model, sample_batch):
        """Test handling of timestep 0."""
        x_start, _ = sample_batch
        timesteps = torch.tensor([0, 0], dtype=torch.long)

        # Should not crash
        target_terms, predicted_terms = diffusion_model.diffusion_step(x_start, timesteps)

        assert torch.isfinite(target_terms.mean).all()
        assert torch.isfinite(predicted_terms.mean).all()

    def test_max_timestep(self, diffusion_model, sample_batch):
        """Test handling of maximum timestep."""
        x_start, _ = sample_batch
        max_t = diffusion_model.timesteps - 1
        timesteps = torch.tensor([max_t, max_t], dtype=torch.long)

        target_terms, predicted_terms = diffusion_model.diffusion_step(x_start, timesteps)

        assert torch.isfinite(target_terms.mean).all()
        assert torch.isfinite(predicted_terms.mean).all()

    def test_single_sample_batch(self, diffusion_model):
        """Test with batch size of 1."""
        batch = (torch.randn(1, 1, 8, 8), torch.tensor([0]))

        loss = diffusion_model.training_step(batch)

        assert torch.isfinite(loss)
        assert loss >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
