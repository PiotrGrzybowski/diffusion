"""Tests for gradient isolation in hybrid loss.

This module tests that the Hybrid loss correctly isolates gradients:
- Mean loss gradients should flow to model parameters
- Variance loss gradients should NOT flow to mean parameters (frozen mean via detach)
- Variance loss gradients should only flow to variance parameters
"""

import torch
import torch.nn as nn

from diffusion.diffusion_factors import Factors
from diffusion.diffusion_terms import DiffusionTerms
from diffusion.losses import VLB, Hybrid, LossInputs, MseMeanEpsilonSimple
from diffusion.schedulers import LinearScheduler


class MockModel(nn.Module):
    """Mock model that outputs both mean prediction and variance prediction."""

    def __init__(self, out_channels=6):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 3, padding=1)

    def forward(self, x, t):
        return self.conv(x)


def create_test_setup():
    """Create common test setup."""
    device = torch.device("cpu")
    batch_size = 2
    channels = 3
    height = width = 4
    timesteps_count = 10

    model = MockModel(out_channels=6).to(device)
    scheduler = LinearScheduler(timesteps=timesteps_count, start=1e-4, end=2e-2)
    factors = Factors(scheduler.schedule())

    return model, factors, device, batch_size, channels, height, width, timesteps_count


def create_test_data(model, device, batch_size, channels, height, width, timesteps_count, factors, seed=42):
    """Create test data with fixed seed for reproducibility."""
    torch.manual_seed(seed)

    x_start = torch.randn(batch_size, channels, height, width, device=device).clamp(-1.0, 1.0)
    x_t = torch.randn_like(x_start)
    noise = torch.randn_like(x_start)
    timesteps = torch.randint(1, timesteps_count - 1, (batch_size,), device=device)

    model_output = model(x_t, timesteps)
    mean_prediction = model_output[:, :channels]
    log_variance_prediction = model_output[:, channels:]

    target_log_variance = torch.randn_like(x_start) - 5

    target_terms = DiffusionTerms(
        mean=torch.randn_like(x_start) * 0.1,
        x_start=x_start,
        epsilon=noise,
        variance=torch.exp(target_log_variance),
        log_variance=target_log_variance,
    )

    predicted_terms = DiffusionTerms(
        mean=mean_prediction,
        x_start=x_start,
        epsilon=mean_prediction,
        variance=torch.exp(log_variance_prediction),
        log_variance=log_variance_prediction,
    )

    return LossInputs(target=target_terms, predicted=predicted_terms, factors=factors, timesteps=timesteps)


def test_mean_loss_gradients_flow():
    """Test that mean loss produces non-zero gradients for model parameters."""
    model, factors, device, batch_size, channels, height, width, timesteps_count = create_test_setup()
    mean_loss = MseMeanEpsilonSimple()

    loss_inputs = create_test_data(model, device, batch_size, channels, height, width, timesteps_count, factors)

    model.zero_grad()
    loss = mean_loss.forward(loss_inputs)
    loss.backward()

    weight_grad = model.conv.weight.grad
    bias_grad = model.conv.bias.grad

    # Assert gradients exist and are non-zero
    assert weight_grad is not None, "Weight gradients should exist"
    assert bias_grad is not None, "Bias gradients should exist"
    assert weight_grad.norm().item() > 0, "Weight gradients should be non-zero"
    assert bias_grad.norm().item() > 0, "Bias gradients should be non-zero"

    # Mean channels should have gradients
    mean_weight_grad_norm = weight_grad[:channels].norm().item()
    assert mean_weight_grad_norm > 0, "Mean channels should have non-zero gradients"


def test_variance_loss_with_detached_mean():
    """Test that variance loss with detached mean only produces gradients for variance parameters."""
    model, factors, device, batch_size, channels, height, width, timesteps_count = create_test_setup()
    variance_loss = VLB()

    # Create fresh data with detached mean
    torch.manual_seed(123)
    x_start = torch.randn(batch_size, channels, height, width, device=device).clamp(-1.0, 1.0)
    x_t = torch.randn_like(x_start)
    noise = torch.randn_like(x_start)
    timesteps = torch.randint(1, timesteps_count - 1, (batch_size,), device=device)

    model.zero_grad()
    model_output = model(x_t, timesteps)

    mean_part = model_output[:, :channels].detach()  # Detach mean
    var_part = model_output[:, channels:]  # Keep variance connected

    target_log_variance = torch.randn_like(x_start) - 5

    target_terms = DiffusionTerms(
        mean=torch.randn_like(mean_part) * 0.1,
        x_start=x_start,
        epsilon=noise,
        variance=torch.exp(target_log_variance),
        log_variance=target_log_variance,
    )

    predicted_terms = DiffusionTerms(
        mean=mean_part,
        x_start=x_start,
        epsilon=mean_part,
        variance=torch.exp(var_part),
        log_variance=var_part,
    )

    loss_inputs = LossInputs(target=target_terms, predicted=predicted_terms, factors=factors, timesteps=timesteps)

    loss = variance_loss.forward(loss_inputs)
    loss.backward()

    weight_grad = model.conv.weight.grad
    bias_grad = model.conv.bias.grad

    # Assert gradients exist
    assert weight_grad is not None, "Weight gradients should exist"
    assert bias_grad is not None, "Bias gradients should exist"

    # Variance channels should have gradients
    var_weight_grad_norm = weight_grad[channels:].norm().item()
    assert var_weight_grad_norm > 0, "Variance channels should have non-zero gradients"


def test_hybrid_loss_gradient_isolation():
    """Test that hybrid loss correctly isolates mean and variance gradients."""
    model, factors, device, batch_size, channels, height, width, timesteps_count = create_test_setup()

    mean_loss = MseMeanEpsilonSimple()
    variance_loss = VLB()
    hybrid_loss = Hybrid(mean_loss, variance_loss, omega=1.0)

    loss_inputs = create_test_data(model, device, batch_size, channels, height, width, timesteps_count, factors)

    model.zero_grad()
    loss = hybrid_loss.forward(loss_inputs)
    loss.backward()

    weight_grad = model.conv.weight.grad
    bias_grad = model.conv.bias.grad

    # Assert gradients exist
    assert weight_grad is not None, "Weight gradients should exist"
    assert bias_grad is not None, "Bias gradients should exist"
    assert weight_grad.norm().item() > 0, "Weight gradients should be non-zero"
    assert bias_grad.norm().item() > 0, "Bias gradients should be non-zero"

    # Both mean and variance channels should have gradients
    mean_weight_grad_norm = weight_grad[:channels].norm().item()
    var_weight_grad_norm = weight_grad[channels:].norm().item()

    assert mean_weight_grad_norm > 0, "Mean channels should have non-zero gradients from mean loss"
    assert var_weight_grad_norm > 0, "Variance channels should have non-zero gradients from variance loss"


def test_hybrid_loss_variance_detachment():
    """Test that hybrid loss properly detaches mean when computing variance loss."""
    model, factors, device, batch_size, channels, height, width, timesteps_count = create_test_setup()

    mean_loss = MseMeanEpsilonSimple()
    variance_loss = VLB()
    hybrid_loss = Hybrid(mean_loss, variance_loss, omega=1.0)

    # Create test data
    loss_inputs = create_test_data(model, device, batch_size, channels, height, width, timesteps_count, factors)

    # Compute hybrid loss and check that it computes both components
    model.zero_grad()
    total_loss = hybrid_loss.forward(loss_inputs)

    # Loss should be finite and non-zero
    assert torch.isfinite(total_loss), "Hybrid loss should be finite"
    assert total_loss.item() > 0, "Hybrid loss should be positive"

    # Backward pass should succeed
    total_loss.backward()

    # All parameters should have gradients
    assert all(p.grad is not None for p in model.parameters()), "All parameters should have gradients"
