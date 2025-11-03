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
        return self.conv(x) - 3


def create_test_data(model, device, batch_size, channels, height, width, timesteps_count, factors, seed=None):
    """Create test data with optional fixed seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)

    # Create sample data with reasonable ranges
    x_start = torch.randn(batch_size, channels, height, width, device=device).clamp(-1.0, 1.0)
    x_t = torch.randn_like(x_start)
    noise = torch.randn_like(x_start)
    timesteps = torch.randint(1, timesteps_count - 1, (batch_size,), device=device)

    model_output = model(x_t, timesteps)

    mean_prediction = model_output[:, :3]
    log_variance_prediction = model_output[:, 3:]

    target_log_variance = torch.randn_like(x_start) - 5

    target_terms = DiffusionTerms(
        mean=torch.randn_like(x_start),
        x_start=x_start,
        epsilon=noise,  # Ground truth noise
        variance=torch.exp(target_log_variance),
        log_variance=target_log_variance,
    )

    # Create predicted terms
    predicted_terms = DiffusionTerms(
        mean=mean_prediction,  # Will be detached in hybrid loss
        x_start=x_start,
        epsilon=mean_prediction,  # Used in epsilon loss
        variance=torch.exp(log_variance_prediction),
        log_variance=log_variance_prediction,
    )

    # Create loss inputs
    loss_inputs = LossInputs(target=target_terms, predicted=predicted_terms, factors=factors, timesteps=timesteps)

    return loss_inputs


def test_hybrid_loss_gradient_isolation():
    """
    Test that HybridLoss correctly isolates gradients:
    - Mean gradients should flow to model parameters
    - Variance gradients should NOT flow to mean parameters (frozen mean)
    - Variance gradients should flow to variance parameters
    """
    # Setup
    device = torch.device("cpu")
    batch_size = 2
    channels = 3
    height = width = 2
    timesteps_count = 3

    # Create model with separate mean and variance outputs
    model = MockModel(out_channels=6).to(device)

    # Create scheduler and factors
    scheduler = LinearScheduler(timesteps=timesteps_count, start=1e-4, end=2e-2)
    factors = Factors(scheduler.schedule())

    # Create losses
    mean_loss = MseMeanEpsilonSimple()
    variance_loss = VLB()
    hybrid_loss = Hybrid(mean_loss, variance_loss, omega=1.0)

    # Test individual losses first to understand gradient flow
    print("=== INDIVIDUAL LOSS ANALYSIS ===")

    # Test mean loss only
    loss_inputs1 = create_test_data(model, device, batch_size, channels, height, width, timesteps_count, factors, seed=42)

    model.zero_grad()
    mean_only_loss = mean_loss.forward(loss_inputs1)
    print(f"\nMean loss: {mean_only_loss.item():.6f}")

    mean_only_loss.backward()
    mean_loss_weight_grad = model.conv.weight.grad.clone()
    mean_loss_bias_grad = model.conv.bias.grad.clone()

    print("Mean loss only:")
    print(f"  Weight grad norm: {mean_loss_weight_grad.norm().item():.6f}")
    print(f"  Bias grad norm: {mean_loss_bias_grad.norm().item():.6f}")
    print(f"  Mean channels grad norm: {mean_loss_weight_grad[:3].norm().item():.6f}")
    print(f"  Var channels grad norm: {mean_loss_weight_grad[3:].norm().item():.6f}")

    # Test variance loss only (with frozen mean)
    # Need fresh forward pass to properly isolate gradients
    torch.manual_seed(123)
    x_start = torch.randn(batch_size, channels, height, width, device=device).clamp(-1.0, 1.0)
    x_t = torch.randn_like(x_start)
    noise = torch.randn_like(x_start)
    timesteps = torch.randint(1, timesteps_count - 1, (batch_size,), device=device)

    model.zero_grad()
    model_output = model(x_t, timesteps)

    mean_part = model_output[:, :3]
    var_part = model_output[:, 3:]

    target_log_variance = torch.randn_like(x_start) - 5

    target_terms = DiffusionTerms(
        mean=torch.randn_like(mean_part) * 0.1,  # Independent target mean
        x_start=x_start,
        epsilon=noise,
        variance=torch.exp(target_log_variance),
        log_variance=target_log_variance,
    )

    predicted_terms = DiffusionTerms(
        mean=mean_part.detach(),
        x_start=x_start,
        epsilon=mean_part.detach(),
        variance=torch.exp(var_part),  # Connected variance
        log_variance=var_part,
    )

    loss_inputs2 = LossInputs(target=target_terms, predicted=predicted_terms, factors=factors, timesteps=timesteps)

    var_only_loss = variance_loss.forward(loss_inputs2)
    print(f"\nVariance loss: {var_only_loss.item():.6f}")

    var_only_loss.backward()

    var_loss_weight_grad = model.conv.weight.grad.clone()
    var_loss_bias_grad = model.conv.bias.grad.clone()

    print("Variance loss only (with detached mean):")
    print(f"  Weight grad norm: {var_loss_weight_grad.norm().item():.6f}")
    print(f"  Bias grad norm: {var_loss_bias_grad.norm().item():.6f}")
    print(f"  Mean channels grad norm: {var_loss_weight_grad[:3].norm().item():.6f}")
    print(f"  Var channels grad norm: {var_loss_weight_grad[3:].norm().item():.6f}")


def test_hybrid_loss_gradient_2():
    device = torch.device("cpu")
    batch_size = 2
    channels = 3
    height = width = 2
    timesteps_count = 3

    # Create model with separate mean and variance outputs
    model = MockModel(out_channels=6).to(device)

    # Create scheduler and factors
    scheduler = LinearScheduler(timesteps=timesteps_count, start=1e-4, end=2e-2)
    factors = Factors(scheduler.schedule())

    # Create losses
    mean_loss = MseMeanEpsilonSimple()
    variance_loss = VLB()
    hybrid_loss = Hybrid(mean_loss, variance_loss, omega=1.0)

    print("=== INDIVIDUAL LOSS ANALYSIS ===")

    # Test mean loss only
    loss_inputs1 = create_test_data(model, device, batch_size, channels, height, width, timesteps_count, factors, seed=42)

    model.zero_grad()
    # mu_loss = mean_loss.forward(loss_inputs1)
    # print(f"\nMean loss: {mu_loss.item():.6f}")

    loss_inputs1.predicted.mean = loss_inputs1.predicted.mean.detach()
    loss_inputs1.predicted.epsilon = loss_inputs1.predicted.epsilon.detach()

    var_loss = variance_loss.forward(loss_inputs1)
    # mu_loss.backward()
    var_loss.backward()
    weights_grad = model.conv.weight.grad.clone()
    bias_grad = model.conv.bias.grad.clone()

    print("Mean loss only:")
    print(f"  Weight grad norm: {weights_grad.norm().item():.6f}")
    print(f"  Bias grad norm: {bias_grad.norm().item():.6f}")
    print(f"  Mean channels grad norm: {weights_grad[:3].norm().item():.6f}")
    print(f"  Var channels grad norm: {bias_grad[3:].norm().item():.6f}")


def test_hybrid_loss_gradient_final():
    device = torch.device("cpu")
    batch_size = 2
    channels = 3
    height = width = 2
    timesteps_count = 3

    # Create model with separate mean and variance outputs
    model = MockModel(out_channels=6).to(device)

    # Create scheduler and factors
    scheduler = LinearScheduler(timesteps=timesteps_count, start=1e-4, end=2e-2)
    factors = Factors(scheduler.schedule())

    # Create losses
    mean_loss = MseMeanEpsilonSimple()
    variance_loss = VLB()
    hybrid_loss = Hybrid(mean_loss, variance_loss, omega=1.0)

    print("=== INDIVIDUAL LOSS ANALYSIS ===")

    # Test mean loss only
    loss_inputs1 = create_test_data(model, device, batch_size, channels, height, width, timesteps_count, factors, seed=42)

    model.zero_grad()
    loss = hybrid_loss.forward(loss_inputs1)
    loss.backward()
    weights_grad = model.conv.weight.grad.clone()
    bias_grad = model.conv.bias.grad.clone()

    print("Mean loss only:")
    print(f"  Weight grad norm: {weights_grad.norm().item():.6f}")
    print(f"  Bias grad norm: {bias_grad.norm().item():.6f}")
    print(f"  Mean channels grad norm: {weights_grad[:3].norm().item():.6f}")
    print(f"  Var channels grad norm: {bias_grad[3:].norm().item():.6f}")


if __name__ == "__main__":
    test_hybrid_loss_gradient_isolation()
