# Diffusion Models: From Theory to Practice

A PyTorch Lightning reimplementation of **DDPM** and **Improved Diffusion** papers, serving as the practical companion to the book *"Foundations of Diffusion Models"*.

## Overview

This repository bridges the gap between mathematical theory and practical implementation of diffusion models. Every component from noise schedulers to loss functions maps directly to the mathematical foundations presented in the accompanying book, making it an ideal resource for both learning and experimentation.

### What This Repository Offers

- **Principled Implementation**: Code structure mirrors mathematical derivations
- **Lightning Architecture**: Modern, scalable training framework with multi-GPU support
- **Flexible Configuration**: Hydra-based compositional configs for rapid experimentation
- **Educational Focus**: Direct mapping between code modules and book chapters
- **Production-Ready**: Experiment tracking, checkpointing, and distributed training out-of-the-box

### Papers Implemented
- **[DDPM]** Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **[Improved Diffusion]** Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd diffusion

# Install with uv (recommended)
uv sync
```

## Quick Start

### Training Your First Model

Here's a complete example that trains a diffusion model on CIFAR-10 with the hybrid loss (Simple MSE + VLB):

```bash
uv run python src/diffusion/scripts/train.py \
    train=True \
    validate=True \
    trainer=ddp \
    trainer.max_epochs=70 \
    diffusion/model=unet \
    diffusion/mean_strategy=epsilon \
    diffusion/variance_strategy=trainable_range \
    diffusion/loss=hybrid \
    logger=tensorboard \
    callbacks.image_generation.every_n_epochs=5 \
    data=cifar10 \
    dim=32 \
    in_channels=3 \
    out_channels=6 \
    predict_samples=16 \
    batch_size=32 \
    run_name="epsilon-trainable_range-hybrid"
```

**What's happening here?**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `trainer=ddp` | Distributed Data Parallel | Multi-GPU training |
| `diffusion/mean_strategy=epsilon` | Predict noise $\epsilon$ | Model predicts added noise |
| `diffusion/variance_strategy=trainable_range` | Learned variance | Model learns variance interpolation |
| `diffusion/loss=hybrid` | MSE + VLB | Combines simple loss with VLB (ω=0.001) |
| `out_channels=6` | 3 mean + 3 variance | RGB channels × 2 for learned variance |
| `logger=tensorboard` | TensorBoard | Experiment tracking |
| `run_name` | Experiment identifier | Used for logging and checkpoint organization |

### Simpler Example: MNIST with Basic Setup

```bash
uv run python src/diffusion/scripts/train.py \
    trainer=gpu \
    trainer.max_epochs=30 \
    diffusion/mean_strategy=epsilon \
    diffusion/variance_strategy=fixed_small \
    diffusion/loss=mse_mean_epsilon_simple \
    data=mnist \
    logger=wandb \
    run_name="mnist-basic"
```

### Sampling from a Trained Model

After training, generate samples using the `run_name` that was used during training:

```bash
uv run python src/diffusion/scripts/sample.py \
    run_name="epsilon-trainable_range-hybrid" \
    samples=36 \
    sample_timesteps=200
```

**Note**: The sampling script automatically locates the checkpoint and configuration from the logs directory based on the `run_name`. All training runs are saved to `logs/` with their configurations and checkpoints preserved.

### Fast Sampling

Reduce `sample_timesteps` for faster generation (with potential quality trade-off):


## Core Components
The implementation is organized around five key architectural decisions that you can mix and match:

### 1. Noise Schedulers
- **Linear** (`diffusion/scheduler=linear`): Linear $\beta$ schedule from DDPM paper
- **Cosine** (`diffusion/scheduler=cosine`): Cosine schedule from Improved Diffusion

### 2. Model Parameterizations
- **Direct Mean ($\mu$)** (`diffusion/mean_strategy=direct`): Predict the denoised image at t-1
- **XStart ($x_0$)** (`diffusion/mean_strategy=xstart`): Directly predict the original image
- **Epsilon ($$)** (`diffusion/mean_strategy=epsilon`): Predict the noise added at timestep t

### 3. Variance Strategies
- **Fixed Small** (`diffusion/variance_strategy=fixed_small`): (posterior variance)
- **Fixed Large** (`diffusion/variance_strategy=fixed_large`): (forward variance)
- **Direct** (`diffusion/variance_strategy=direct`): Model directly outputs variance
- **Direct Log** (`diffusion/variance_strategy=direct_log`): Model outputs log-variance
- **Trainable Range** (`diffusion/variance_strategy=trainable_range`): Learn interpolation between variance bounds.

**Note**: When using learned variance strategies, set `out_channels = in_channels × 2` (e.g., 6 for RGB images).

### 4. Loss Functions
- **VLB** (`diffusion/loss=vlb`): Variational lower bound
- **Weighted MSE** (`diffusion/loss=mse_mean_direct`, `mse_mean_xstart`, `mse_mean_epsilon`): SNR-weighted losses
- **Simple MSE** (`diffusion/loss=mse_mean_direct_simple`, `mse_mean_xstart_simple`, `mse_mean_epsilon_simple`): Simple, unweighted version of MSE.
- **Hybrid** (`diffusion/loss=hybrid`): $\text{MSE} + \lambda \text{VLB}$: Improved Diffusion, combines simple MSE for Mean and VLB for variance.

### 5. Samplers
- **DDPM** (`diffusion/sampler=ddpm`): Stochastic sampling following the original paper

**Files**: Core implementations in `src/diffusion/`: `schedulers.py`, `losses.py`, `means.py`, `variances.py`, `samplers.py`

## Configuration System

This repository uses **Hydra** for compositional configuration. All configs are in `configs/` organized by component:

```
configs/
├── train.yaml                    # Main training config
├── sample.yaml                   # Sampling config
├── trainer/                      # gpu, cpu, mps, ddp
├── data/                         # mnist, cifar10
├── logger/                       # wandb, tensorboard
└── diffusion/
    ├── scheduler/                # linear, cosine
    ├── mean_strategy/            # epsilon, xstart, direct
    ├── variance_strategy/        # fixed_small, fixed_large, direct, direct_log, trainable_range
    ├── loss/                     # vlb, mse_mean_{direct, xstart, epsilon}_{simple}
    ├── sampler/                  # ddpm
    └── model/                    # unet, small_unet
```

### Override Configs from CLI

```bash
# Change learning rate
uv run python src/diffusion/scripts/train.py optimizer.lr=1e-4

# Use different scheduler
uv run python src/diffusion/scripts/train.py diffusion/scheduler=cosine

# Combine multiple overrides
uv run python src/diffusion/scripts/train.py \
    diffusion/mean_strategy=xstart \
    diffusion/loss=mse_mean_xstart_simple \
    batch_size=64 \
    trainer.max_epochs=100
```

## Experiment Tracking

### Supported Loggers

- **Weights & Biases** (`logger=wandb`): Cloud-based tracking with team collaboration
- **TensorBoard** (`logger=tensorboard`): Local visualization with PyTorch integration

**Note**: Even without a logger, all checkpoints and configurations are automatically saved to Hydra's output directory in `logs/`. This ensures you never lose experimental results.

### Logged Metrics
- Training/validation loss
- VLB terms (per-timestep KL divergence)
- Generated samples (periodic, configurable via callbacks)
- Model parameters and gradients

## Mapping to Book Structure

The code is organized to mirror the mathematical progression in the book:

| Book Chapter | Code Modules | Key Concepts |
|--------------|--------------|--------------|
| **Chapter 1** | `schedulers.py`, `diffusion_factors.py` | Forward process q(xₜ\|x₀), noise scheduling, reverse process pθ(x₀:T) |
| **Chapter 2** | `losses.py` (VLB), `gaussian_utils.py` | Variational inference, KL divergence, ELBO decomposition |
| **Chapter 3** | `means.py`, `variances.py` | ε-prediction vs x₀-prediction, learned variance |
| **Chapters 4-5** | `losses.py` (Hybrid), configs | Hybrid loss, experimental refinements |

**Example**: The `Hybrid` loss in `src/diffusion/losses.py:94` implements the training strategy from Improved Diffusion, combining the simple MSE objective with the VLB for learned variance.

## Repository Structure

```
src/diffusion/
├── gaussian_diffusion.py      # Main Lightning module
├── schedulers.py              # Noise schedules (linear, cosine)
├── losses.py                  # Loss functions (MSE, VLB, Hybrid)
├── means.py                   # Mean parameterizations (ε, x₀, μ)
├── variances.py               # Variance strategies
├── samplers.py                # DDPM sampling
├── diffusion_factors.py       # Precomputed α, β, γ terms
├── gaussian_utils.py          # KL divergence, log-likelihood
└── scripts/
    ├── train.py               # Training entry point
    ├── sample.py              # Sampling entry point
    └── fid.py                 # FID score computation
```

## Example Configurations

### DDPM (Ho et al., 2020)

```bash
uv run python src/diffusion/scripts/train.py \
    diffusion/scheduler=linear \
    diffusion/mean_strategy=epsilon \
    diffusion/variance_strategy=fixed_small \
    diffusion/loss=mse_mean_epsilon_simple \
    data=cifar10 \
    run_name="ddpm-baseline"
```

### Improved Diffusion (Nichol & Dhariwal, 2021)

```bash
uv run python src/diffusion/scripts/train.py \
    diffusion/scheduler=cosine \
    diffusion/mean_strategy=epsilon \
    diffusion/variance_strategy=trainable_range \
    diffusion/loss=hybrid \
    out_channels=6 \
    data=cifar10 \
    run_name="improved-diffusion"
```

### Fast Experimentation (MNIST)

```bash
uv run python src/diffusion/scripts/train.py \
    data=mnist \
    dim=28 \
    in_channels=1 \
    out_channels=1 \
    batch_size=128 \
    trainer=gpu \
    trainer.max_epochs=20 \
    diffusion/model=small_unet \
    run_name="mnist-quick"
```

## Testing

```bash
# Run all tests
uv run pytest

# Specific test modules
uv run pytest tests/test_losses.py -v
uv run pytest tests/test_schedulers.py -v
uv run pytest tests/test_samplers.py -v
```

## Key Implementation Details

### Diffusion Factors

Precomputed terms for efficient training (`src/diffusion/diffusion_factors.py`):

- **βₜ** (`betas`): Noise schedule
- **αₜ** (`alphas`): 1 - βₜ
- **γₜ** (`gammas`): Cumulative product ∏(1 - βᵢ) = ᾱₜ
- **γₜ₋₁** (`gammas_prev`): Previous timestep's cumulative product

### Hybrid Loss Strategy

From the Improved Diffusion paper (`src/diffusion/losses.py:94`):

1. Train mean prediction with simple MSE
2. Train variance with VLB (stop gradients on mean to prevent interference)
3. Combine: `mean_loss + ω * variance_loss` where ω = 0.001

This allows learning variance without destabilizing the mean prediction.

## Multi-GPU Training

```bash
# Distributed Data Parallel (recommended)
uv run python src/diffusion/scripts/train.py \
    trainer=ddp \
    trainer.devices=4 \
    batch_size=32  # per-GPU batch size
```

## Companion Book

This repository serves as the practical implementation for *"Foundations of Diffusion Models"* by Piotr Grzybowski.

The book emphasizes understanding *why* equations work, not just *how* to implement them. Every formula is derived from first principles with clear intuition and experimental validation.

**Book Structure**:
- **Chapter 1**: Mathematical framework of diffusion models
- **Chapter 2**: ELBO derivation and posterior distributions
- **Chapter 3**: Model parameterization and loss functions
- **Chapters 4-5**: Empirical experiments and progressive refinements

## Citation

If you use this code or book in your research, please cite:

```bibtex
@book{grzybowski2025diffusion,
  title={Foundations of Diffusion Models: From Theory to Practice},
  author={Grzybowski, Piotr},
  year={2025}
}
```

## Acknowledgments

- **DDPM**: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
- **Improved Diffusion**: Nichol, A. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models.
- U-Net architecture adapted from OpenAI's `improved-diffusion` repository.

## License

[Your license here]

---

**Built with ❤️ for the curious mind that values *why* as much as *how*.**
