# Diffusion Models: From Theory to Practice

A PyTorch Lightning reimplementation of **DDPM** and **Improved Diffusion** papers, serving as the practical companion to the book *"Foundations of Diffusion Models"*.

## Overview

This repository bridges the gap between mathematical theory and practical implementation of diffusion models. Every component from noise schedulers to loss functions maps directly to the mathematical foundations presented in the accompanying book, making it an ideal resource for both learning and experimentation.

### What This Repository Offers
- **Principled Implementation**: Code structure mirrors mathematical derivations
- **Flexible Configuration**: Hydra-based compositional configs for rapid experimentation
- **Educational Focus**: Direct mapping between code modules and book chapters
- **Production-Ready**: Experiment tracking, checkpointing, and distributed training out-of-the-box

### Papers Implemented
- **[DDPM]** Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **[Improved Diffusion]** Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)

## Installation
```bash
git clone https://github.com/PiotrGrzybowski/diffusion.git
cd diffusion
uv sync
```

## Quick Start

### Training Your First Model

Here's a complete example that trains a diffusion model on MNIST with the `epsilon` mean parametrization, `fixed_small` variance with `simple MSE` loss. It is enough to run if for `10` epochs to see initial results. Approximatelly 20 minutes on a single GPU.

```bash
./scripts/mnist-epsilon-fixed_small-mse_simple.sh
```

### Sampling from the Trained Model
After training you should observe the checkpoints and logs in the `logs/` directory. You can sample from the trained model using the `ls` command. The `sample.py` script will automatically locate the entire configuration and checkpoint based on the `task_name` and `run_name` used during training. In our case `mnist` and `epsilon-fixed_small-mse_simple`.

```bash
uv run python src/diffusion/scripts/sample.py \
    task_name="mnist" \
    run_name="epsilon-fixed_small-mse_simple" \
    samples=16 \
    show=True
```

Progressive denoising steps will be displayed in a pop-up window. Final images will be saved in the `logs/mnist/epsilon-fixed_small-mse_simple/samples/` directory.

## Core Components
The implementation is organized around five key architectural decisions that you can mix and match:

### 1. Noise Schedulers
- **Linear** (`diffusion/scheduler=linear`): Linear $\beta$ schedule from DDPM paper
- **Cosine** (`diffusion/scheduler=cosine`): Cosine schedule from Improved Diffusion

### 2. Mean Parameterizations
Select the mean parameterization by setting `diffusion/mean_strategy={option}` to one of the following:
- `direct`: Direct Mean predicts $\mu_t$ directly. 
- `xstart`: XStart predicts the original image $x_0$.
- `epsilon`: Epsilon predicts the noise $\epsilon$ added at timestep t.

### 3. Variance Strategies
Choose the variance strategy by setting `diffusion/variance_strategy={option}` to one of the following:
- `direct`: Direct Variance predicts variance directly.
- `direct_log`: Direct Log Variance predicts log-variance.
- `fixed_small`: Fixed Small uses the posterior variance $\hat{\sigma}^2_t$
- `fixed_large`: Fixed Large uses the forward variance $\beta_t$
- `trainable_range`: Trainable Range learns to interpolate between fixed small and large variances.

### 4. Loss Functions
Select the loss function by setting `diffusion/loss={option}` to one of the following:
- `vlb`: Variational Lower Bound loss
- `mse_mean_{direct,xstart,epsilon}`: Weighted MSE losses for different mean strategies
- `mse_mean_{direct,xstart,epsilon}_simple`: Simple unweighted MSE losses
- `hybrid`: Hybrid loss combining MSE and VLB

**Files**: Core implementations in `src/diffusion/`: `schedulers.py`, `losses.py`, `means.py`, `variances.py`

## Configuration System
To effectively track and organize experiments, each run requires the following parameters:
- `task_name`: Group of experiments, we typically use the dataset name here (e.g., `cifar10`, `mnist`).
- `run_name`: Unique experiment identifier, we follow a convention of `{mean_strategy}-{variance_strategy}-{loss}`.

Specify the necessary dataset configuration:
- `data`: Dataset to use (e.g., `cifar10`, `mnist`)
- `batch_size`: Batch size for training and validation
- `dim`: Image dimension (e.g., `32` for CIFAR-10, `28` for MNIST)
- `in_channels`: Number of input channels (e.g., `3` for RGB, `1` for grayscale)
- `out_channels`: Number of output channels (e.g., for RGB `3` for fixed variance, `6` for learned variance)

Diffusion related parameters:
- `timesteps`: Number of diffusion steps (e.g., `1000`)
- `predict_samples`: Number of samples to generate during evaluation

Trainer and acceleration settings:
- `trainer`: Training backend (e.g., `gpu`, `ddp`, `cpu`)
- `trainer.max_epochs`: Maximum number of training epochs
- `trainer.devices`: Number of devices to use (e.g., `1` for single GPU, `4` for 4 GPUs in DDP)

Logger and Callbacks:
- `logger`: Logger or multiple loggers to use (e.g., `wandb`, [`wandb`, `tensorboard`])
- `callbacks`: We recommend to use the default callbacks provided in the config. 
- For longer training consider increasing the `callbacks.image_generation.every_n_epochs` from `5` to larger value as image sampling is time consuming and may overwhelm the training time.


All configs are in `configs/` organized by component:

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
    ├── loss/                     # vlb, mse_mean_{direct, xstart, epsilon}_{simple}, hybrid
    └── model/                    # unet, small_unet
```

### Override Configs from CLI
You can easily override any configuration parameter directly from the command line. Here is the example of the quick start training script with available configuration options:

```bash
uv run python src/diffusion/scripts/train.py \
    trainer=ddp \
    trainer.max_epochs=20 \
    diffusion/model=unet \
    diffusion/mean_strategy=epsilon \
    diffusion/variance_strategy=fixed_small \
    diffusion/loss=mse_mean_epsilon_simple \
    logger=wandb \
    data=mnist \
    task_name=mnist \
    batch_size=128 \
    run_name="epsilon-fixed_small-mse"
```

## Experiment Tracking

### Supported Loggers

- **Weights & Biases** (`logger=wandb`): Cloud-based tracking with team collaboration
- **TensorBoard** (`logger=tensorboard`): Local visualization with PyTorch integration

**Note**: Even without a logger, all checkpoints and configurations are automatically saved to Hydra's output directory in `logs/`. This ensures you never lose experimental results.

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

## Testing

```bash
# Run all tests
uv run pytest

# Specific test modules
uv run pytest tests/test_losses.py
uv run pytest tests/test_schedulers.py
uv run pytest tests/test_samplers.py
```

## Multi-GPU Training

```bash
# Distributed Data Parallel (recommended)
uv run python src/diffusion/scripts/train.py \
    trainer=ddp \
    trainer.devices=4 \
    batch_size=32  # per-GPU batch size
```


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

