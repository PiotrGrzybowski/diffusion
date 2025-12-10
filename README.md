# Diffusion Models: From Theory to Practice

Practical companion repository to the paper  **"Foundations of Diffusion Models" (Grzybowski, 2025).**

This project implements diffusion models **exactly** as derived in the paper, following the mathematical identities, factorizations, and objectives. The repository is designed for researchers and students who want to **understand diffusion models from first principles**, not just run an existing implementation.

## Purpose of This Repository

The goal of this codebase is to provide a clean, modular, and mathematically faithful implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) along with a series of improvements from [Improved Diffusion](https://arxiv.org/pdf/2102.09672)

Every component—noise schedulers, mean and variance strategies, samplers, objectives—corresponds one-to-one to sections of the paper. The code does *not* re-explain theory; instead, it realizes the formulas exactly as presented.

> 📝 **Start with the paper.**
> The repository assumes familiarity with the derivations and terminology introduced there.

## What This Repository Offers

- **Mathematical clarity** - code mirrors the notation, factors, and equations used in the paper.
- **Modularity** - interchangeable schedulers, mean strategies, variance strategies, and loss functions.
- **Hydra-driven experiments** - clean experiment reproducibility with compositional configs.
- **Research-first architecture** - minimal abstractions, maximal transparency.

## Installation
```bash
git clone https://github.com/PiotrGrzybowski/diffusion.git
cd diffusion
uv sync
```

## Training Your First Model
To train your first diffusion model, run the MNIST example below. By default the `tensorboard` logger is enabled.

```bash
uv run train experiment=quick_start
```

For multi-GPU training, add the `trainer=ddp` flag and specify the number of devices:

```bash
uv run train experiment=quick_start trainer=ddp trainer.devices=4
```

For `wandb` logging, login first using `wandb login`, then use the `logger=wandb` flag:

```bash
uv run train experiment=quick_start logger=wandb
```

This configuration uses the `epsilon` mean strategy, the `fixed_small` variance strategy, and the `mse_mean_epsilon_simple` loss function. Training for approximately 10 epochs is sufficient to obtain initial results (around 20 minutes on a single GPU).

After training completes, checkpoints, logs, and validation samples will be stored under:

```
logs/
└── mnist/
    ├── hydra/
    │   └── unet-epsilon-fixed_small-mse_epsilon_simple/
    │       ├── checkpoints/
    │       │   ├── epoch_009.ckpt
    │       │   └── last.ckpt
    │       ├── config.yaml
    │       ├── config_tree.log
    │       ├── images/
    │       │   ├── sample_4.png
    │       │   └── sample_9.png
    │       └── mnist.log
    └── tensorboard/
        └── unet-epsilon-fixed_small-mse_epsilon_simple/
```

When you later run the sampling script, an additional `samples/` directory will appear next to `images/`, containing generated outputs.

### Sampling from the Trained Model
The `sample.py` script reconstructs the full training configuration and automatically locates the corresponding checkpoint using the `task_name` and `run_name`. For the quick start example, set these to `mnist` and `unet-epsilon-fixed_small-mse_epsilon_simple`:

```bash
uv run sample task_name="mnist" run_name="unet-epsilon-fixed_small-mse_epsilon_simple" samples=16 show=True
```

During sampling, the progressive denoising steps will be displayed in a pop-up window. Final generated images will be written to: `logs/mnist/hydra/unet-epsilon-fixed_small-mse_epsilon_simple/samples`. Images are also available in the configured logger (e.g., TensorBoard).

## Core Components
The implementation is organized around four key architectural decisions that you can mix and match:

#### 1. Mean Strategy
Choose the mean strategy by setting `diffusion/mean_strategy={option}`:
- `direct`: Predicts mean $\mu_t$ directly
- `xstart`: Predicts original image $x_0$
- `epsilon`: Predicts noise $\epsilon_t$ added to the image

#### 2. Variance Strategy
Choose the variance strategy by setting `diffusion/variance_strategy={option}`:
- `direct`: Predicts variance $\sigma_t^2$ directly
- `direct_log`: Predicts log-variance $\log \sigma_t^2$
- `fixed_small`: Uses posterior variance $\hat{\sigma}^2_t$
- `fixed_large`: Uses forward variance $\beta_t$
- `trainable_range`: Predicts interpolation between `fixed_small` and `fixed_large` variances

#### 3. Loss Function
Select the loss function by setting `diffusion/loss={option}`:
- `vlb`: Variational Lower Bound loss
- `mse_{direct,xstart,epsilon}`: Weighted MSE losses for different mean strategies
- `mse_{direct,xstart,epsilon}_simple`: Simple unweighted MSE losses
- `hybrid`: Hybrid loss combining MSE and VLB

#### 4. Image Sampler
Select the sampling strategy by setting `diffusion/image_sampler={option}`:
- `ddpm`: Denoising Diffusion Probabilistic Model (ancestral sampling)
- `ddim`: Denoising Diffusion Implicit Models (deterministic sampling)

#### 5. Timestep Sampler
- `uniform`: Uniformly samples timesteps from 1 to T (default)
- `tbd`

#### 6. Noise Scheduler
Choose the noise scheduler by setting `diffusion/scheduler={option}`:
- `linear`: Linear noise schedule
- `cosine`: **TODO** Cosine noise schedule
**Files**: Core implementations in `src/diffusion/`, `schedulers.py`, `losses.py`, `means.py`, `variances.py`, `images_samplers.py`, `timestep_samplers.py`

## Configuration System
Specify the necessary dataset configuration:
- `data`: Dataset to use (e.g., `cifar10`, `mnist`, `fasion`, `kmnist`, `cifar100`)
- `batch_size`: Batch size for training and validation

Diffusion related parameters:
- `timesteps`: Number of diffusion steps (e.g., `1000`)
- `predict_samples`: Number of samples to generate during sampling (by default same as `timesteps`)

Trainer and acceleration settings:
- `trainer`: Training backend - available options:
  - `gpu`: Single GPU training
  - `cpu`: CPU training
  - `ddp`: Distributed Data Parallel (multi-GPU)
- `trainer.devices`: Number of devices to use (e.g., `1` for single GPU, `4` for 4 GPUs in DDP)
- `trainer.max_epochs`: Maximum number of training epochs

Logger options:
- `logger`: Logger or multiple loggers to use. Available options:
  - `wandb`: Weights & Biases
  - `tensorboard`: TensorBoard

Callbacks:
- `callbacks`: Training callbacks for monitoring and checkpointing
  - `model_checkpoint`: Save best models based on metrics (monitor, save_top_k)
  - `early_stopping`: Stop training when metric stops improving (monitor, patience)
  - `image_generation`: Generate sample images during training (every_n_epochs)
  - `rich_progress_bar`: Enhanced terminal progress display
  - For longer training, increase `callbacks.image_generation.every_n_epochs` from `10` to a larger value as image sampling is time-consuming

To effectively track and organize experiments, each run is identified by two key parameters:
- `task_name`: Group of experiments (auto-generated from dataset name, e.g., `mnist`, `cifar10`)
- `run_name`: Unique experiment identifier (auto-generated as `{model}-{mean}-{variance}-{loss}`)
- Example: `unet-epsilon-fixed_small-mse_epsilon_simple`
- Both names can be explicitly set via CLI or config files if desired

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

The default training configuration is defined in `configs/train.yaml`. You can use it directly and override parameters as needed:

```bash
# Use default train.yaml and adjust epochs and batch size
uv run train \
    trainer=gpu \
    trainer.max_epochs=20 \
    batch_size=64 \
    predict_samples=16 \
    data=mnist \
    diffusion/model=unet \
    diffusion/mean_strategy=epsilon \
    diffusion/variance_strategy=fixed_small \
    diffusion/loss=mse_epsilon_simple \
    diffusion/scheduler=linear \
    logger=tensorboard
```

This is equivalent to the `quick_start` experiment. Now you can override specific components:

```bash
# Change loss function to VLB:
uv run train ... diffusion/loss=vlb

# Use trainable variance:
uv run train ... diffusion/variance_strategy=direct_log

# Switch to CIFAR-10:
uv run train ... data=cifar10
```

## Dataset Configuration
### Supported Datasets

The repository supports multiple datasets:
- **MNIST**: Handwritten digits (28x28, grayscale) - `data=mnist`
- **CIFAR-10**: Natural images (32x32, RGB) - `data=cifar10`
- **KMNIST**: Kuzushiji-MNIST (28x28, grayscale) - `data=mnist dataset_name=kmnist`
- **FashionMNIST**: Fashion items (28x28, grayscale) - `data=mnist dataset_name=fashion`
- **CIFAR-100**: Natural images with 100 classes (32x32, RGB) - `data=cifar100`

### Advanced Dataset Options

Filter specific classes and limit samples per label:

```yaml
data.labels: [2, 7]                    # Select only specific labels
data.train_samples_per_label: 1000    # Limit training samples per label
data.val_samples_per_label: 10        # Limit validation samples per label
```

Example - Train on only MNIST digits 2 and 7 with 1000 samples each:
```bash
uv run train experiment=quick_start data.labels=[2,7] data.train_samples_per_label=1000
```

## Repository Structure

```
src/diffusion/
├── gaussian_diffusion.py        # Lightning module
├── schedulers.py                # Noise schedulers
├── losses.py                    # Loss functions
├── means.py                     # Mean parameterizations
├── variances.py                 # Variance parameterizations
├── samplers.py                  # Samplers
├── diffusion_factors.py         # Definition of alphas, betas, gammas
├── gaussian_utils.py            # KL divergence, log-likelihood
├── scripts/
│   ├── train.py                 # Training entry point
│   └── sample.py                # Sampling entry point
└── models/
    ├── attention.py             # Attention modules
    ├── normalization.py         # Normalization layers
    ├── time_embedding.py        # Sinusoidal time embeddings
    └── unet.py                  # U-Net architecture
```


### Code Quality

The project uses `ruff` for linting and formatting:

```bash
# Check code style
uv run ruff check .

# Auto-format code
uv run ruff format .

# Run tests
uv run pytest
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

## Model Zoo
Pretrained models will be made available soon.

## License

MIT License - See LICENSE file for details
