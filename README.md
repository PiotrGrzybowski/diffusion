# Diffusion Models: From Theory to Practice

Practical companion repository to the book **"Diffusion Models from First Principles" (Grzybowski, 2025)**.

Read the book online: **https://piotrgrzybowski.github.io/diffusion-foundations/**

This project implements diffusion models **exactly** as derived in the book, following the mathematical identities, factorizations, and objectives. The repository is designed for researchers and students who want to **understand diffusion models from first principles**, not just run an existing implementation.

## Purpose of This Repository

The goal of this codebase is to provide a clean, modular, and mathematically faithful implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) along with a series of improvements from [Improved Diffusion](https://arxiv.org/pdf/2102.09672)

Every component—noise schedulers, mean and variance strategies, samplers, objectives—corresponds one-to-one to sections of the book. The code does *not* re-explain theory; instead, it realizes the formulas exactly as presented.

> 📝 **Start with the book.**
> The repository assumes familiarity with the derivations and terminology introduced there.

## What This Repository Offers

- **Mathematical clarity** - code mirrors the notation, factors, and equations used in the book.
- **Modularity** - interchangeable schedulers, mean strategies, variance strategies, and loss functions.
- **Hydra-driven experiments** - clean experiment reproducibility with compositional configs.
- **Research-first architecture** - minimal abstractions, maximal transparency.

## Installation
```bash
git clone https://github.com/PiotrGrzybowski/diffusion.git
cd diffusion
uv venv -p python3.12
uv sync
source ./.venv/bin/activate
```

For development (tests, lint, coverage), install dev dependencies as well:

```bash
uv sync --dev
```

> **Note:** This project pins **PyTorch 2.3** to maintain backward compatibility with Pascal GPU architecture (e.g., GTX 1080, Tesla P100). If you are using a modern GPU (Turing, Ampere, or newer), feel free to upgrade PyTorch to a newer version.
>
> If dataset downloads fail due to local SSL certificate setup, you can use `DIFFUSION_INSECURE_SSL=1` as a temporary workaround while fixing certificates on your machine.

## Quick Start
### Sampling Using Pretrained Weights
If you want to skip training and jump straight to inference, download the pretrained checkpoint from the model zoo:

```bash
uv run zoo download cifar10 unet-epsilon-fixed_small-mse_epsilon_simple-linear
```

Then sample directly:

```bash
uv run sample task_name="zoo_cifar10" run_name="unet-epsilon-fixed_small-mse_epsilon_simple-linear" samples=16 show=True
```

See the [Model Zoo](#model-zoo) section for all 9 available pretrained CIFAR-10 models.

### Training
Train a diffusion model on CIFAR-10 or MNIST data with a predefined configurations:

```bash
uv run train experiment=quick_cifar
uv run train experiment=quick_mnist
```

For multi-GPU training, add the `trainer=ddp` flag and specify the number of devices:

```bash
uv run train experiment=quick_cifar trainer=ddp trainer.devices=4
```

For `wandb` logging, login first using `wandb login`, then use the `logger=wandb` flag:

```bash
uv run train experiment=quick_cifar logger=wandb
```

After training completes, checkpoints, logs, and validation samples will be stored under. In `experiment=quick_cifar`, the defaults are `task_name=quick_start` and `run_name=cifar10`:

```
logs/
└── quick_start/
    ├── hydra/
    │   └── cifar10/
    │       ├── checkpoints/
    │       │   ├── epoch_009.ckpt
    │       │   └── last.ckpt
    │       ├── config.yaml
    │       ├── config_tree.log
    │       ├── images/
    │       │   ├── sample_4.png
    │       │   └── sample_9.png
    │       └── cifar10.log
    └── tensorboard/
        └── cifar10/
```


### Sampling from the Trained Model

The `sample.py` script reconstructs the full training configuration and automatically locates the corresponding checkpoint using the `task_name` and `run_name`. For `experiment=quick_cifar`, use `task_name=quick_start` and `run_name=cifar10`:

```bash
uv run sample task_name="quick_start" run_name="cifar10" samples=16 show=True
```

During sampling, the progressive denoising steps will be displayed in a pop-up window. Final generated images will be written to: `logs/quick_start/hydra/cifar10/samples`. Images are also available in the configured logger (e.g., TensorBoard).

## Core Components
The implementation is organized around following key architectural decisions that you can mix and match:

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
- `mse_{mean,xstart,epsilon}`: Weighted MSE losses for different mean strategies
- `mse_{mean,xstart,epsilon}_simple`: Simple unweighted MSE losses
- `hybrid`: Hybrid loss combining MSE and VLB

#### 4. Image Sampler
Select the sampling strategy by setting `diffusion/image_sampler={option}`:
- `ddpm`: Denoising Diffusion Probabilistic Model (ancestral sampling)
- `ddim`: Denoising Diffusion Implicit Models (deterministic sampling)

#### 5. Timestep Sampler
Select the timestep sampling strategy by setting `diffusion/timestep_sampler={option}`:
- `uniform`: Uniformly samples timesteps from 0 to T-1 (default)

#### 6. Noise Scheduler
Choose the noise scheduler by setting `diffusion/scheduler={option}`:
- `linear`: Linear noise schedule
- `cosine`: Cosine noise schedule from Improved Diffusion
**Files**: Core implementations in `src/diffusion/`: `schedulers.py`, `losses.py`, `means.py`, `variances.py`, `image_samplers.py`, `timestep_samplers.py`

## Configuration System
Specify the necessary dataset configuration:
- `data`: Dataset to use (e.g., `cifar10`, `mnist`, `fashion`, `kmnist`)
- `batch_size`: Batch size for training and validation

Diffusion related parameters:
- `timesteps`: Number of diffusion steps (e.g., `1000`)
- `predict_samples`: Number of samples to generate during validation-time sampling callbacks (default: `4`)

Trainer and acceleration settings:
- `trainer`: Training backend - available options:
  - `gpu`: Single GPU training
  - `cpu`: CPU training
  - `ddp`: Distributed Data Parallel (multi-GPU)
- `trainer.devices`: Number of devices to use (e.g., `1` for single GPU, `4` for 4 GPUs in DDP)
- `trainer.max_epochs`: Maximum number of training epochs

Optimizer settings:
- `optimizer`: Optimizer to use (`adam` or `adamw`)
- `optimizer.lr`: Learning rate (e.g., `1e-4`)
- `optimizer.weight_decay`: Weight decay (e.g., `1e-5`)

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
  - For longer training, increase `callbacks.image_generation.every_n_epochs` from `5` to a larger value as image sampling is time-consuming

To effectively track and organize experiments, each run is identified by two key parameters:
- `task_name`: Group of experiments (auto-generated from dataset name, e.g., `mnist`, `cifar10`)
- `run_name`: Unique experiment identifier (auto-generated as `{model}-{mean}-{variance}-{loss}-{scheduler}`)
- Example: `unet-epsilon-fixed_small-mse_epsilon_simple-linear`
- Both names can be explicitly set via CLI or config files if desired

All configs are in `configs/` organized by component:

```
configs/
├── train.yaml                    # Main training config
├── sample.yaml                   # Sampling config
├── callbacks/                    # model_checkpoint, early_stopping, image_generation, ...
├── data/                         # mnist, cifar10
├── experiment/                   # quick_cifar, quick_mnist
├── logger/                       # wandb, tensorboard, csv
├── optimizer/                    # adam, adamw
├── trainer/                      # gpu, cpu, mps, ddp
└── diffusion/
    ├── scheduler/                # linear, cosine
    ├── mean_strategy/            # epsilon, xstart, direct
    ├── variance_strategy/        # fixed_small, fixed_large, direct, direct_log, trainable_range
    ├── loss/                     # vlb, mse_{mean,xstart,epsilon}{,_simple}, hybrid
    ├── image_sampler/            # ddpm, ddim
    ├── timestep_sampler/         # uniform
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

You can also start from an experiment preset and override specific components:

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
- **CIFAR-100**: Natural images with 100 classes (32x32, RGB) - `data=cifar10 data.dataset_name=cifar100`

### Advanced Dataset Options

Filter specific classes and limit samples per label:

```yaml
data.labels: [2, 7]                    # Select only specific labels
data.train_samples_per_label: 1000    # Limit training samples per label
data.val_samples_per_label: 10        # Limit validation samples per label
```

Example - Train on only MNIST digits 2 and 7 with 1000 samples each:
```bash
uv run train experiment=quick_mnist data.labels=[2,7] data.train_samples_per_label=1000
```

## Repository Structure

```
src/diffusion/
├── gaussian_diffusion.py        # Lightning module
├── schedulers.py                # Noise schedulers
├── losses.py                    # Loss functions
├── means.py                     # Mean parameterizations
├── variances.py                 # Variance parameterizations
├── image_samplers.py            # Image samplers (DDPM, DDIM)
├── timestep_samplers.py         # Timestep samplers
├── diffusion_factors.py         # Definition of alphas, betas, gammas
├── diffusion_terms.py           # Diffusion term data structures
├── gaussian_utils.py            # KL divergence, log-likelihood
├── metrics.py                   # Evaluation metrics
├── callbacks/
│   ├── image_generation.py      # Generate samples during training
│   └── rich_bar.py              # Rich progress bar
├── data/
│   ├── mnist_datamodule.py      # MNIST/FashionMNIST/KMNIST data module
│   ├── cifar_datamodule.py      # CIFAR-10/CIFAR-100 data module
│   ├── filtered_mnist.py        # Class filtering and sampling
│   └── dataset_map.py           # Dataset mapping utilities
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── sample.py                # Sampling entry point
│   └── zoo.py                   # Model zoo download/manage
├── utils/                       # Hydra utils, naming, logging, etc.
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

If you only ran `uv sync` during installation, run `uv sync --dev` before the commands above.


## Citation
If you use this code or book in your research, please cite:

```bibtex
@book{grzybowski2025diffusion,
  title={Diffusion Models from First Principles},
  author={Grzybowski, Piotr},
  year={2025}
}
```

## Acknowledgments
- **DDPM**: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
- **Improved Diffusion**: Nichol, A. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models.
- U-Net architecture adapted from OpenAI's `improved-diffusion` repository.

## Model Zoo

9 pretrained CIFAR-10 models are available, covering all mean/variance/loss combinations discussed in the book. All files (configs, training samples, generated samples, and checkpoints) are hosted on [Hugging Face Hub](https://huggingface.co/PiotrGrzybowski/diffusion-model-zoo).

### Available Models

| Run Name | Mean | Variance | Loss | Scheduler |
|---|---|---|---|---|
| `unet-epsilon-fixed_small-mse_epsilon_simple-linear` | epsilon | fixed_small | mse_epsilon_simple | linear |
| `unet-epsilon-fixed_small-vlb-linear` | epsilon | fixed_small | vlb | linear |
| `unet-epsilon-direct_log-hybrid-linear` | epsilon | direct_log | hybrid | linear |
| `unet-epsilon-trainable_range-hybrid-linear` | epsilon | trainable_range | hybrid | linear |
| `unet-xstart-fixed_small-mse_xstart_simple-linear` | xstart | fixed_small | mse_xstart_simple | linear |
| `unet-xstart-fixed_small-vlb-linear` | xstart | fixed_small | vlb | linear |
| `unet-direct-fixed_small-vlb-linear` | direct | fixed_small | vlb | linear |
| `unet-direct-direct_log-vlb-linear` | direct | direct_log | vlb | linear |
| `unet-direct-direct-vlb-linear` | direct | direct | vlb | linear |

### Download Pretrained Models

Downloaded models are stored under `logs/zoo_{task}/` (e.g., `cifar10` → `logs/zoo_cifar10/`), keeping `logs/cifar10/` reserved for your own experiments:

```bash
# List available tasks
uv run zoo list

# List runs for a task (shows download status)
uv run zoo list cifar10

# Download a specific run
uv run zoo download cifar10 unet-epsilon-fixed_small-mse_epsilon_simple-linear

# Download all runs for a task
uv run zoo download cifar10

# Force re-download
uv run zoo download cifar10 unet-epsilon-fixed_small-vlb-linear --force

# Delete a downloaded run
uv run zoo delete cifar10 unet-epsilon-fixed_small-vlb-linear

# Delete all downloaded runs for a task
uv run zoo delete cifar10
```

### Sample from a Pretrained Model

After downloading, sample directly — the model appears as task `zoo_cifar10` in `logs/`:

```bash
uv run sample task_name="zoo_cifar10" run_name="unet-epsilon-fixed_small-mse_epsilon_simple-linear" \
    samples=16 show=True
```

## License

MIT License - See LICENSE file for details
