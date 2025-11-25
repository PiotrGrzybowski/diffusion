# Diffusion Models: From Theory to Practice

Practical companion repository to the paper  **"Foundations of Diffusion Models" (Grzybowski, 2025).**

This project implements diffusion models **exactly** as derived in the paper, following the mathematical identities, factorizations, and objectives. The repository is designed for researchers and students who want to **understand diffusion models from first principles**, not just run an existing implementation.

## Purpose of This Repository

The goal of this codebase is to provide a clean, modular, and mathematically faithful implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) among with a series of improvement from [Improved Diffusion](https://arxiv.org/pdf/2102.09672)

Every component namely, nose schedulers, mean and variance strategies, samplers, objectives corresponds one-to-one to sections of the paper. The code does *not* re-explain theory; instead, it realizes the formulas exactly as presented.

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

### Training Your First Model

To train your first diffusion model, run the MNIST example below. This configuration uses the `epsilon` mean parameterization, the `fixed_small` variance, and the simple MSE objective. Training for approximately 10 epochs is sufficient to obtain initial results (around 20 minutes on a single GPU).

```bash
./scripts/mnist-epsilon-fixed_small-mse_simple.sh
```

After training completes, checkpoints, logs, and validation samples will be stored under:

```
logs/
└── mnist/
    └── hydra/
        └── epsilon-fixed_small-mse_simple/
            ├── checkpoints/
            │   ├── epoch_009.ckpt
            │   └── last.ckpt
            ├── config.yaml
            ├── config_tree.log
            ├── images/
            │   ├── sample_4.png
            │   └── sample_9.png
            └── mnist.log
```

When you later run the sampling script, an additional `samples/` directory will appear next to `images/`, containing generated outputs.

### Sampling from the Trained Model

The `sample.py` script reconstructs the full training configuration and automatically locates the corresponding checkpoint using the `task_name` and `run_name`. For the quick start example, set these to `mnist` and `epsilon-fixed_small-mse_simple`:

```bash
uv run python src/diffusion/scripts/sample.py \
    task_name="mnist" \
    run_name="epsilon-fixed_small-mse_simple" \
    samples=16 \
    show=True
```

During sampling, the progressive denoising steps will be displayed in a pop-up window. Final generated images will be written to: `logs/mnist/hydra/epsilon_fixed_small-mse_simple/samples`.


## Core Components
The implementation is organized around five key architectural decisions that you can mix and match:

#### 1. Noise Scheduler
Choose the noise scheduler by setting `diffusion/scheduler={option}` to one of the following:
- **Linear** (`diffusion/scheduler=linear`): Linear scheduler
- **Cosine** (`diffusion/scheduler=cosine`): Cosine scheduler

#### 2. Mean Parameterization
Choose the mean parameterization by setting `diffusion/mean_strategy={option}` to one of the following:
- `direct`: Predicts mean $\mu_t$ directly. 
- `xstart`: Predicts original image $x_0$.
- `epsilon`: Predicts noise $\epsilon_t$ added to the image.

#### 3. Variance Parameterization
Choose the variance parametrization by setting `diffusion/variance_strategy={option}` to one of the following:
- `direct`:  Predicts variance $\sigma_t^2$ directly.
- `direct_log`: Predicts log-variance $\log \sigma_t^2$
- `fixed_small`: Uses posterior variance $\hat{\sigma}^2_t$
- `fixed_large`: Uses forward variance $\beta_t$
- `trainable_range`: Predicts interpolation between `fixed_small` and `fixed_large` variances.

#### 4. Loss Functions
Select the loss function by setting `diffusion/loss={option}` to one of the following:
- `vlb`: Variational Lower Bound loss
- `mse_mean_{direct,xstart,epsilon}`: Weighted MSE losses for different mean strategies
- `mse_mean_{direct,xstart,epsilon}_simple`: Simple unweighted MSE losses
- `hybrid`: Hybrid loss combining MSE and VLB

**Files**: Core implementations in `src/diffusion/`, `schedulers.py`, `losses.py`, `means.py`, `variances.py`

## Configuration System
To effectively track and organize experiments, each run requires the following parameters:
- `task_name`: Group of experiments, we typically use the dataset name here (e.g., `cifar10`, `mnist`).
- `run_name`: Unique experiment identifier, we follow a convention of `{mean_strategy}-{variance_strategy}-{loss}`.

Specify the necessary dataset configuration:
- `data`: Dataset to use (e.g., `cifar10`, `mnist`)
- `batch_size`: Batch size for training and validation

Diffusion related parameters:
- `timesteps`: Number of diffusion steps (e.g., `1000`)
- `predict_samples`: Number of samples to generate duringa sampling

Trainer and acceleration settings:
- `trainer`: Training backend (e.g., `gpu`, `ddp`, `cpu`)
- `trainer.devices`: Number of devices to use (e.g., `1` for single GPU, `4` for 4 GPUs in DDP)
- `trainer.max_epochs`: Maximum number of training epochs

Logger and Callbacks:
- `logger`: Logger or multiple loggers to use (e.g., `wandb`, `tensorboard`)
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

## Testing

```bash
# Run all tests
uv run pytest

# Specific test modules
uv run pytest tests/test_losses.py
uv run pytest tests/test_schedulers.py
uv run pytest tests/test_samplers.py
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

## Advanced Dataset Configurations
Datamodules are prepared in such a way that you can easily select a subset of categories and number of samples. For example in MNIST you can select only digits `2` and `1` and limit the number of samples to `1000` by adding the following parameters to your config:

```yaml
data.labels: [2, 7]
data.train_samples_per_label: 1000
```

## License

[Your license here]

