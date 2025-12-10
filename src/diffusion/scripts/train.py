from pathlib import Path

import hydra
import rootutils
from lightning import Callback, LightningDataModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.utils.extras import extras
from diffusion.utils.instantiators import instantiate_callbacks, instantiate_loggers
from diffusion.utils.naming import auto_generate_names, generate_run_name_from_hydra, validate_naming_config
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.run_utils import find_ckpt_path
from diffusion.utils.task_wrapper import task_wrapper


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


# Register OmegaConf resolver for auto-generating run names
if not OmegaConf.has_resolver("auto_run_name"):
    OmegaConf.register_new_resolver("auto_run_name", generate_run_name_from_hydra)


log = RankedLogger(__name__, rank_zero_only=True)


def resolve_config(cfg: DictConfig) -> DictConfig:
    """Resolve configuration before training starts.

    This function:
    1. Auto-generates run_name and task_name if not provided
    2. Instantiates components to determine model dimensions
    3. Sets derived config values (channels, dimensions)
    """
    # Auto-generate names FIRST (before any path-dependent operations)
    cfg = auto_generate_names(cfg)
    validate_naming_config(cfg)

    # Now instantiate components (these may create directories)
    datamodule = hydra.utils.instantiate(cfg.data)
    variance = hydra.utils.instantiate(cfg.diffusion.variance_strategy)

    # Set derived dimensions
    cfg.in_channels = datamodule.channels
    cfg.out_channels = cfg.in_channels * 2 if variance.trainable else cfg.in_channels
    cfg.dim = datamodule.shape[-1]

    return cfg


@task_wrapper
def train(cfg: DictConfig):
    print(cfg.run_name)
    print(cfg.task_name)
    cfg = resolve_config(cfg)

    log.info("Serialize config")
    OmegaConf.save(cfg, Path(cfg.paths.output_dir) / "config.yaml")

    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.diffusion._target_}>")
    model: GaussianDiffusion = hydra.utils.instantiate(cfg.diffusion)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log.info("Finding checkpoint path...")
    ckpt_path = find_ckpt_path(cfg)

    for logger in loggers:
        logger.log_hyperparams(dict(model.hparams))

    if cfg.train:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.validate:
        log.info("Starting validation!")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


@hydra.main(version_base="1.3", config_path=str(configs_path), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    def a():
        extras(cfg)
        train(cfg)

    extras(cfg)
    train(cfg)


WandbLogger

if __name__ == "__main__":
    main()
