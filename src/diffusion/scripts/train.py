from pathlib import Path

import hydra
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from diffusion.utils.extras import extras
from diffusion.utils.instantiators import instantiate_callbacks, instantiate_loggers
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.run_utils import custom_main, find_ckpt_path
from diffusion.utils.task_wrapper import task_wrapper


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig):
    log.info("Serialize config")
    OmegaConf.save(cfg, Path(cfg.paths.output_dir) / "config.yaml")

    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.diffusion._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.diffusion)

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


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
