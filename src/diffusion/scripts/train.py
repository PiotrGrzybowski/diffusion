import hydra
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from diffusion.utils.extras import extras
from diffusion.utils.instantiators import instantiate_callbacks, instantiate_loggers
from diffusion.utils.logging_utils import log_hyperparameters
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.run_utils import custom_main, find_ckpt_path
from diffusion.utils.task_wrapper import task_wrapper


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig):
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.diffusion._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.diffusion)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    ckpt_path = find_ckpt_path(cfg)

    object_dict = {
        "diffusion": model,
        "trainer": trainer,
        "loss": model.loss.__class__.__name__,
        "mean": model.model_mean.__class__.__name__,
        "variance": model.model_variance.__class__.__name__,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.train:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.test:
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.predict:
        log.info("Starting prediction!")
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
