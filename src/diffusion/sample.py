import math
import typing
from pathlib import Path

import hydra
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torchvision.utils import save_image

from diffusion.utils.extras import extras
from diffusion.utils.instantiators import instantiate_callbacks, instantiate_loggers
from diffusion.utils.logging_utils import log_hyperparameters
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.task_wrapper import task_wrapper


rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)


log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, object], dict[str, object]]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    path = Path(cfg.paths.output_dir) / "result.png"
    log.info(f"Result path: {path}")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)  # callbacks=callbacks, logger=logger)

    log.info("Sampling from model...")
    result = typing.cast(list[torch.Tensor], trainer.predict(model, datamodule=datamodule, ckpt_path=cfg["ckpt_path"]))[0]

    log.info("Saving result...")
    save_image(result, path, nrow=int(math.sqrt(result.shape[0])))

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    return None, None


@hydra.main(version_base="1.3", config_path="configs", config_name="sample.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: dictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # return optimized metric
    return metric_dict


if __name__ == "__main__":
    main()
