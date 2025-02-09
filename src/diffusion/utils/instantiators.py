import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from diffusion.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_config: DictConfig) -> list[Callback]:
    callbacks: list[Callback] = []

    if not callbacks_config:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, config in callbacks_config.items():
        if isinstance(config, DictConfig) and "_target_" in config:
            log.info(f"Instantiating callback <{config._target_}>")
            callbacks.append(hydra.utils.instantiate(config))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
