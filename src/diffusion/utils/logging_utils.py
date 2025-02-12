from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only

from diffusion.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    hparams = {}

    diffusion = object_dict["diffusion"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["diffusion"] = diffusion

    hparams["loss"] = object_dict["loss"]
    hparams["mean"] = object_dict["mean"]
    hparams["variance"] = object_dict["variance"]

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
