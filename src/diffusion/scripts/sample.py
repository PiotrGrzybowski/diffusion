import math
from pathlib import Path

import hydra
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from PIL import Image
from torchvision.utils import make_grid

from diffusion.utils.extras import extras
from diffusion.utils.instantiators import instantiate_callbacks, instantiate_loggers
from diffusion.utils.logging_utils import log_hyperparameters
from diffusion.utils.ranked_logger import RankedLogger


a = (math, Path, hydra, rootutils, torch, seed_everything, DictConfig, Image, make_grid, RankedLogger, extras)

root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


def sample(cfg: DictConfig):
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

    log.info("Sampling images")
    state = torch.load("/home/alphabrain/Workspace/Projects/diffusion/src/diffusion/scripts/last.ckpt")
    model.load_state_dict(state["state_dict"])
    model.eval()
    trainer.predict(
        model,
        datamodule=datamodule,  # , ckpt_path="/home/alphabrain/Workspace/Projects/diffusion/src/diffusion/scripts/last.ckpt"
    )  # cfg.ckpt_path)

    # model.load_state_dict(ckpt["state_dict"], strict=False)
    # trainer = hydra.utils.instantiate(cfg.trainer)
    #
    # model.to(trainer.strategy.root_device)
    # path = Path(cfg.paths.output_dir) / "sample.png"
    # model.eval()
    # result = model.sample((cfg.samples, cfg.in_channels, cfg.dim, cfg.dim), verbose=True)
    #
    # grid = make_grid(result, nrow=int(math.sqrt(result.shape[0])))
    # grid = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # image = Image.fromarray(grid)
    # image.save(path)


@hydra.main(version_base="1.3", config_path=str(configs_path), config_name="sample.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)

    sample(cfg)


if __name__ == "__main__":
    main()
