import math
from pathlib import Path

import hydra
import rootutils
import torch
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.utils.extras import extras
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.task_wrapper import task_wrapper
from lightning import seed_everything
from omegaconf import DictConfig
from torchvision.utils import save_image


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def sample(cfg: DictConfig):
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: GaussianDiffusion = hydra.utils.instantiate(cfg.diffusion)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    path = Path(cfg.paths.output_dir) / "sample.png"
    model.eval()
    result = model.sample((cfg.samples, cfg.in_channels, 28, 28))
    save_image(result, path, nrow=int(math.sqrt(result.shape[0])))

    # model.load_from_checkpoint(cfg.ckpt_path)
    #
    # log.info("Instantiating callbacks...")
    # callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    #
    # log.info("Instantiating loggers...")
    # logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    #
    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    #
    # path = Path(cfg.paths.output_dir) / "result.png"
    # log.info(f"Result path: {path}")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer)  # callbacks=callbacks, logger=logger)
    #
    # log.info("Sampling from model...")
    # result = typing.cast(list[torch.Tensor], trainer.predict(model, datamodule=datamodule, ckpt_path=cfg["ckpt_path"]))[0]
    #
    # log.info("Saving result...")
    # save_image(result, path, nrow=int(math.sqrt(result.shape[0])))
    #
    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }
    #
    # if logger:
    #     log.info("Logging hyperparameters!")
    #     log_hyperparameters(object_dict)
    #
    return None, None


@hydra.main(version_base="1.3", config_path=str(configs_path), config_name="sample.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: dictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = sample(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # return optimized metric
    return metric_dict


if __name__ == "__main__":
    main()
