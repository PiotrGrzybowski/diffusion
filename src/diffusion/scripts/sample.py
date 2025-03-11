from pathlib import Path

import hydra
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

from diffusion.utils.extras import extras
from diffusion.utils.instantiators import instantiate_callbacks
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.run_utils import custom_main


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


def sample(sample_config: DictConfig):
    run_path = Path(sample_config.paths.output_dir)

    cfg = OmegaConf.load(run_path / "config.yaml")
    cfg.predict_samples = sample_config.predict_samples
    cfg.trainer = sample_config.trainer
    cfg.sample_timesteps = sample_config.sample_timesteps

    cfg.data.batch_size = sample_config.batch_size
    cfg.data.predict_samples = sample_config.predict_samples

    sample_config.in_channels = cfg.in_channels
    sample_config.dim = cfg.dim

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.diffusion._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.diffusion)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(sample_config.callbacks)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    ckpt_path = run_path / "checkpoints" / sample_config.ckpt_name

    log.info("Starting prediction!")
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="sample.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    sample(cfg)


if __name__ == "__main__":
    main()
