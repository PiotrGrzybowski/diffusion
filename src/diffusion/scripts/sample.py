import math
from pathlib import Path

import cv2
import hydra
import numpy as np
import rootutils
import torch
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from torchvision.utils import make_grid

from diffusion.utils.extras import extras
from diffusion.utils.instantiators import instantiate_loggers
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.run_utils import custom_main


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


def sample(cfg: DictConfig):
    run_path = Path(cfg.paths.output_dir)
    module_cfg = OmegaConf.load(run_path / "config.yaml")

    fabric = Fabric(devices=1, accelerator=cfg.accelerator)
    fabric.launch()

    log.info(f"Instantiating model <{module_cfg.diffusion._target_}>")
    model = hydra.utils.instantiate(module_cfg.diffusion)

    log.info("Instantiating loggers...")
    loggers = instantiate_loggers(cfg.get("logger"))

    ckpt_path = run_path / "checkpoints" / cfg.ckpt_name
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict)
    module = fabric.setup_module(model)

    batch = torch.randn(cfg.samples, module_cfg.in_channels, module_cfg.dim, module_cfg.dim, device=fabric.device)

    console = Console()

    x_t = torch.empty(0)
    result = np.empty(0)
    for timestep, x_t in enumerate(module.sample(batch, cfg.sample_timesteps)):
        console.print(f"Val sampling: {timestep}/{module.sample_timesteps}", end="\r")
        if cfg.show:
            result = make_grid(x_t, padding=0, nrow=int(math.sqrt(x_t.shape[0])))
            result = result.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            cv2.imshow("image", cv2.resize(result, (512, 512)))
            cv2.waitKey(1)

    for logger in loggers:
        if hasattr(logger, "log_image"):
            key = f"sample_{cfg.ckpt_name}"
            logger.log_image(key=key, images=[result], step=None)

    sample_path = run_path / "samples"
    sample_path.mkdir(exist_ok=True)
    cv2.imwrite(str(sample_path / f"sample_{cfg.ckpt_name}.png"), result)
    torch.save(x_t, sample_path / f"sample_{cfg.ckpt_name}.pt")


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="sample.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    sample(cfg)


if __name__ == "__main__":
    main()
