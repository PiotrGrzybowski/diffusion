import math
from pathlib import Path

import hydra
import rootutils
import torch
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.utils.extras import extras
from diffusion.utils.ranked_logger import RankedLogger
from lightning import seed_everything
from omegaconf import DictConfig
from PIL import Image
from torchvision.utils import make_grid


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


def sample(cfg: DictConfig):
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: GaussianDiffusion = hydra.utils.instantiate(cfg.diffusion)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer = hydra.utils.instantiate(cfg.trainer)

    model.to(trainer.strategy.root_device)
    path = Path(cfg.paths.output_dir) / "sample.png"
    model.eval()
    result = model.sample((cfg.samples, cfg.in_channels, cfg.dim, cfg.dim), verbose=True)

    grid = make_grid(result, nrow=int(math.sqrt(result.shape[0])))
    grid = grid.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    image = Image.fromarray(grid)
    image.save(path)


@hydra.main(version_base="1.3", config_path=str(configs_path), config_name="sample.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)

    sample(cfg)


if __name__ == "__main__":
    main()
