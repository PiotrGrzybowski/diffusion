from pathlib import Path

import hydra
import rootutils
import torch
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

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

    fabric = Fabric(devices=2, accelerator=cfg.accelerator)
    fabric.launch()

    log.info(f"Instantiating model <{module_cfg.diffusion._target_}>")
    model = hydra.utils.instantiate(module_cfg.diffusion)

    log.info("Instantiating loggers...")
    loggers = instantiate_loggers(cfg.get("logger"))

    ckpt_path = run_path / "checkpoints" / cfg.ckpt_name
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict)
    module = fabric.setup_module(model)

    samples = 64
    batch_size = 4
    batches = samples // batch_size
    timesteps = cfg.sample_timesteps

    columns = (
        TextColumn("[bold]{task.fields[title]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("• {task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with Progress(*columns) as progress:
        overall = progress.add_task("", total=batches * timesteps, title="Overall")
        batch_bar = progress.add_task("", total=batches, title="Batch")
        step_bar = progress.add_task("", total=timesteps, title="Timesteps (Batch 1)")

        for i in range(batches):
            progress.update(batch_bar, advance=1)
            progress.reset(step_bar, total=timesteps)
            progress.update(step_bar, title=f"Timesteps (Batch {i + 1}/{batches})")

            batch = torch.randn(batch_size, module_cfg.in_channels, module_cfg.dim, module_cfg.dim, device=fabric.device)

            for x_t in module.sample(batch, cfg.sample_timesteps):
                progress.update(step_bar, advance=1)
                progress.advance(overall)
            x = ((x_t + 1) * 127.5).clamp(0, 255).to(device=fabric.device, dtype=torch.uint8)

    # sample_path = run_path / "samples"
    # sample_path.mkdir(exist_ok=True)
    # cv2.imwrite(str(sample_path / f"sample_{cfg.ckpt_name}.png"), result)
    # torch.save(x_t, sample_path / f"sample_{cfg.ckpt_name}.pt")


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="sample.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    sample(cfg)


if __name__ == "__main__":
    main()
