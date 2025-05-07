from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from rich.console import Console
from torchvision.utils import make_grid, math


class ImageGenerationCallback(Callback):
    def __init__(self, samples: int, in_channels: int, dim: int, output_dir: Path | str, every_n_epochs: int = 1) -> None:
        super().__init__()
        self.shape = (samples, in_channels, dim, dim)
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            pl_module.eval()

            batch = torch.randn(self.shape, device=pl_module.device)
            timesteps = pl_module.sample_timesteps

            console = Console()
            for i, x_t in enumerate(pl_module.sample(batch, timesteps)):
                console.print(f"Predict sampling: {timesteps - i}/{timesteps}", end="\r")

            result = x_t.detach().cpu()

            path = self.output_dir / "images"
            path.mkdir(exist_ok=True)
            result = make_grid(result, padding=0, nrow=int(math.sqrt(result.shape[0])))
            result = result.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            image = Image.fromarray(result)
            image.save(path / f"sample_{trainer.current_epoch}.png")

            if trainer.loggers:
                for logger in trainer.loggers:
                    if isinstance(logger, WandbLogger):
                        logger.log_image("samples", [image], trainer.current_epoch)
