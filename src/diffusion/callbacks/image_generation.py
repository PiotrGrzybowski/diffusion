import math
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torchvision.utils import save_image


class ImageGenerationCallback(Callback):
    def __init__(self, shape: list[int], output_dir: Path | str, every_n_epochs: int = 1) -> None:
        super().__init__()
        self.shape = list(shape)
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            pl_module.eval()
            x = torch.randn(self.shape, device=pl_module.device)
            result = pl_module(x)
            path = self.output_dir / "images"
            path.mkdir(exist_ok=True)
            save_image(result, path / f"sample_{trainer.current_epoch}.png", nrow=int(math.sqrt(result.shape[0])))
