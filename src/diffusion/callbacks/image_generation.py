from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
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
            result = pl_module.sample(self.shape)
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

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_predict_epoch_end(trainer, pl_module)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs: torch.Tensor, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        print("hella")
        print(batch_idx)
        result = make_grid(outputs, padding=0, nrow=int(math.sqrt(outputs.shape[0])))
        result = result.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        image = Image.fromarray(result)

        path = self.output_dir / "predicts"
        path.mkdir(exist_ok=True)
        image.save(path / f"predict_{batch_idx}.png")

        if trainer.loggers:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image("predict", [image], trainer.current_epoch)
