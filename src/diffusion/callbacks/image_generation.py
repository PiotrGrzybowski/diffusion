from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from PIL import Image
from rich.console import Console
from torchvision.utils import make_grid, math


class ImageGenerationCallback(Callback):
    def __init__(self, samples: int, output_dir: Path | str, every_n_epochs: int = 1, verbose: bool = False) -> None:
        super().__init__()
        self.shape = (samples, 0, 0, 0)
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.verbose = verbose

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            pl_module.eval()

            batch = torch.randn(self.shape, device=pl_module.device)
            timesteps = pl_module.sample_timesteps

            console = Console(soft_wrap=True)
            for i, x_t in enumerate(pl_module.sample(batch, timesteps)):
                if self.verbose:
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
                    if isinstance(logger, TensorBoardLogger):
                        logger.experiment.add_image(
                            "samples",
                            torch.tensor(result).permute(2, 0, 1),
                            global_step=trainer.current_epoch,
                        )

    @rank_zero_only
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, _ = batch
        self.shape = (self.shape[0], x.shape[1], x.shape[2], x.shape[3])
