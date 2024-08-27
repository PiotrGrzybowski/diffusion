import math
from pathlib import Path

from lightning import Callback
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torchvision.utils import save_image


class ImageGenerationCallback(Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 10 == 0:
            dataloader = trainer.datamodule.predict_dataloader()
            pl_module.eval()
            result = pl_module.predict_step(next(iter(dataloader)).to(pl_module.device))
            path = self.output_dir / "images"
            path.mkdir(exist_ok=True)
            save_image(result, path / f"sample_{trainer.current_epoch}.png", nrow=int(math.sqrt(result.shape[0])))
