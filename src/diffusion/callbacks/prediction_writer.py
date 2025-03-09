import math
import os
import shutil
import tempfile
from pathlib import Path

import torch.distributed
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torchvision.utils import make_grid

from diffusion.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


# https://github.com/Lightning-AI/pytorch-lightning/discussions/9259#discussioncomment-4441284
class PredictionWriter(BasePredictionWriter):
    def __init__(self, run_dir: str | Path) -> None:
        super().__init__("epoch")
        self.run_dir = Path(run_dir) / "predictions"
        self.temp_dir = ""
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if trainer.is_global_zero:
            temp_dir = [tempfile.mkdtemp()]
            log.info("Created temporary folder to store predictions: {}.".format(temp_dir[0]))
        else:
            temp_dir = [""]

        if torch.distributed.is_initialized() and torch.distributed.is_available():
            torch.distributed.broadcast_object_list(temp_dir)
            torch.distributed.barrier()

        self.temp_dir = temp_dir[0]
        for i, prediction in enumerate(predictions):
            torch.save(prediction, os.path.join(self.temp_dir, f"pred_{trainer.global_rank}_{i}.pt"))

    @rank_zero_only
    def on_predict_end(self, trainer, pl_module) -> None:
        self.gather()

    @rank_zero_only
    def gather(self):
        filenames = (filename for filename in os.listdir(self.temp_dir))
        tensors = [torch.load(os.path.join(self.temp_dir, filename)) for filename in filenames]

        result = torch.cat(tensors, dim=0)
        torch.save(result, os.path.join(self.run_dir, "result.pt"))
        log.info(f"Saved predictions to {os.path.join(self.run_dir, 'result.pt')}, shape {result.shape}.")

        result = make_grid(result, padding=0, nrow=int(math.sqrt(result.shape[0])))
        result = result.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        image = Image.fromarray(result)
        image.save(self.run_dir / "result.png")

        log.info("Cleanup temporary folder: {}.".format(self.temp_dir))
        shutil.rmtree(self.temp_dir)
