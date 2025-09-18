from pathlib import Path

import hydra
import numpy as np
import rootutils
import torch
import torchvision.transforms as transforms
from lightning import LightningDataModule
from omegaconf import DictConfig
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from diffusion.utils.extras import extras
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.run_utils import custom_main


transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.expand(-1, 3, -1, -1)),
        transforms.Resize((299, 299), antialias=True),
    ]
)

root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


def fid(cfg: DictConfig):
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")
    device = "cuda"
    run_path = cfg.paths.log_dir
    run_path = Path(cfg.paths.log_dir) / cfg.task_name
    fid_path = run_path / "fid.pt"

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()[0]
    # resize = transforms.Resize((299, 299), antialias=True)

    train_images = []
    print(fid_path)
    if fid_path.exists():
        log.info(f"Loading FID from {fid_path}")
        fid = torch.load(fid_path)
    else:
        log.info(f"FID not found at {fid_path}")
        fid = FrechetInceptionDistance(feature=2048, input_img_size=(3, 32, 32), reset_real_features=False, normalize=False).to(device)

        for batch in tqdm(train_loader):
            x, _ = batch
            x = ((x + 1) * 127.5).clamp(0, 255).to(device=device, dtype=torch.uint8)
            fid.update(x, real=True)
            torch.save(fid, fid_path)

    for batch in tqdm(datamodule.train_dataloader()):
        x, _ = batch
        x = ((x + 1) * 127.5).clamp(0, 255).to(device=device, dtype=torch.uint8)
        train_images.append(x.detach().cpu().numpy())
    train_images = np.concatenate(train_images, axis=0)
    np.save("train.npy", train_images)

    # for batch in tqdm(val_loader):
    #     x, _ = batch
    #     x = ((x + 1) * 127.5).clamp(0, 255).to(device=device, dtype=torch.uint8)
    #     fid.update(x, real=False)
    #     torch.save(fid, fid_path)
    # print(f"Final FID: {fid.compute()}")


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    fid(cfg)


if __name__ == "__main__":
    main()
