import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.expand(-1, 3, -1, -1)),  # Convert 1-channel to 3-channel RGB
        transforms.Resize((299, 299), antialias=True),
    ]
)

# run_path = "/home/alphabrain/Workspace/Projects/diffusion/new_logs/mnist_diffusion/hydra/icy-comet-33/predictions"
# image = cv2.imread(f"{run_path}/result.png")
# print(image.shape)
# print(image.max())
#
# batch = torch.load(f"{run_path}/result.pt")
# print(batch.shape)
#
# batch = transform(batch)
#
#
# device = "cuda:0"
# print(batch.shape)
# result = make_grid(
#     batch,
#     nrow=int(math.sqrt(batch.size(0))),
#     padding=0,
# )
# result = result.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
# image = Image.fromarray(result)
# image.save(Path(run_path) / "result_resized.png")
# # device = "cpu"
#
#
# for _ in tqdm(range(1)):
#     # generate two slightly overlapping image intensity distributions
#     imgs_dist1 = torch.randint(0, 127, (100, 3, 28, 28), dtype=torch.uint8, device=device)
#     imgs_dist2 = torch.randint(0, 40, (100, 3, 28, 28), dtype=torch.uint8, device=device)
#     fid.update(imgs_dist1, real=True)
#     fid.update(imgs_dist2, real=False)
# score = fid.compute()
# print(score)


import hydra
import rootutils
from lightning import LightningDataModule
from omegaconf import DictConfig

from diffusion.utils.extras import extras
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.run_utils import custom_main


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)

device = "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"


def fid(cfg: DictConfig):
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")
    print(datamodule)

    # datamodule.prepare_data()
    # datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.val_dataloader()

    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=False).to(device)

    for batch in tqdm(train_loader):
        x, _ = batch
        x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        print(x.shape)
        print(x.min(), x.max())

        fid.update(x, real=True)
        break

    # if (run_path / "fid.pt").exists():
    #     log.info(f"Loading FID from {run_path / 'fid.pt'}")
    #     fid = torch.load(run_path / "fid.pt")
    # else:
    #     log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    #     fid = FrechetInceptionDistance(feature=2048).to(device)
    #     total = 0
    #     for batch in tqdm(train_loader):
    #         x, _ = batch
    #         total += len(x)
    #         x = x.to(device)
    #         x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    #         x = x.expand(-1, 3, -1, -1)
    #         fid.update(x, real=True)
    #         # if total > 5000:
    #         #     break
    #
    #     torch.save(fid, run_path / "fid.pt")
    #
    #     print(f"Total real: {total}")
    # images = torch.load(run_path / "predictions/result.pt")
    # dataset = TensorDataset(images)
    # dataloader = DataLoader(dataset, batch_size=sample_config.batch_size, shuffle=False)
    # # test_loader = dataloader
    # total = 0
    # for batch in tqdm(test_loader):
    #     # x, _ = batch
    #     x = batch[0]
    #     total += len(x)
    #     x = x.to(device)
    #     x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    #     x = x.expand(-1, 3, -1, -1)
    #     fid.update(x, real=False)
    #     if total > 1000:
    #         break
    #
    # print("Computing FID...")
    # print(f"Total fake: {total}")
    # print(fid.compute())
    #
    #     torch.save(fid, run_path / "fid.pt")
    #
    # images = torch.load(run_path / "predictions/result.pt")
    # dataset = TensorDataset(images)
    # dataloader = DataLoader(dataset, batch_size=sample_config.batch_size, shuffle=False)
    #
    # for batch in tqdm(dataloader):
    #     x = batch[0]
    #     x = x.to(device)
    #     x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    #     x = x.expand(-1, 3, -1, -1)
    #     fid.update(x, real=False)
    #
    # score = fid.compute()
    # print(score)


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    fid(cfg)


if __name__ == "__main__":
    main()
