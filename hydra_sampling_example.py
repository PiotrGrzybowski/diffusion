from pathlib import Path

import hydra
import rootutils
import torch
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from diffusion.utils.extras import extras
from diffusion.utils.fabric_progress import RichRankedLogger, create_rich_tracker
from diffusion.utils.run_utils import custom_main


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"

logger = RichRankedLogger("sampling", rank_zero_only=True)


def run_sampling(cfg: DictConfig):
    fabric = Fabric(devices=2, accelerator="gpu")
    fabric.launch()

    run_path = Path(cfg.paths.output_dir)
    config = OmegaConf.load(run_path / "config.yaml")

    logger.info(f"Instantiating model <{config.diffusion._target_}>")
    model = hydra.utils.instantiate(config.diffusion)

    ckpt_path = run_path / "checkpoints" / cfg.ckpt_name
    logger.info(f"Loading checkpoint {ckpt_path}")
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    model = fabric.setup_module(model)

    fid_path = Path(cfg.paths.log_dir) / cfg.task_name / "fid.pt"
    logger.info(f"Loading FID from {fid_path}")
    fid_metric = torch.load(fid_path, map_location=fabric.device)

    resize = transforms.Resize((299, 299), antialias=True)

    batches = cfg.samples // cfg.batch_size
    tracker = create_rich_tracker(logger, fabric.global_rank, name="sampling", log_interval=10)

    with tracker:
        progress = tracker.setup_sampling(batches, cfg.sample_timesteps)
        samples = []

        for batch_idx in range(batches):
            progress.next_batch()
            batch = torch.randn(cfg.batch_size, config.in_channels, config.dim, config.dim, device=fabric.device)

            x_t = torch.empty(0)
            for x_t in model.sample(batch, cfg.sample_timesteps):
                progress.step()

            x_t = resize(x_t)

            fid_metric.update(x_t, real=False)

            current_fid = fid_metric.compute()
            progress.update_metrics(FID=float(current_fid), Batch=f"{batch_idx + 1}/{batches}")

            samples.append(x_t)

        # Final FID computation
        final_fid = fid_metric.compute()
        fabric.barrier()
        progress.finish(f"✅ Sampling completed! Final FID: {final_fid:.4f}")
        logger.info(f"🎯 Final FID Score: {final_fid:.4f}")

    # rank_samples = torch.cat(samples, dim=0)
    #
    # fabric.barrier()
    # logger.info(f"🔄 Rank {fabric.global_rank} completed, waiting for other ranks...")
    #
    # if fabric.world_size > 1:
    #     gathered_samples = fabric.all_gather(rank_samples)
    #     assert isinstance(gathered_samples, torch.Tensor)
    #
    #     if fabric.global_rank == 0:
    #         all_gathered = gathered_samples.view(-1, *gathered_samples.shape[2:])
    #         torch.save(all_gathered, run_path / "dd-samples_all_ranks.pt")
    #         logger.info(f"💾 Saved {all_gathered.shape[0]} samples from all ranks to samples_all_ranks.pt")
    # else:
    #     torch.save(rank_samples, run_path / "dd-samples_all_ranks.pt")
    #
    # torch.save(rank_samples, run_path / f"dd-samples_rank_{fabric.global_rank}.pt")
    # fabric.barrier()


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="sample.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    run_sampling(cfg)


if __name__ == "__main__":
    main()
