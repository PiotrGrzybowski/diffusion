"""Minimalistic sampling example with Hydra integration and progress tracking only."""

import time
from pathlib import Path

import hydra
import rootutils
import torch
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf

from diffusion.utils.fabric_progress import RichRankedLogger, create_rich_tracker


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)


def run_sampling(cfg: DictConfig):
    fabric = Fabric(devices=1, accelerator="cpu")
    fabric.launch()
    samples = getattr(cfg, "samples", 64)
    batch_size = 4
    batches = samples // batch_size
    timesteps = getattr(cfg, "sample_timesteps", 100)

    # Create logger and tracker separately - cleaner separation of concerns
    logger = RichRankedLogger("sampling", rank_zero_only=True)
    tracker = create_rich_tracker(logger, fabric.global_rank, name="sampling", log_interval=10)

    run_path = Path(cfg.paths.output_dir)
    module_cfg = OmegaConf.load(run_path / "config.yaml")

    logger.info(f"Instantiating model <{module_cfg.diffusion._target_}>")
    model = hydra.utils.instantiate(module_cfg.diffusion)

    with tracker:
        progress = tracker.setup_sampling(batches, timesteps)

        # Now we can use logger directly - much cleaner!
        logger.info("🚀 Starting sampling with Rich logging + progress bars")

        for batch_idx in range(batches):
            # Direct logger access - no more nested calls!
            logger.info(f"📦 Processing batch {batch_idx + 1}/{batches}")
            progress.next_batch()
            batch = torch.randn(batch_size, 3, 32, 32, device=fabric.device)

            for step in range(timesteps):
                noise_scale = (timesteps - step) / timesteps
                batch = batch * noise_scale

                # Add some logging during progress
                if step % 20 == 0:
                    logger.info(f"⏰ Denoising step {step + 1}/{timesteps} in batch {batch_idx + 1}")

                if step % 50 == 0:
                    logger.warning(f"🔄 Halfway through batch {batch_idx + 1}")

                time.sleep(0.05)  # Slightly faster for testing

                progress.step()

        progress.finish("✅ Sampling completed successfully!")
        logger.info("🎉 All done! Rich logging + progress bars working together!")


@hydra.main(version_base=None, config_path="configs", config_name="sample")
def main(cfg: DictConfig) -> None:
    run_sampling(cfg)


if __name__ == "__main__":
    main()
