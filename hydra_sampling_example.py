"""Minimalistic sampling example with Hydra integration and progress tracking only."""

import time

import hydra
import rootutils
import torch
from lightning import Fabric
from omegaconf import DictConfig

from diffusion.utils.fabric_progress import create_nested_progress_tracker


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)


def run_sampling(cfg: DictConfig):
    fabric = Fabric(devices=1, accelerator="cpu")
    fabric.launch()
    samples = getattr(cfg, "samples", 64)
    batch_size = 4
    batches = samples // batch_size
    timesteps = getattr(cfg, "sample_timesteps", 100)

    tracker = create_nested_progress_tracker(fabric, name="sampling", log_interval=10)

    with tracker:
        progress = tracker.setup_sampling(batches, timesteps)

        # Test logging that would normally break progress bars
        tracker.logger.log_info("🚀 Starting sampling with Rich logging + progress bars")

        for batch_idx in range(batches):
            # This logging should now work without breaking progress bars!
            tracker.logger.log_info(f"📦 Processing batch {batch_idx + 1}/{batches}")
            progress.next_batch()
            batch = torch.randn(batch_size, 3, 32, 32, device=fabric.device)

            for step in range(timesteps):
                noise_scale = (timesteps - step) / timesteps
                batch = batch * noise_scale

                # Add some logging during progress
                if step % 20 == 0:
                    tracker.logger.log_info(f"⏰ Denoising step {step + 1}/{timesteps} in batch {batch_idx + 1}")

                if step % 50 == 0:
                    tracker.logger.log_warning(f"🔄 Halfway through batch {batch_idx + 1}")

                time.sleep(0.05)  # Slightly faster for testing

                progress.step()

        progress.finish("✅ Sampling completed successfully!")
        tracker.logger.log_info("🎉 All done! Rich logging + progress bars working together!")


@hydra.main(version_base=None, config_path="configs", config_name="sample")
def main(cfg: DictConfig) -> None:
    run_sampling(cfg)


if __name__ == "__main__":
    main()
