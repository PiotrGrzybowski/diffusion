"""Example usage of FabricProgressLogger for nested loops in DDP setup."""

import time

import torch
from lightning import Fabric

from diffusion.utils.fabric_progress import FabricProgressLogger, create_nested_progress_tracker


def example_training_loop():
    """Example showing how to use FabricProgressLogger in a training setup."""

    # Initialize Fabric (this would be your actual setup)
    fabric = Fabric(devices=2, accelerator="auto")
    fabric.launch()

    # Create the progress logger
    progress_logger = FabricProgressLogger(
        fabric=fabric,
        name="training",
        log_interval=5,  # Log every 5 steps
    )

    # Training parameters
    epochs = 3
    batches_per_epoch = 20
    timesteps_per_batch = 50

    with progress_logger.progress_context():
        # Add overall progress task
        progress_logger.add_task(
            name="overall",
            title="Overall Progress",
            total=epochs * batches_per_epoch * timesteps_per_batch,
            description="Training diffusion model",
        )

        # Add epoch progress task
        progress_logger.add_task(name="epochs", title="Epochs", total=epochs, description="Training epochs")

        for epoch in range(epochs):
            progress_logger.update_task("epochs", advance=1, force_log=True)
            progress_logger.log_info(f"Starting epoch {epoch + 1}/{epochs}")

            # Add/reset batch progress for this epoch
            if epoch == 0:
                progress_logger.add_task(name="batches", title=f"Batches (Epoch {epoch + 1})", total=batches_per_epoch)
            else:
                progress_logger.reset_task("batches", total=batches_per_epoch, title=f"Batches (Epoch {epoch + 1})")

            for batch_idx in range(batches_per_epoch):
                progress_logger.update_task("batches", advance=1)

                # Add/reset timestep progress for this batch
                if epoch == 0 and batch_idx == 0:
                    progress_logger.add_task(
                        name="timesteps", title=f"Timesteps (Epoch {epoch + 1}, Batch {batch_idx + 1})", total=timesteps_per_batch
                    )
                else:
                    progress_logger.reset_task(
                        "timesteps", total=timesteps_per_batch, title=f"Timesteps (Epoch {epoch + 1}, Batch {batch_idx + 1})"
                    )

                # Simulate diffusion timesteps
                for timestep in range(timesteps_per_batch):
                    # Simulate some work
                    x = torch.randn(4, 3, 32, 32, device=fabric.device)

                    # Update progress
                    progress_logger.update_task("timesteps", advance=1)
                    progress_logger.update_task("overall", advance=1)

                    # Simulate loss calculation and logging
                    if timestep % 10 == 0:
                        fake_loss = torch.rand(1).item()
                        progress_logger.update_task(
                            "timesteps",
                            advance=0,  # Don't advance, just log
                            log_message=f"Timestep {timestep}, Loss: {fake_loss:.4f}",
                        )

                # Log batch completion
                if batch_idx % 5 == 0:
                    progress_logger.log_info(f"Completed batch {batch_idx + 1}/{batches_per_epoch} in epoch {epoch + 1}")

        # Final completion message
        progress_logger.finish_task("overall", "Training completed successfully!")


def example_sampling_loop():
    """Example showing sampling progress (similar to your generate.py)."""

    fabric = Fabric(devices=2, accelerator="auto")
    fabric.launch()

    samples = 64
    batch_size = 4
    batches = samples // batch_size
    timesteps = 1000

    tracker = create_nested_progress_tracker(fabric, name="sampling", log_interval=10)

    with tracker:
        progress = tracker.setup_sampling(batches, timesteps)

        for i in range(batches):
            progress.next_batch()
            batch = torch.randn(batch_size, 3, 32, 32, device=fabric.device)

            for step in range(timesteps):
                batch = batch * 0.99
                time.sleep(0.01)  # Reduced for faster demo
                progress.step()
        progress.finish()


if __name__ == "__main__":
    example_sampling_loop()
