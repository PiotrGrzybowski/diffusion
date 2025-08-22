import time

import rootutils
import torch
from lightning import Fabric
from omegaconf import DictConfig

from diffusion.utils.extras import extras
from diffusion.utils.fabric_progress import create_nested_progress_tracker
from diffusion.utils.ranked_logger import RankedLogger
from diffusion.utils.run_utils import custom_main


root_path = rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=False)
configs_path = root_path / "configs"


log = RankedLogger(__name__, rank_zero_only=True)


def sample(cfg: DictConfig):
    fabric = Fabric(devices=2, accelerator="auto")
    fabric.launch()

    samples = 64
    batch_size = 4
    batches = samples // batch_size
    timesteps = 1000

    # for i in range(batches):
    #     batch = torch.randn(batch_size, 3, 32, 32, device=fabric.device)
    #
    #     for step in range(timesteps):
    #         batch = batch * 0.99
    #         time.sleep(0.01)  # Reduced for faster demo

    def a():
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

    a()

    # run_path = Path(cfg.paths.output_dir)
    # module_cfg = OmegaConf.load(run_path / "config.yaml")
    #
    # fabric = Fabric(devices=2, accelerator=cfg.accelerator)
    # fabric.launch()
    #
    # log.info(f"Instantiating model <{module_cfg.diffusion._target_}>")
    # model = hydra.utils.instantiate(module_cfg.diffusion)
    #
    # log.info("Instantiating loggers...")
    # loggers = instantiate_loggers(cfg.get("logger"))
    #
    # ckpt_path = run_path / "checkpoints" / cfg.ckpt_name
    # state_dict = torch.load(ckpt_path)["state_dict"]
    # model.load_state_dict(state_dict)
    # module = fabric.setup_module(model)
    #
    # samples = 64
    # batch_size = 4
    # batches = samples // batch_size
    # timesteps = cfg.sample_timesteps
    #
    # tracker = create_nested_progress_tracker(fabric, name="sampling", log_interval=10)
    #
    # with tracker:
    #     progress = tracker.setup_sampling(batches, timesteps)
    #
    #     for i in range(batches):
    #         progress.next_batch()
    #         batch = torch.randn(batch_size, module_cfg.in_channels, module_cfg.dim, module_cfg.dim, device=fabric.device)
    #
    #         for x_t in module.sample(batch, cfg.sample_timesteps):
    #             progress.step()
    #         x = ((x_t + 1) * 127.5).clamp(0, 255).to(device=fabric.device, dtype=torch.uint8)
    #
    #     progress.finish("Sample generation completed!")

    # sample_path = run_path / "samples"
    # sample_path.mkdir(exist_ok=True)
    # cv2.imwrite(str(sample_path / f"sample_{cfg.ckpt_name}.png"), result)
    # torch.save(x_t, sample_path / f"sample_{cfg.ckpt_name}.pt")


@custom_main(version_base="1.3", config_path=str(configs_path), config_name="sample.yaml")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    sample(cfg)


if __name__ == "__main__":
    main()
