from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig

from diffusion.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def task_wrapper(task_func: Callable) -> Callable:
    def wrap(cfg: DictConfig) -> None:
        try:
            task_func(cfg=cfg)
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")

            if find_spec("wandb"):
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

    return wrap
