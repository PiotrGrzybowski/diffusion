from pathlib import Path

from omegaconf import DictConfig

from diffusion.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def find_ckpt_path(cfg: DictConfig) -> str | None:
    """Find checkpoint path for run resumption.

    Returns None if:
    - Checkpoint file doesn't exist (new run initialization)
    """
    if cfg.run_name and cfg.task_name:
        run_path = Path(cfg.paths.log_dir) / cfg.task_name / "hydra" / cfg.run_name
        ckpt_path = run_path / "checkpoints" / cfg.ckpt_name
        if not ckpt_path.exists():
            log.info(f"Checkpoint not found at {ckpt_path}.")
            log.info(f"Initializing new run '{cfg.run_name}' in task '{cfg.task_name}'...")
            return None
        else:
            log.info(f"Resuming run '{cfg.run_name}' from {cfg.ckpt_name}...")
            return str(ckpt_path)
    else:
        log.info("No run_name or task_name specified - running without checkpointing")
        return None
