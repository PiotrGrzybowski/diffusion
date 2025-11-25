from pathlib import Path

from omegaconf import DictConfig

from diffusion.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def find_ckpt_path(cfg: DictConfig) -> str | None:
    if cfg.run_name and cfg.task_name:
        run_path = Path(cfg.paths.log_dir) / cfg.task_name / "hydra" / cfg.run_name
        ckpt_path = run_path / "checkpoints" / cfg.ckpt_name
        if not ckpt_path.exists():
            log.info(f"Checkpoint not found at {ckpt_path}.")
            log.info(f"Initalizing run {cfg.run_name}...")
            return None
        else:
            log.info(f"Loading ckpt from {cfg.run_name}/{cfg.ckpt_name}...")
            return str(ckpt_path)
