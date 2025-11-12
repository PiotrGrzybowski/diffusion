import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable

import hydra
from hydra import TaskFunction
from omegaconf import DictConfig, OmegaConf

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


def custom_main(
    config_path: str,
    config_name: str,
    version_base: str,
) -> Callable[[TaskFunction], Any]:
    config_stem = config_name.split(".")[0] if "." in config_name else config_name
    new_config_name = f"{config_stem}_run.yaml"
    new_config_path = Path(config_path) / new_config_name

    shutil.copy(Path(config_path) / config_name, new_config_path)

    cfg = OmegaConf.load(new_config_path)
    if cfg.run_name is None:
        cfg.run_name = generate_run_name()
    OmegaConf.save(cfg, new_config_path)

    try:
        return hydra.main(config_path=config_path, config_name=new_config_name, version_base=version_base)
    finally:
        os.remove(new_config_path)


ADJECTIVES = [
    "snowy",
    "rapid",
    "gentle",
    "mighty",
    "icy",
    "clever",
    "brave",
    "fierce",
    "silent",
    "bold",
    "swift",
    "noble",
    "radiant",
    "vivid",
    "luminous",
]
NOUNS = [
    "rocket",
    "comet",
    "falcon",
    "glacier",
    "horizon",
    "whale",
    "tiger",
    "eagle",
    "phoenix",
    "dragon",
    "panther",
    "wolf",
    "leopard",
    "jaguar",
    "lynx",
]


def generate_run_name():
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    number = random.randint(0, 99)
    return f"{adjective}-{noun}-{number}"
