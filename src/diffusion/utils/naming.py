"""Automatic run naming utilities for Hydra/WandB synchronization.

This module provides functionality to automatically generate run names from Hydra
configuration choices, ensuring synchronized directory structures between Hydra
and WandB logging systems.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from diffusion.utils.ranked_logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def generate_run_name_from_hydra(debug_mode: bool = False) -> str:
    """Generate run name from Hydra runtime choices for use as OmegaConf resolver.

    This function is called during Hydra config resolution, before directories
    are created, allowing the run_name to be set dynamically.

    Naming convention: {model}-{mean}-{variance}-{loss}

    Args:
        debug_mode: If True, return "debug" as the run name

    Returns:
        Generated run name string, or "auto" if Hydra config not yet available
    """
    # Handle debug mode
    if debug_mode:
        return "debug"

    try:
        hydra_cfg = HydraConfig.get()
        choices = hydra_cfg.runtime.choices

        model = choices.get("diffusion/model", "unknown")
        mean = choices.get("diffusion/mean_strategy", "unknown")
        variance = choices.get("diffusion/variance_strategy", "unknown")
        loss = choices.get("diffusion/loss", "unknown")
        scheduler = choices.get("diffusion/scheduler", "unknown")

        # Sanitize names
        model_safe = _sanitize_name(model)
        mean_safe = _sanitize_name(mean)
        variance_safe = _sanitize_name(variance)
        loss_safe = _sanitize_name(loss)
        scheduler_safe = _sanitize_name(scheduler)

        return f"{model_safe}-{mean_safe}-{variance_safe}-{loss_safe}-{scheduler_safe}"
    except Exception:
        # Hydra config not yet initialized, return placeholder
        return "auto"


def _sanitize_name(name: str) -> str:
    """Make name filesystem and URL safe.

    Args:
        name: The name to sanitize

    Returns:
        Sanitized name that is filesystem and URL safe

    Example:
        >>> _sanitize_name(
        ...     "Model-Name_123"
        ... )
        'model-name_123'
        >>> _sanitize_name(
        ...     "special@chars#here"
        ... )
        'special_chars_here'
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9_-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _get_timestamp_suffix() -> str:
    """Generate compact timestamp for collision avoidance.

    Returns:
        Timestamp string in format YYYYMMDDHHmmss

    Example:
        >>> _get_timestamp_suffix()  # doctest: +SKIP
        '20251209143022'
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def generate_run_name(
    choices: Dict[str, str],
    task_name: str,
    log_dir: Path,
    append_timestamp_on_collision: bool = True,
) -> str:
    """Generate run name from Hydra runtime choices.

    Args:
        choices: HydraConfig runtime choices dict
        task_name: Task name (e.g., 'mnist', 'cifar10')
        log_dir: Base logging directory
        append_timestamp_on_collision: Add timestamp if name exists

    Returns:
        Generated run name string

    Example:
        >>> import tempfile
        >>> choices = {
        ...     "diffusion/model": "unet",
        ...     "diffusion/mean_strategy": "epsilon",
        ...     "diffusion/variance_strategy": "fixed_small",
        ...     "diffusion/loss": "mse_epsilon_simple",
        ... }
        >>> with (
        ...     tempfile.TemporaryDirectory() as tmpdir
        ... ):
        ...     generate_run_name(
        ...         choices,
        ...         "mnist",
        ...         Path(
        ...             tmpdir
        ...         ),
        ...         False,
        ...     )
        'unet-epsilon-fixed_small-mse_epsilon_simple'
    """
    # Extract choices
    model = choices.get("diffusion/model", "unknown_model")
    mean = choices.get("diffusion/mean_strategy", "unknown_mean")
    variance = choices.get("diffusion/variance_strategy", "unknown_variance")
    loss = choices.get("diffusion/loss", "unknown_loss")

    # Sanitize names (config names are already concise)
    model_safe = _sanitize_name(model)
    mean_safe = _sanitize_name(mean)
    variance_safe = _sanitize_name(variance)
    loss_safe = _sanitize_name(loss)

    # Generate base name (convention: model-mean-variance-loss)
    base_name = f"{model_safe}-{mean_safe}-{variance_safe}-{loss_safe}"

    # Check for collision
    if append_timestamp_on_collision and task_name:
        expected_path = log_dir / task_name / "hydra" / base_name
        if expected_path.exists():
            timestamp = _get_timestamp_suffix()
            final_name = f"{base_name}-{timestamp}"
            log.info(f"Run name '{base_name}' already exists. Using timestamped name: '{final_name}'")
            return final_name

    return base_name


def validate_naming_config(cfg: DictConfig) -> None:
    """Validate that naming configuration is properly set.

    Args:
        cfg: Hydra configuration object

    Raises:
        ValueError: If critical naming fields are invalid
    """
    if not cfg.task_name:
        raise ValueError("task_name cannot be empty after auto-generation. Check that 'data' choice is set in Hydra config.")

    if not cfg.run_name:
        raise ValueError("run_name cannot be empty after auto-generation. Check that diffusion model/loss/variance choices are set.")

    # Validate filesystem safety
    invalid_chars = set('<>:"/\\|?*')
    if any(c in cfg.run_name for c in invalid_chars):
        raise ValueError(f"run_name '{cfg.run_name}' contains invalid filesystem characters. Invalid chars: {invalid_chars}")


def auto_generate_names(cfg: DictConfig) -> DictConfig:
    """Auto-generate run_name and task_name if not provided.

    This function should be called early in resolve_config() to ensure
    names are available for:
    - Hydra output directory creation
    - WandB logger initialization
    - Checkpoint path discovery

    Args:
        cfg: Hydra DictConfig object

    Returns:
        Modified cfg with run_name and task_name set

    Side Effects:
        - Mutates cfg.run_name if null
        - Mutates cfg.task_name if null (falls back to data name)
        - Mutates cfg.logger.wandb.offline if debug_mode is enabled
    """
    hydra_cfg = HydraConfig.get()
    choices = hydra_cfg.runtime.choices

    # Auto-generate task_name from data choice if not set
    if cfg.task_name is None:
        data_choice = choices.get("data", "unknown_task")
        cfg.task_name = data_choice
        log.info(f"Auto-generated task_name from data choice: '{cfg.task_name}'")

    # Handle debug mode
    if cfg.get("debug_mode", False):
        if cfg.run_name is None:
            cfg.run_name = "debug"
        # Force WandB offline for debug runs to prevent cloud pollution
        if "logger" in cfg and "wandb" in cfg.logger:
            cfg.logger.wandb.offline = True
        log.info("Debug mode: using run_name='debug' with offline logging")
        return cfg

    # Auto-generate run_name from config choices if not set
    if cfg.run_name is None:
        log_dir = Path(cfg.paths.log_dir)
        cfg.run_name = generate_run_name(
            choices=choices,
            task_name=cfg.task_name,
            log_dir=log_dir,
            append_timestamp_on_collision=True,
        )
        log.info(f"Auto-generated run_name: '{cfg.run_name}'")
    else:
        log.info(f"Using explicit run_name: '{cfg.run_name}'")

    return cfg
