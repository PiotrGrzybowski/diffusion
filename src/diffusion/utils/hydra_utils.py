import hydra
from omegaconf import DictConfig

from diffusion.utils.naming import auto_generate_names, validate_naming_config


def resolve_config(cfg: DictConfig) -> DictConfig:
    """Resolve configuration before training starts."""
    cfg = auto_generate_names(cfg)
    validate_naming_config(cfg)

    # Now instantiate components (these may create directories)
    datamodule = hydra.utils.instantiate(cfg.data)
    variance = hydra.utils.instantiate(cfg.diffusion.variance_strategy)

    # Set derived dimensions
    cfg.in_channels = datamodule.channels
    cfg.out_channels = cfg.in_channels * 2 if variance.trainable else cfg.in_channels
    cfg.dim = datamodule.shape[-1]

    return cfg
