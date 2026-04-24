"""
Configuration system using Hydra OmegaConf.
All configuration parameters are loaded from YAML config files.
"""

import logging
import torch
from omegaconf import DictConfig, OmegaConf
from typing import Optional

log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration parameters."""
    log.debug("Validating configuration")

    # Validate shape types
    valid_shapes = {'square', 'rectangle', 'triangle', 'hexagon', 'trapezoid'}
    for shape in cfg.shapes.shape_types:
        if shape not in valid_shapes:
            raise ValueError(f"Invalid shape type: {shape}. Valid types: {valid_shapes}")

    # Validate environment type
    valid_envs = {'rectangle', 'circle', 'square'}
    if cfg.physics.environment_type not in valid_envs:
        raise ValueError(f"Invalid environment type: {cfg.physics.environment_type}. Valid types: {valid_envs}")

    # Validate force field
    if cfg.physics.force_field is not None:
        valid_fields = {'gravity', 'custom_vector', 'turbulence', 'none'}
        if cfg.physics.force_field not in valid_fields:
            raise ValueError(f"Invalid force field: {cfg.physics.force_field}. Valid types: {valid_fields}")

    # Validate split ratios
    total_ratio = cfg.parallel.train_ratio + cfg.parallel.val_ratio + cfg.parallel.test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    log.debug("Configuration validation passed")


def get_device(cfg: DictConfig) -> torch.device:
    """Get PyTorch device based on configuration."""
    device = cfg.get('device')
    if device:
        return torch.device(device)

    use_gpu = cfg.get('use_gpu', True)
    if use_gpu:
        if torch.cuda.is_available():
            log.info("Using CUDA GPU")
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            log.info("Using MPS GPU")
            return torch.device('mps')

    log.info("Using CPU device")
    return torch.device('cpu')


def print_config(cfg: DictConfig) -> None:
    """Pretty print configuration."""
    log.info("Running with configuration:")
    log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
