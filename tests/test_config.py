"""
Unit tests for configuration system with Hydra/OmegaConf.
"""

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from figureland.config import validate_config, get_device


def create_test_config():
    """Create a minimal valid config for testing."""
    cfg = OmegaConf.create({
        'resolution': [256, 256],
        'episode_length': 100,
        'fps': 30,
        'anti_alias': 2,
        'batch_size': 32,
        'use_gpu': False,
        'device': None,
        'parallel_workers': 4,
        'seed': 42,
        'deterministic': True,
        'output_dir': './output',
        'shapes': {
            'shape_types': ['square', 'rectangle', 'triangle'],
            'size_range': [0.05, 0.2],
            'mass_range': [0.1, 10.0],
            'elasticity_range': [0.2, 0.9],
            'aspect_ratio_mode': 'variable',
            'fixed_aspect_ratio': None
        },
        'physics': {
            'environment_type': 'rectangle',
            'bounds': [-1.0, 1.0],
            'gravity': 9.8,
            'friction': 0.05,
            'air_resistance': 0.01,
            'force_field': None,
            'dt': 0.016666666666666666,
            'substeps': 4,
            'collisions_enabled': True
        },
        'parallel': {
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'seed_offsets': {'train': 0, 'val': 1000000, 'test': 2000000}
        },
        'output': {
            'formats': ['png', 'h5'],
            'video_codec': 'mp4v',
            'video_fps': 30,
            'h5_compression': 'gzip'
        }
    })
    return cfg


def test_validate_config():
    """Test config validation."""
    cfg = create_test_config()
    validate_config(cfg)  # Should not raise


def test_invalid_shape_type():
    """Test invalid shape type raises error."""
    cfg = create_test_config()
    cfg.shapes.shape_types = ['invalid_shape']
    with pytest.raises(ValueError, match="Invalid shape type"):
        validate_config(cfg)


def test_invalid_environment_type():
    """Test invalid environment type raises error."""
    cfg = create_test_config()
    cfg.physics.environment_type = 'invalid_env'
    with pytest.raises(ValueError, match="Invalid environment type"):
        validate_config(cfg)


def test_invalid_force_field():
    """Test invalid force field raises error."""
    cfg = create_test_config()
    cfg.physics.force_field = 'invalid_field'
    with pytest.raises(ValueError, match="Invalid force field"):
        validate_config(cfg)


def test_invalid_split_ratios():
    """Test invalid split ratios raise errors."""
    cfg = create_test_config()
    cfg.parallel.train_ratio = 0.5
    cfg.parallel.val_ratio = 0.5
    cfg.parallel.test_ratio = 0.5  # Sum > 1
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        validate_config(cfg)


def test_negative_ratios():
    """Test negative ratios raise error."""
    cfg = create_test_config()
    cfg.parallel.train_ratio = -0.1
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_get_device():
    """Test device detection."""
    cfg = create_test_config()
    cfg.use_gpu = False
    device = get_device(cfg)
    assert device.type == 'cpu'


def test_config_from_dict():
    """Test creating generator from dict config."""
    from figureland import DatasetGenerator

    config_dict = {
        'resolution': [128, 128],
        'episode_length': 50,
        'fps': 30,
        'anti_alias': 2,
        'batch_size': 16,
        'use_gpu': False,
        'seed': 42,
        'shapes': {
            'shape_types': ['square'],
            'size_range': [0.05, 0.2],
            'mass_range': [0.1, 10.0],
            'elasticity_range': [0.2, 0.9]
        },
        'physics': {
            'environment_type': 'rectangle',
            'bounds': [-1.0, 1.0],
            'gravity': 9.8,
            'friction': 0.05,
            'air_resistance': 0.01,
            'force_field': None,
            'dt': 0.016666666666666666,
            'substeps': 4,
            'collisions_enabled': True
        },
        'parallel': {
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1
        }
    }
    generator = DatasetGenerator(config_dict)
    assert generator.cfg.resolution == [128, 128]
    assert generator.cfg.episode_length == 50
    assert generator.cfg.batch_size == 16


def test_valid_shape_types():
    """Test all valid shape types pass validation."""
    cfg = create_test_config()
    for shape_type in ['square', 'rectangle', 'triangle', 'hexagon', 'trapezoid']:
        cfg.shapes.shape_types = [shape_type]
        validate_config(cfg)  # Should not raise


def test_valid_environment_types():
    """Test all valid environment types pass validation."""
    cfg = create_test_config()
    for env_type in ['rectangle', 'circle', 'square']:
        cfg.physics.environment_type = env_type
        validate_config(cfg)  # Should not raise


def test_valid_force_fields():
    """Test valid force field options."""
    cfg = create_test_config()
    for field in [None, 'gravity', 'turbulence', 'none']:
        cfg.physics.force_field = field
        validate_config(cfg)  # Should not raise
