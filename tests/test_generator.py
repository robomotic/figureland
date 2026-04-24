"""
Unit tests for main dataset generator.
"""

import torch
import pytest
from figureland import DatasetGenerator
from omegaconf import OmegaConf


def create_test_config():
    """Create a complete test configuration."""
    cfg = OmegaConf.create({
        'resolution': [64, 64],
        'episode_length': 10,
        'fps': 30,
        'anti_alias': 2,
        'batch_size': 2,
        'use_gpu': False,
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


def test_generator_creation():
    """Test generator initialization."""
    cfg = create_test_config()
    generator = DatasetGenerator(cfg)
    assert generator is not None
    assert generator.device.type == 'cpu'


def test_episode_generation():
    """Test single episode generation."""
    cfg = create_test_config()
    generator = DatasetGenerator(cfg)
    episode = generator.generate_episode(seed=42)

    assert 'frames' in episode
    assert 'metadata' in episode
    assert 'shape_history' in episode
    assert episode['frames'].shape[0] == 10  # episode_length
    assert episode['frames'].shape[1] == 2   # batch_size


def test_deterministic_generation():
    """Test same seed produces identical output."""
    cfg = create_test_config()
    cfg.episode_length = 5
    cfg.batch_size = 1
    cfg.resolution = [32, 32]

    generator1 = DatasetGenerator(cfg)
    episode1 = generator1.generate_episode(seed=123)

    generator2 = DatasetGenerator(cfg)
    episode2 = generator2.generate_episode(seed=123)

    assert torch.allclose(episode1['frames'], episode2['frames'], atol=1e-6)
    assert episode1['metadata']['seed'] == episode2['metadata']['seed']


def test_generator_device_transfer():
    """Test generator can be moved between devices."""
    cfg = create_test_config()
    generator = DatasetGenerator(cfg)
    generator_cpu = generator.to('cpu')
    assert generator_cpu.device.type == 'cpu'


def test_generate_multiple_episodes():
    """Test sequential generation of multiple episodes."""
    cfg = create_test_config()
    cfg.episode_length = 5
    cfg.batch_size = 1
    generator = DatasetGenerator(cfg)
    result = generator.generate(n_episodes=3, split='train')

    assert result['split'] == 'train'
    assert result['n_episodes'] == 3
    assert len(result['episodes']) == 3
