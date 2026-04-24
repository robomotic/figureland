"""
Unit tests for deterministic seed management.
"""

import torch
import numpy as np
import random
import pytest
from figureland.parallel.seed import SeedManager, get_split_seed


def test_seed_manager_creation():
    """Test seed manager initialization."""
    manager = SeedManager(base_seed=42)
    assert manager.base_seed == 42
    assert manager.counters['train'] == 0
    assert manager.counters['val'] == 0
    assert manager.counters['test'] == 0


def test_split_seed_generation():
    """Test split seeds are isolated."""
    manager = SeedManager(base_seed=42)

    seed_train = manager.get_seed('train', 0)
    seed_val = manager.get_seed('val', 0)
    seed_test = manager.get_seed('test', 0)

    assert seed_train != seed_val
    assert seed_val != seed_test
    assert seed_train != seed_test


def test_seed_determinism():
    """Test same split and index produce same seed."""
    manager1 = SeedManager(base_seed=42)
    manager2 = SeedManager(base_seed=42)

    assert manager1.get_seed('train', 0) == manager2.get_seed('train', 0)
    assert manager1.get_seed('val', 100) == manager2.get_seed('val', 100)


def test_counter_increment():
    """Test automatic counter increment works."""
    manager = SeedManager(base_seed=42)

    seed1 = manager.get_seed('train')
    seed2 = manager.get_seed('train')

    assert seed1 != seed2
    assert manager.counters['train'] == 2


def test_deterministic_mode():
    """Test deterministic mode sets all RNG states."""
    seed = 12345
    SeedManager.set_deterministic_mode(seed)

    # All RNGs should produce same sequence
    rand1 = torch.rand(1).item()
    rand2 = np.random.rand(1).item()
    rand3 = random.random()

    SeedManager.set_deterministic_mode(seed)

    assert torch.rand(1).item() == rand1
    assert np.random.rand(1).item() == rand2
    assert random.random() == rand3


def test_get_split_seed_standalone():
    """Test standalone split seed function."""
    seed1 = get_split_seed(42, 'train', 0)
    seed2 = get_split_seed(42, 'train', 0)
    seed3 = get_split_seed(42, 'val', 0)

    assert seed1 == seed2
    assert seed1 != seed3


def test_seed_reset():
    """Test seed manager reset."""
    manager = SeedManager(base_seed=42)
    manager.get_seed('train')
    manager.get_seed('train')
    assert manager.counters['train'] == 2

    manager.reset()
    assert manager.counters['train'] == 0
