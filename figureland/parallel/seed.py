"""
Deterministic seed management system.
Ensures reproducible generation with proper train/val/test splitting.
"""

import torch
import numpy as np
import random
from typing import Dict, Optional


class SeedManager:
    """Manages deterministic seeds for train/val/test splits."""

    def __init__(
        self,
        base_seed: int = 42,
        split_offsets: Optional[Dict[str, int]] = None
    ):
        self.base_seed = base_seed
        self.split_offsets = split_offsets or {
            'train': 0,
            'val': 1000000,
            'test': 2000000
        }
        self.counters = {split: 0 for split in self.split_offsets.keys()}

    def get_seed(self, split: str, index: Optional[int] = None) -> int:
        """Get deterministic seed for given split and episode index."""
        if index is None:
            index = self.counters[split]
            self.counters[split] += 1

        return self.base_seed + self.split_offsets[split] + index

    def reset(self) -> None:
        """Reset all split counters to zero."""
        for split in self.counters:
            self.counters[split] = 0

    @staticmethod
    def set_deterministic_mode(seed: int, device: Optional[torch.device] = None) -> None:
        """Set all random number generators to deterministic state."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available() and device and device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if torch.backends.mps.is_available() and device and device.type == 'mps':
            torch.mps.manual_seed(seed)


def get_split_seed(base_seed: int, split: str, episode_index: int) -> int:
    """Standalone function to get split seed without manager instance."""
    offsets = {
        'train': 0,
        'val': 1000000,
        'test': 2000000
    }
    return base_seed + offsets[split] + episode_index
