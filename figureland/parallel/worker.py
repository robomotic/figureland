"""
Parallel generation workers using standard multiprocessing.
Stateless design suitable for cloud deployment.
"""

import torch
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import DictConfig
from tqdm import tqdm

from .seed import SeedManager, get_split_seed


def generate_episode_worker(args: Tuple[DictConfig, str, int, int]) -> Dict[str, Any]:
    """Worker function for parallel episode generation. Stateless and cloud compatible."""
    # Import here to avoid circular import
    from ..generator import DatasetGenerator

    config, split, episode_index, base_seed = args

    # Set deterministic mode for this worker
    seed = get_split_seed(base_seed, split, episode_index)
    device = config.get_device()
    SeedManager.set_deterministic_mode(seed, device)

    # Create generator instance for this worker
    generator = DatasetGenerator(config)
    generator.seed = seed

    # Generate single episode
    episode = generator.generate_episode()

    return {
        'split': split,
        'index': episode_index,
        'seed': seed,
        'frames': episode['frames'].cpu().numpy(),
        'metadata': episode['metadata']
    }


class ParallelGenerator:
    """Parallel generation manager using multiprocessing pool."""

    def __init__(
        self,
        config: DictConfig,
        num_workers: Optional[int] = None
    ):
        self.config = config
        self.num_workers = num_workers or config.parallel_workers or cpu_count()
        self.seed_manager = SeedManager(config.seed or 42)

    def generate(
        self,
        n_episodes: int,
        split: str = 'train',
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate multiple episodes in parallel."""
        args = [
            (self.config, split, i, self.config.seed or 42)
            for i in range(n_episodes)
        ]

        with Pool(processes=self.num_workers) as pool:
            if show_progress:
                episodes = list(tqdm(
                    pool.imap(generate_episode_worker, args),
                    total=n_episodes,
                    desc=f"Generating {split} episodes"
                ))
            else:
                episodes = pool.map(generate_episode_worker, args)

        return episodes

    def generate_splits(
        self,
        n_total: int,
        show_progress: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all splits (train/val/test) with configured ratios."""
        splits = {}
        splits['train'] = self.generate(
            int(n_total * self.config.split_config.train_ratio),
            split='train',
            show_progress=show_progress
        )
        splits['val'] = self.generate(
            int(n_total * self.config.split_config.val_ratio),
            split='val',
            show_progress=show_progress
        )
        splits['test'] = self.generate(
            int(n_total * self.config.split_config.test_ratio),
            split='test',
            show_progress=show_progress
        )

        return splits
