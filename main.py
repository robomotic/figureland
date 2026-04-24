#!/usr/bin/env python3
"""
Figureland main entry point with Hydra integration.
"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from figureland import DatasetGenerator
from figureland.config import validate_config, print_config
from figureland.parallel import ParallelGenerator
from figureland.output import ImageExporter, VideoExporter, H5Exporter

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for Figureland dataset generator."""
    hydra_cfg = HydraConfig.get()
    log.info(f"Hydra run directory: {hydra_cfg.runtime.output_dir}")

    # Print configuration
    if log.isEnabledFor(logging.DEBUG):
        print_config(cfg)

    # Validate configuration
    validate_config(cfg)

    # Initialize generator
    generator = DatasetGenerator(cfg)

    # Run generation
    if cfg.get('parallel_generation', False):
        log.info(f"Running parallel generation with {cfg.parallel_workers} workers")
        parallel_gen = ParallelGenerator(cfg, num_workers=cfg.parallel_workers)

        n_total = cfg.get('n_episodes', 1000)
        log.info(f"Generating {n_total} total episodes")

        dataset = parallel_gen.generate_splits(n_total=n_total)

        log.info(f"Generated {len(dataset['train'])} training episodes")
        log.info(f"Generated {len(dataset['val'])} validation episodes")
        log.info(f"Generated {len(dataset['test'])} test episodes")

        # Export datasets
        if 'h5' in cfg.output.formats:
            log.info("Exporting to HDF5 format")
            h5_exporter = H5Exporter(cfg.output_dir)
            h5_exporter.save_dataset(dataset['train'], 'train')
            h5_exporter.save_dataset(dataset['val'], 'val')
            h5_exporter.save_dataset(dataset['test'], 'test')

    else:
        log.info("Running single episode generation")
        episode = generator.generate_episode()
        log.info(f"Generated episode with {episode['frames'].shape[0]} frames")

        # Export episode
        if 'png' in cfg.output.formats:
            log.info("Exporting frames to PNG")
            img_exporter = ImageExporter(cfg.output_dir, format='png')
            img_exporter.save_episode(episode['frames'][:, 0].cpu().numpy(), 0)

        if 'mp4' in cfg.output.formats or 'video' in cfg.output.formats:
            log.info("Exporting video")
            video_exporter = VideoExporter(cfg.output_dir, format='mp4', fps=cfg.output.video_fps)
            video_exporter.save_episode(episode['frames'][:, 0].cpu().numpy(), 0)

    log.info("Generation completed successfully!")


if __name__ == "__main__":
    main()
