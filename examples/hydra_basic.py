"""
Basic Hydra usage example.
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Running Figureland with Hydra configuration")

    # Override parameters from code
    cfg.resolution = [512, 512]
    cfg.episode_length = 200
    cfg.batch_size = 16

    from figureland import DatasetGenerator
    generator = DatasetGenerator(cfg)

    episode = generator.generate_episode()
    log.info(f"Generated episode shape: {episode['frames'].shape}")


if __name__ == "__main__":
    main()
