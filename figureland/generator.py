"""
Main dataset generator class.
Integrates all components into a single interface.
"""

import logging
import torch
from typing import List, Dict, Any, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from .config import validate_config, get_device
from .shapes import batch_generate_shapes
from .physics import PhysicsEngine, Environment
from .rendering import Renderer
from .parallel.seed import SeedManager

log = logging.getLogger(__name__)


class DatasetGenerator:
    """Main dataset generator interface."""

    def __init__(self, cfg: Union[DictConfig, dict]):
        # Convert plain dict to OmegaConf if needed
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        validate_config(cfg)
        self.cfg = cfg
        self.device = get_device(cfg)
        self.seed = cfg.seed or 42

        log.info(f"Initializing DatasetGenerator on {self.device}")

        # Initialize components
        self.environment = Environment(
            bounds=tuple(cfg.physics.bounds),
            environment_type=cfg.physics.environment_type,
            gravity=cfg.physics.gravity,
            friction=cfg.physics.friction,
            air_resistance=cfg.physics.air_resistance,
            force_field=cfg.physics.force_field,
            device=self.device
        )
        log.debug("Environment initialized")

        self.physics_engine = PhysicsEngine(
            environment=self.environment,
            dt=cfg.physics.dt,
            substeps=cfg.physics.substeps,
            collisions_enabled=cfg.physics.collisions_enabled
        )
        log.debug("Physics engine initialized")

        self.renderer = Renderer(
            resolution=tuple(cfg.resolution),
            anti_alias=cfg.anti_alias,
            background_image=cfg.get('background_image', None),
            tile_background=cfg.get('tile_background', False),
            device=self.device
        )
        log.debug("Renderer initialized")

        self.seed_manager = SeedManager(self.seed)
        log.info("DatasetGenerator initialized successfully")

    def generate_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a single full episode."""
        episode_seed = seed or self.seed_manager.get_seed('train')
        log.debug(f"Generating episode with seed: {episode_seed}")

        SeedManager.set_deterministic_mode(episode_seed, self.device)

        # Clear any existing shapes
        self.environment.clear_shapes()

        # Generate shapes
        shapes = batch_generate_shapes(
            shape_types=self.cfg.shapes.shape_types,
            batch_size=self.cfg.batch_size,
            bounds=tuple(self.cfg.physics.bounds),
            size_range=tuple(self.cfg.shapes.size_range),
            mass_range=tuple(self.cfg.shapes.mass_range),
            elasticity_range=tuple(self.cfg.shapes.elasticity_range),
            seed=episode_seed,
            device=self.device
        )

        # Initialize random velocities
        for shape in shapes:
            shape.velocity = torch.randn_like(shape.position) * 0.5

        # Add shapes to environment (strict=False allows overlapping during generation,
        # physics will resolve collisions; bounds are still enforced by from_random)
        self.environment.add_shapes(shapes, strict=False)

        # Simulate episode
        shape_history = []
        velocity_history = []
        position_history = []

        log.debug(f"Simulating {self.cfg.episode_length} frames")
        for _ in range(self.cfg.episode_length):
            # Save current state as copies
            current_shapes = []
            for shape in self.environment.shapes:
                current_shapes.append(type(shape)(
                    position=shape.position.clone(),
                    size=shape.size.clone(),
                    rotation=shape.rotation.clone(),
                    color=shape.color.clone(),
                    mass=shape.mass.clone(),
                    elasticity=shape.elasticity.clone(),
                    velocity=shape.velocity.clone(),
                    device=self.device
                ))
            shape_history.append(current_shapes)

            velocity_history.append(torch.cat([s.velocity for s in self.environment.shapes], dim=1))
            position_history.append(torch.cat([s.position for s in self.environment.shapes], dim=1))

            # Step physics - uses shapes from environment
            self.physics_engine.step()

        # Render episode
        frames = self.renderer.render_episode(shape_history)

        # Collect metadata
        metadata = {
            'seed': episode_seed,
            'episode_length': self.cfg.episode_length,
            'num_shapes': len(shapes),
            'resolution': tuple(self.cfg.resolution),
            'velocity_history': torch.stack(velocity_history).cpu().numpy(),
            'position_history': torch.stack(position_history).cpu().numpy()
        }

        log.debug(f"Episode generated successfully with {len(frames)} frames")

        return {
            'frames': frames,
            'metadata': metadata,
            'shape_history': shape_history
        }

    def generate(self, n_episodes: int, split: str = 'train') -> Dict[str, Any]:
        """Generate multiple episodes sequentially."""
        episodes = []
        for i in range(n_episodes):
            seed = self.seed_manager.get_seed(split, i)
            episodes.append(self.generate_episode(seed))

        return {
            'split': split,
            'n_episodes': n_episodes,
            'episodes': episodes
        }

    def to(self, device: Union[torch.device, str]) -> 'DatasetGenerator':
        """Move generator to specified device."""
        device = torch.device(device) if isinstance(device, str) else device
        self.device = device
        self.environment = self.environment.to(device)
        self.physics_engine = self.physics_engine.to(device)
        self.renderer = self.renderer.to(device)
        return self
