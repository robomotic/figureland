#!/usr/bin/env python3
"""
Example with automatic Hydra config saving.

Config is automatically saved alongside video for full reproducibility.
"""

import hydra
from omegaconf import DictConfig
from figureland import SimulationExporter
from figureland.shapes import batch_generate_shapes
from figureland.physics import PhysicsEngine, Environment


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize exporter WITH Hydra config - config will be saved automatically
    exporter = SimulationExporter(
        resolution=tuple(cfg.resolution),
        fps=cfg.fps,
        cfg=cfg  # Pass Hydra config here, it will be saved automatically
    )

    print(f"Running with config automatically saved to {exporter.output_dir}/config.yaml")

    # Initialize physics
    env = Environment(
        bounds=tuple(cfg.physics.bounds),
        gravity=cfg.physics.gravity
    )
    engine = PhysicsEngine(env, dt=cfg.physics.dt)

    # Create shapes
    shapes = batch_generate_shapes(
        shape_types=cfg.shapes.shape_types,
        batch_size=1,
        bounds=tuple(cfg.physics.bounds),
        size_range=tuple(cfg.shapes.size_range),
        mass_range=tuple(cfg.shapes.mass_range),
        elasticity_range=tuple(cfg.shapes.elasticity_range),
        seed=cfg.seed
    )

    # Position shapes
    for i, shape in enumerate(shapes):
        shape.position[0, 0] = -0.5 + (i * 1.0)
        shape.position[0, 1] = 0.8
        shape.velocity[:] = 0.0

    # Run simulation
    for _ in range(cfg.episode_length):
        exporter.add_frame(shapes)
        shapes, _ = engine.step(shapes)

    # Save video - config.yaml is saved automatically
    video_path = exporter.save_video("simulation.mp4")
    print(f"✅ Video saved: {video_path}")
    print(f"✅ Config auto-saved: {exporter.output_dir}/config.yaml")
    print(f"✅ 100% reproducible - just run with the saved config")


if __name__ == "__main__":
    main()
