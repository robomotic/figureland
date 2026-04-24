#!/usr/bin/env python3
"""
Parallel generation of bouncing ball simulations using multiprocessing.

Generates multiple independent simulations in parallel using standard library multiprocessing.
Each worker process generates a complete simulation with unique initial conditions.
"""

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple
from figureland.shapes import Square
from figureland.physics import Environment, PhysicsEngine
from figureland import SimulationExporter


def generate_single_simulation(args: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    """
    Worker function to generate a single simulation.

    Args:
        args: (simulation_id, seed)

    Returns:
        (simulation_id, video_frames)
    """
    sim_id, seed = args

    # Each worker gets its own seed for deterministic independent generation
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create simulation
    env = Environment(bounds=(-1.0, 1.0), gravity=9.8)
    engine = PhysicsEngine(env)
    exporter = SimulationExporter((100, 400), fps=30)

    # Generate 2 balls with random positions
    shapes = [
        Square.from_random(1, (-1,1), (0.1,0.1), (1,1), (0.8,0.8), seed=seed),
        Square.from_random(1, (-1,1), (0.1,0.1), (1,1), (0.8,0.8), seed=seed+1)
    ]

    # Random initial positions at top
    shapes[0].position[0] = torch.tensor([np.random.uniform(-0.8, -0.2), 0.8])
    shapes[1].position[0] = torch.tensor([np.random.uniform(0.2, 0.8), 0.8])

    # Random colors
    shapes[0].color[:] = torch.rand(3)
    shapes[1].color[:] = torch.rand(3)

    env.add_shapes(shapes)

    # Run simulation
    frames = []
    for _ in range(200):
        frames.append(exporter.render_frame(env.shapes))
        engine.step()

    return sim_id, np.array(frames)


def main():
    # Configuration
    NUM_SIMULATIONS = 8
    NUM_WORKERS = cpu_count()

    print(f"Generating {NUM_SIMULATIONS} bouncing ball simulations in parallel")
    print(f"Using {NUM_WORKERS} worker processes")

    # Generate unique seeds for each simulation
    base_seed = 42
    seeds = [base_seed + i for i in range(NUM_SIMULATIONS)]
    args = [(i, seeds[i]) for i in range(NUM_SIMULATIONS)]

    # Run in parallel with progress bar
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(generate_single_simulation, args),
            total=NUM_SIMULATIONS,
            desc="Generating simulations"
        ))

    # Save all results
    print("\nSaving results:")
    for sim_id, frames in results:
        exporter = SimulationExporter((100, 400), fps=30, output_dir=f"./output/sim_{sim_id}")
        video_path = exporter.save_video(f"simulation_{sim_id}.mp4", frames)
        print(f"  ✅ Saved simulation {sim_id}: {video_path}")

    print(f"\n✅ Completed! All {NUM_SIMULATIONS} simulations generated")
    print(f"✅ Outputs saved to ./output/ directory")


if __name__ == "__main__":
    main()
