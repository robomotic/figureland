"""
Basic usage example for Figureland dataset generator.
"""

import torch
from figureland import DatasetGenerator
from figureland.config import GeneratorConfig, ShapeConfig, PhysicsConfig, SplitConfig
from figureland.parallel import ParallelGenerator
from figureland.output import ImageExporter, VideoExporter, H5Exporter


def basic_example():
    """Basic single episode generation example."""
    print("Running basic episode generation example...")

    config = GeneratorConfig(
        resolution=(256, 256),
        episode_length=50,
        batch_size=1,
        shape_config=ShapeConfig(
            shape_types=['square', 'triangle', 'circle'],
            size_range=(0.1, 0.2)
        ),
        physics_config=PhysicsConfig(
            gravity=9.8,
            collisions_enabled=True
        )
    )

    generator = DatasetGenerator(config)
    print(f"Using device: {generator.device}")

    episode = generator.generate_episode()
    print(f"Generated episode with {episode['frames'].shape[0]} frames")
    print(f"Frame shape: {episode['frames'].shape[2:]}")

    # Export frames
    exporter = ImageExporter('./output/basic', format='png')
    exporter.save_episode(episode['frames'].cpu().numpy(), 0)
    print("Saved frames to ./output/basic")


def parallel_example():
    """Parallel generation example with train/val/test splits."""
    print("\nRunning parallel generation example...")

    config = GeneratorConfig(
        resolution=(128, 128),
        episode_length=100,
        batch_size=4,
        parallel_workers=4,
        split_config=SplitConfig(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
    )

    parallel_gen = ParallelGenerator(config, num_workers=4)
    dataset = parallel_gen.generate_splits(n_total=10)

    print(f"Generated {len(dataset['train'])} training episodes")
    print(f"Generated {len(dataset['val'])} validation episodes")
    print(f"Generated {len(dataset['test'])} test episodes")

    # Export as HDF5
    h5_exporter = H5Exporter('./output/parallel')
    h5_exporter.save_dataset(dataset['train'], 'train_dataset')
    print("Saved training dataset to ./output/parallel/train_dataset.h5")


def gpu_example():
    """GPU accelerated batch generation example."""
    print("\nRunning GPU acceleration example...")

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        config = GeneratorConfig(
            resolution=(512, 512),
            episode_length=200,
            batch_size=32,
            use_gpu=True
        )

        generator = DatasetGenerator(config)
        print(f"Running on GPU: {generator.device}")

        episode = generator.generate_episode()
        print(f"Generated {episode['frames'].shape[0]} frames with batch size 32")
        print(f"Total frames generated: {episode['frames'].shape[0] * 32}")

        # Export video
        video_exporter = VideoExporter('./output/gpu', format='mp4', fps=30)
        video_exporter.save_episode(episode['frames'][:, 0].cpu().numpy(), 0)
        print("Saved example video to ./output/gpu/episode_000000.mp4")
    else:
        print("GPU not available, skipping GPU example")


if __name__ == "__main__":
    basic_example()
    parallel_example()
    gpu_example()
