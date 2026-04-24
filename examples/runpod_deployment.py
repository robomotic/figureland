"""
RunPod deployment example.
"""

import os
import torch
from figureland import DatasetGenerator
from figureland.config import GeneratorConfig
from figureland.parallel import ParallelGenerator
from figureland.output import H5Exporter


def generate_dataset_runpod():
    """Generate dataset on RunPod GPU instance."""
    print("Running Figureland dataset generation on RunPod...")

    # Get RunPod environment variables
    gpu_memory = os.environ.get('RUNPOD_GPU_MEMORY', '16')
    instance_id = os.environ.get('RUNPOD_POD_ID', 'local')

    print(f"Running on RunPod instance: {instance_id}")
    print(f"GPU Memory: {gpu_memory}GB")

    # Adjust batch size based on available GPU memory
    gpu_memory_gb = int(gpu_memory)
    if gpu_memory_gb >= 40:
        batch_size = 128
    elif gpu_memory_gb >= 24:
        batch_size = 64
    elif gpu_memory_gb >= 16:
        batch_size = 32
    else:
        batch_size = 16

    print(f"Using batch size: {batch_size}")

    config = GeneratorConfig(
        resolution=(256, 256),
        episode_length=200,
        batch_size=batch_size,
        use_gpu=True,
        parallel_workers=os.cpu_count(),
        seed=42
    )

    # Generate 10,000 episodes
    parallel_gen = ParallelGenerator(config)
    dataset = parallel_gen.generate(n_episodes=10000, split='train')

    # Export dataset
    exporter = H5Exporter('/workspace/output')
    path = exporter.save_dataset(dataset, 'train_dataset_10k')

    print(f"Dataset generated successfully!")
    print(f"Saved to: {path}")
    print(f"Total episodes: {len(dataset)}")

    return path


if __name__ == "__main__":
    generate_dataset_runpod()
