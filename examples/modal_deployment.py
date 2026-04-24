"""
Modal cloud deployment example.
Run with: modal run examples/modal_deployment.py
"""

import modal
import torch

stub = modal.Stub("figureland-generator")

# Create Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.2.0",
    "numpy>=1.26.0",
    "opencv-python>=4.8.0",
    "h5py>=3.10.0",
    "pandas>=2.1.0",
    "pyarrow>=15.0.0",
    "fastavro>=1.9.0",
    "pillow>=10.2.0",
    "tqdm>=4.66.0",
    "pydantic>=2.5.0"
)


@stub.function(
    image=image,
    gpu="T4",
    timeout=3600,
    concurrency_limit=10
)
def generate_episode_modal(episode_index: int, split: str, base_seed: int):
    """Stateless worker function for Modal."""
    from figureland import DatasetGenerator
    from figureland.config import GeneratorConfig

    config = GeneratorConfig(
        resolution=(256, 256),
        episode_length=100,
        batch_size=16,
        use_gpu=True,
        seed=base_seed
    )

    generator = DatasetGenerator(config)
    episode = generator.generate_episode(seed=base_seed + episode_index)

    return {
        'index': episode_index,
        'split': split,
        'frames': episode['frames'].cpu().numpy(),
        'metadata': episode['metadata']
    }


@stub.function(
    image=image,
    timeout=7200
)
def generate_dataset_modal(n_episodes: int = 1000):
    """Generate full dataset using Modal auto-scaling."""
    from tqdm import tqdm
    from figureland.output import H5Exporter

    base_seed = 42

    # Generate training episodes
    episodes = list(tqdm(
        generate_episode_modal.starmap(
            [(i, 'train', base_seed) for i in range(n_episodes)]
        ),
        total=n_episodes,
        desc="Generating dataset on Modal"
    ))

    # Save dataset
    exporter = H5Exporter('./output')
    path = exporter.save_dataset(episodes, 'modal_dataset')

    print(f"Generated {n_episodes} episodes")
    print(f"Dataset saved to: {path}")

    return path


if __name__ == "__main__":
    with stub.run():
        generate_dataset_modal.remote(100)
