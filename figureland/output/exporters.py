"""
Output exporters for all supported formats.
Uses OpenCV exclusively for video encoding per requirements.
"""

import os
import cv2
import numpy as np
import h5py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastavro
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from .codec import CodecDetector
from .config import VideoConfig


class ImageExporter:
    """Image exporter supporting PNG, JPEG, TIFF formats."""

    def __init__(self, output_dir: str, format: str = 'png'):
        self.output_dir = output_dir
        self.format = format.lower()
        os.makedirs(output_dir, exist_ok=True)

        if self.format not in ['png', 'jpeg', 'jpg', 'tiff']:
            raise ValueError(f"Unsupported image format: {format}")

    def save_frame(self, frame: np.ndarray, episode_id: int, frame_idx: int) -> str:
        """Save single frame to file."""
        filename = f"episode_{episode_id:06d}_frame_{frame_idx:06d}.{self.format}"
        path = os.path.join(self.output_dir, filename)

        # Convert from float [0,1] to uint8 [0,255]
        img = (frame * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

        return path

    def save_episode(self, episode: np.ndarray, episode_id: int) -> List[str]:
        """Save all frames from an episode."""
        paths = []
        for frame_idx in range(episode.shape[0]):
            paths.append(self.save_frame(episode[frame_idx], episode_id, frame_idx))
        return paths


class VideoExporter:
    """Video exporter using OpenCV exclusively. Supports MP4, AVI, GIF.
    
    Automatically detects available codecs and selects the best one
    if codec is not explicitly specified.
    """

    def __init__(
        self,
        output_dir: str,
        format: str = 'mp4',
        fps: int = 30,
        codec: Optional[str] = None,
        config: Optional[VideoConfig] = None
    ):
        self.output_dir = output_dir
        self.format = format.lower()
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

        # Use VideoConfig if provided, otherwise build from parameters
        if config is not None:
            self.video_config = config
        else:
            self.video_config = VideoConfig(
                codec=codec,
                format=self.format,
                fps=fps
            )
        
        # Resolve codec (auto-detect if not specified)
        self.codec, self.container_format = self.video_config.resolve_codec()
        
        if self.container_format == 'gif':
            self.fourcc = None  # Handled separately via PIL
        else:
            self.fourcc = cv2.VideoWriter_fourcc(*self.codec)
            if not self.fourcc:
                raise ValueError(f"Failed to create FourCC for codec: {self.codec}")

    def save_episode(self, episode: np.ndarray, episode_id: int) -> str:
        """Save episode as video file."""
        filename = f"episode_{episode_id:06d}.{self.container_format}"
        path = os.path.join(self.output_dir, filename)

        # Convert from float [0,1] to uint8 [0,255]
        video = (episode * 255).astype(np.uint8)
        n_frames, h, w, _ = video.shape

        if self.container_format == 'gif':
            # GIF export using PIL
            frames = [Image.fromarray(frame) for frame in video]
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // self.fps,
                loop=0
            )
        else:
            # Video export using OpenCV with detected codec
            writer = cv2.VideoWriter(path, self.fourcc, self.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(
                    f"Failed to open video writer with codec '{self.codec}'. "
                    f"Available codecs: {CodecDetector.detect_available_codecs()}"
                )
            for frame in video:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
            writer.release()

        return path


class H5Exporter:
    """HDF5 exporter with chunked compression."""

    def __init__(self, output_dir: str, compression: str = 'gzip'):
        self.output_dir = output_dir
        self.compression = compression
        os.makedirs(output_dir, exist_ok=True)

    def save_episode(self, episode: np.ndarray, metadata: Dict[str, Any], episode_id: int) -> str:
        """Save episode to HDF5 file."""
        filename = f"episode_{episode_id:06d}.h5"
        path = os.path.join(self.output_dir, filename)

        with h5py.File(path, 'w') as f:
            f.create_dataset('frames', data=episode, compression=self.compression)

            # Save metadata
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    meta_group.attrs[key] = value
                elif isinstance(value, np.ndarray):
                    meta_group.create_dataset(key, data=value, compression=self.compression)

        return path

    def save_dataset(self, episodes: List[Dict[str, Any]], dataset_name: str) -> str:
        """Save multiple episodes to a single HDF5 file."""
        path = os.path.join(self.output_dir, f"{dataset_name}.h5")

        with h5py.File(path, 'w') as f:
            n_episodes = len(episodes)
            n_frames = episodes[0]['frames'].shape[0]
            h, w, c = episodes[0]['frames'].shape[1:]

            frames_ds = f.create_dataset(
                'frames',
                (n_episodes, n_frames, h, w, c),
                dtype=np.float32,
                compression=self.compression,
                chunks=(1, n_frames, h, w, c)
            )

            for i, episode in enumerate(episodes):
                frames_ds[i] = episode['frames']

        return path


class ParquetExporter:
    """Parquet exporter for columnar storage."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_dataset(self, episodes: List[Dict[str, Any]], dataset_name: str) -> str:
        """Save dataset to Parquet file."""
        path = os.path.join(self.output_dir, f"{dataset_name}.parquet")

        data = []
        for episode in episodes:
            row = {
                'episode_id': episode['index'],
                'split': episode['split'],
                'seed': episode['seed'],
                'frames': episode['frames'].flatten().tolist()
            }
            # Add metadata fields
            row.update(episode['metadata'])
            data.append(row)

        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)

        return path


class AvroExporter:
    """Avro binary serializer."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_dataset(self, episodes: List[Dict[str, Any]], dataset_name: str) -> str:
        """Save dataset to Avro file."""
        path = os.path.join(self.output_dir, f"{dataset_name}.avro")

        # Define Avro schema
        schema = {
            'type': 'record',
            'name': 'Episode',
            'fields': [
                {'name': 'episode_id', 'type': 'int'},
                {'name': 'split', 'type': 'string'},
                {'name': 'seed', 'type': 'int'},
                {'name': 'frames_shape', 'type': {'type': 'array', 'items': 'int'}},
                {'name': 'frames', 'type': {'type': 'array', 'items': 'float'}}
            ]
        }

        parsed_schema = fastavro.parse_schema(schema)

        records = []
        for episode in episodes:
            records.append({
                'episode_id': episode['index'],
                'split': episode['split'],
                'seed': episode['seed'],
                'frames_shape': list(episode['frames'].shape),
                'frames': episode['frames'].flatten().tolist()
            })

        with open(path, 'wb') as f:
            fastavro.writer(f, parsed_schema, records)

        return path
