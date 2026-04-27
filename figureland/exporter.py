"""
High-level exporter for simulation output.
Handles all coordinate conversion, color space conversion, and export logic.
"""

import cv2
import numpy as np
import os
import torch
import json
from omegaconf import OmegaConf
from typing import List, Optional, Tuple, Union, Dict, Any
from .shapes import Shape
from .output.codec import CodecDetector
from .physics.transform import CoordinateTransform


class SimulationExporter:
    """
    High-level exporter that handles all rendering and export logic.

    Automatically handles:
    - Coordinate system conversion from physics space to pixel space
    - RGB to BGR conversion for OpenCV
    - Video encoding with automatic codec detection
    - Frame buffer management
    - Image and video export
    - Automatic Hydra config saving for reproducibility
    """

    def __init__(
        self,
        resolution: Tuple[int, int],
        output_dir: str = "./output",
        fps: int = 30,
        codec: Optional[str] = None,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        cfg: Any = None
    ):
        """
        Initialize exporter.

        Args:
            resolution: (height, width) of output
            output_dir: Directory to save outputs
            fps: Frames per second for video
            codec: Video codec to use. If None, auto-detects best codec.
            bounds: Physics coordinate bounds (min, max)
            cfg: Hydra config object to save for reproducibility
        """
        self.height, self.width = resolution
        self.output_dir = output_dir
        self.fps = fps
        self.cfg = cfg
        self.bounds = bounds
        self.frames: List[np.ndarray] = []

        os.makedirs(output_dir, exist_ok=True)

        # Create coordinate transform for physics-to-pixel conversion
        self.transform = CoordinateTransform(
            physics_bounds=bounds,
            resolution=resolution
        )

        # Auto-detect codec if not specified
        if codec is None:
            self.codec, self.container_format = CodecDetector.get_best_codec('mp4')
        else:
            self.codec = codec
            self.container_format = 'mp4'
        
        self.fourcc = cv2.VideoWriter_fourcc(*self.codec)

        # Save config immediately if provided
        if cfg is not None:
            self.save_config()

    def render_frame(self, shapes: List[Shape], batch_idx: int = None) -> np.ndarray:
        """
        Render a single frame from simulation shapes.

        Handles all coordinate and color conversion automatically.

        Args:
            shapes: List of Shape objects from simulation
            batch_idx: Batch index to render (default: 0)

        Returns:
            Rendered frame as uint8 numpy array
        """
        if batch_idx is None:
            batch_idx = 0

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for shape in shapes:
            # Convert physics coordinates to pixel coordinates using CoordinateTransform
            x = self.transform.physics_to_pixel_x(shape.position[batch_idx, 0].item())
            y = self.transform.physics_to_pixel_y(shape.position[batch_idx, 1].item())
            size = self.transform.physics_size_to_pixel(shape.size[batch_idx, 0].item())

            # Convert RGB to BGR for OpenCV
            color = (
                int(shape.color[batch_idx, 2].item() * 255),
                int(shape.color[batch_idx, 1].item() * 255),
                int(shape.color[batch_idx, 0].item() * 255),
            )

            # Draw shape
            cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, -1)

        return frame

    def add_frame(self, shapes: List[Shape]) -> None:
        """Add frame to internal buffer for video export."""
        self.frames.append(self.render_frame(shapes))

    def save_frame(self, shapes: List[Shape], filename: str) -> str:
        """Save single frame to image file."""
        frame = self.render_frame(shapes)
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def save_video(self, filename: str, frames: Optional[List[np.ndarray]] = None) -> str:
        """
        Save buffered frames to video file using auto-detected codec.
        
        Automatically saves config if available.

        Args:
            filename: Output filename
            frames: Frames to save (uses buffer if None)

        Returns:
            Path to saved video file
        """
        if frames is None:
            frames = self.frames

        if len(frames) == 0:
            raise ValueError("No frames to export")

        path = os.path.join(self.output_dir, filename)
        writer = cv2.VideoWriter(path, self.fourcc, self.fps, (self.width, self.height))

        if not writer.isOpened():
            available = CodecDetector.detect_available_codecs()
            raise RuntimeError(
                f"Failed to open video writer with codec '{self.codec}'. "
                f"Available codecs: {available}"
            )

        for frame in frames:
            writer.write(frame)

        writer.release()
        self.frames.clear()

        # Always save config when saving video if available
        if self.cfg is not None:
            self.save_config()

        return path

    def save_config(self, cfg: Any = None, filename: str = "config.yaml") -> str:
        """
        Save Hydra configuration to output directory for reproducibility.

        Args:
            cfg: Hydra config object (uses self.cfg if not provided)
            filename: Output filename

        Returns:
            Path to saved config file
        """
        config = cfg or self.cfg
        if config is None:
            raise ValueError("No config provided to save")

        path = os.path.join(self.output_dir, filename)
        OmegaConf.save(config, path)
        return path

    def save_metadata(self, metadata: Dict[str, Any], filename: str = "metadata.json") -> str:
        """
        Save additional metadata as JSON.

        Args:
            metadata: Dictionary of metadata to save
            filename: Output filename

        Returns:
            Path to saved metadata file
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return path

    def clear(self) -> None:
        """Clear internal frame buffer."""
        self.frames.clear()
