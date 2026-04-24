"""
Batch renderer using pure PyTorch operations.
Renders multiple shapes and environments simultaneously.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from ..shapes import Shape


class Renderer:
    """Batched renderer with anti-aliasing support."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        anti_alias: int = 2,
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        background_image: Optional[str] = None,
        tile_background: bool = False,
        device: Optional[torch.device] = None
    ):
        self.resolution = resolution
        self.anti_alias = anti_alias
        self.background_color = torch.tensor(background_color, device=device)
        self.background_image = None
        self.tile_background = tile_background
        self.device = device or torch.device('cpu')

        if background_image is not None:
            self.set_background_image(background_image, tile_background)

    def set_background_image(self, image_path: str, tile: bool = False) -> None:
        """Load and set a background image.
        
        Args:
            image_path: Path to the image file
            tile: If True, tile the image to fill the background. If False, stretch to fit.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is required for background images. Install with: pip install pillow")
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Background image not found: {image_path}")
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        if tile:
            # For tiling, we'll store the original image and tile during rendering
            self._bg_image_raw = img
            self.tile_background = True
            self.background_image = None
        else:
            # Stretch to fit resolution
            img_resized = img.resize((self.resolution[1], self.resolution[0]), Image.Resampling.LANCZOS)
            img_tensor = torch.tensor(np.array(img_resized), device=self.device).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
            self.background_image = img_tensor
            self.tile_background = False

    def _create_tiled_background(self, batch_size: int) -> torch.Tensor:
        """Create a tiled background for the given resolution."""
        import numpy as np
        from PIL import Image
        
        img = self._bg_image_raw
        img_w, img_h = img.size
        res_h, res_w = self.resolution
        
        # Create a tiled image
        tiled = Image.new('RGB', (res_w, res_h))
        for y in range(0, res_h, img_h):
            for x in range(0, res_w, img_w):
                tiled.paste(img, (x, y))
        
        img_tensor = torch.tensor(np.array(tiled), device=self.device).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
        # Expand to batch dimension: (1, C, H, W) -> (batch_size, C, H, W)
        return img_tensor.unsqueeze(0).expand(batch_size, *img_tensor.shape)

    def render(self, shapes: List[Shape]) -> torch.Tensor:
        """Render a list of shapes into a single frame in batch."""
        batch_size = shapes[0].position.shape[0] if shapes else 1

        # Initialize background
        if self.tile_background and hasattr(self, '_bg_image_raw'):
            # Tiled background
            bg = self._create_tiled_background(batch_size)
            frames = bg.permute(0, 2, 3, 1)  # BCHW to BHWC
        elif self.background_image is not None:
            # Single (stretched) background image
            bg = self.background_image.unsqueeze(0).expand(batch_size, *self.background_image.shape)
            frames = bg.permute(0, 2, 3, 1)  # BCHW to BHWC
        else:
            # Solid color background
            frames = torch.ones(batch_size, self.resolution[0], self.resolution[1], 3, device=self.device)
            frames *= self.background_color.view(1, 1, 1, 3)

        # Render shapes in order (back to front)
        for shape in shapes:
            shape_frames = shape.render(self.resolution, self.anti_alias)

            # Alpha compositing
            mask = torch.sum(shape_frames, dim=-1) > 0
            frames[mask] = shape_frames[mask]

        return frames

    def render_episode(self, shape_history: List[List[Shape]]) -> torch.Tensor:
        """Render full episode from shape history."""
        n_frames = len(shape_history)
        batch_size = shape_history[0][0].position.shape[0]

        episode = torch.zeros(
            n_frames,
            batch_size,
            self.resolution[0],
            self.resolution[1],
            3,
            device=self.device
        )

        for frame_idx, shapes in enumerate(shape_history):
            episode[frame_idx] = self.render(shapes)

        return episode

    def to(self, device: torch.device) -> 'Renderer':
        """Move renderer to specified device."""
        new_renderer = Renderer(
            resolution=self.resolution,
            anti_alias=self.anti_alias,
            background_color=tuple(self.background_color.cpu().numpy()),
            device=device
        )
        # Move background image to new device if it exists
        if self.background_image is not None and not self.tile_background:
            new_renderer.background_image = self.background_image.to(device)
        elif self.tile_background:
            new_renderer._bg_image_raw = self._bg_image_raw
            new_renderer.tile_background = True
            new_renderer.background_image = None
        return new_renderer
