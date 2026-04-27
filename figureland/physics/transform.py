"""
Coordinate transformation utilities.

Handles bidirectional conversion between physics space and pixel space.
Physics space uses normalized coordinates (typically -1 to 1).
Pixel space uses integer pixel coordinates (0 to resolution).
"""

import torch
from typing import Tuple, Optional


class CoordinateTransform:
    """Handles bidirectional conversion between physics space and pixel space.
    
    Physics space: normalized coordinates, typically (-1, 1) for both axes
    Pixel space: integer pixel coordinates (0, width) x (0, height)
    
    Attributes:
        physics_min: Minimum physics coordinate value
        physics_max: Maximum physics coordinate value
        height: Output image height in pixels
        width: Output image width in pixels
        flip_y: If True, flip Y axis for image coordinates (top-left origin)
    """
    
    def __init__(
        self,
        physics_bounds: Tuple[float, float] = (-1.0, 1.0),
        resolution: Tuple[int, int] = (256, 256),
        flip_y: bool = True
    ):
        """
        Initialize coordinate transform.
        
        Args:
            physics_bounds: (min, max) physics coordinate range
            resolution: (height, width) output resolution in pixels
            flip_y: If True, flip Y axis for image coordinates
        """
        self.physics_min, self.physics_max = physics_bounds
        self.physics_range = physics_bounds[1] - physics_bounds[0]
        self.height, self.width = resolution
        self.flip_y = flip_y
    
    def physics_to_pixel_x(self, physics_x: float) -> int:
        """Convert physics X coordinate to pixel X.
        
        Args:
            physics_x: X coordinate in physics space
            
        Returns:
            X coordinate in pixel space
        """
        normalized = (physics_x - self.physics_min) / self.physics_range
        return int(normalized * self.width)
    
    def physics_to_pixel_y(self, physics_y: float) -> int:
        """Convert physics Y coordinate to pixel Y.
        
        Args:
            physics_y: Y coordinate in physics space
            
        Returns:
            Y coordinate in pixel space
        """
        normalized = (physics_y - self.physics_min) / self.physics_range
        if self.flip_y:
            normalized = 1.0 - normalized
        return int(normalized * self.height)
    
    def physics_to_pixel(self, position: torch.Tensor) -> torch.Tensor:
        """Convert batch of physics positions to pixel positions.
        
        Args:
            position: Tensor of shape (batch_size, 2) with (x, y) physics coords
            
        Returns:
            Tensor of shape (batch_size, 2) with (x, y) pixel coords
        """
        x = ((position[:, 0] - self.physics_min) / self.physics_range) * self.width
        y = ((position[:, 1] - self.physics_min) / self.physics_range) * self.height
        if self.flip_y:
            y = self.height - y
        return torch.stack([x, y], dim=1)
    
    def physics_size_to_pixel(self, physics_size: float) -> int:
        """Convert physics size to pixel size.
        
        Args:
            physics_size: Size in physics space
            
        Returns:
            Size in pixel space
        """
        return int(physics_size * (self.height / 2))
    
    def pixel_to_physics_x(self, pixel_x: int) -> float:
        """Convert pixel X coordinate to physics X.
        
        Args:
            pixel_x: X coordinate in pixel space
            
        Returns:
            X coordinate in physics space
        """
        normalized = pixel_x / self.width
        return normalized * self.physics_range + self.physics_min
    
    def pixel_to_physics_y(self, pixel_y: int) -> float:
        """Convert pixel Y coordinate to physics Y.
        
        Args:
            pixel_y: Y coordinate in pixel space
            
        Returns:
            Y coordinate in physics space
        """
        normalized = pixel_y / self.height
        if self.flip_y:
            normalized = 1.0 - normalized
        return normalized * self.physics_range + self.physics_min
    
    def pixel_to_physics(self, position: torch.Tensor) -> torch.Tensor:
        """Convert batch of pixel positions to physics positions.
        
        Args:
            position: Tensor of shape (batch_size, 2) with (x, y) pixel coords
            
        Returns:
            Tensor of shape (batch_size, 2) with (x, y) physics coords
        """
        x = (position[:, 0] / self.width) * self.physics_range + self.physics_min
        y = (position[:, 1] / self.height) * self.physics_range + self.physics_min
        if self.flip_y:
            y = self.physics_max - (y - self.physics_min)
        return torch.stack([x, y], dim=1)
    
    def create_pixel_grid(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create a pixel coordinate grid mapped to physics space.
        
        Creates a grid where each pixel location is mapped to its corresponding
        physics coordinate. Used by Renderer for SDF-based rendering.
        
        Args:
            device: Target device for the grid tensor
            
        Returns:
            Tensor of shape (height, width, 2) with physics coordinates
        """
        if device is None:
            device = torch.device('cpu')
        
        # Create normalized coordinates from -1 to 1
        # For Y axis: image coordinates go top-to-bottom, so physics goes max-to-min
        if self.flip_y:
            y = torch.linspace(self.physics_max, self.physics_min, self.height, device=device)
        else:
            y = torch.linspace(self.physics_min, self.physics_max, self.height, device=device)
        x = torch.linspace(self.physics_min, self.physics_max, self.width, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        return grid
    
    def __repr__(self) -> str:
        return (
            f"CoordinateTransform(bounds=({self.physics_min}, {self.physics_max}), "
            f"resolution=({self.height}, {self.width}), flip_y={self.flip_y})"
        )
