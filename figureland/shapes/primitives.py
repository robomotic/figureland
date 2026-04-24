"""
Shape primitives implementation with PyTorch tensor operations.
All shapes are implemented as pure PyTorch operations for GPU acceleration.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union, List


class Shape:
    """Base class for all shapes with batched PyTorch operations."""

    def __init__(
        self,
        position: torch.Tensor,
        size: torch.Tensor,
        rotation: torch.Tensor,
        color: torch.Tensor,
        mass: torch.Tensor,
        elasticity: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or position.device
        self.position = position.to(self.device)
        self.size = size.to(self.device)
        self.rotation = rotation.to(self.device)
        self.color = color.to(self.device)
        self.mass = mass.to(self.device)
        self.elasticity = elasticity.to(self.device)
        self.velocity = velocity.to(self.device) if velocity is not None else torch.zeros_like(position)

    @classmethod
    def from_random(
        cls,
        batch_size: int,
        bounds: Tuple[float, float],
        size_range: Tuple[float, float],
        mass_range: Tuple[float, float],
        elasticity_range: Tuple[float, float],
        seed: Optional[int] = None,
        device: Optional[torch.device] = None
    ) -> 'Shape':
        """Generate random shape instances in batch."""
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        # Generate size first
        size = torch.rand(batch_size, 2, device=device, generator=generator) * (size_range[1] - size_range[0]) + size_range[0]

        # Compute valid position range to keep shape fully within bounds
        min_bound, max_bound = bounds
        min_pos = min_bound + size
        max_pos = max_bound - size

        # Ensure min <= max to avoid errors if size too large
        min_pos = torch.min(min_pos, max_pos)
        max_pos = torch.max(max_pos, min_pos)

        position = torch.rand(batch_size, 2, device=device, generator=generator) * (max_pos - min_pos) + min_pos

        rotation = torch.rand(batch_size, device=device, generator=generator) * 2 * torch.pi
        color = torch.rand(batch_size, 3, device=device, generator=generator)
        mass = torch.rand(batch_size, device=device, generator=generator) * (mass_range[1] - mass_range[0]) + mass_range[0]
        elasticity = torch.rand(batch_size, device=device, generator=generator) * (elasticity_range[1] - elasticity_range[0]) + elasticity_range[0]

        return cls(position, size, rotation, color, mass, elasticity, device=device)

    def to(self, device: torch.device) -> 'Shape':
        """Move shape tensors to specified device."""
        return self.__class__(
            self.position.to(device),
            self.size.to(device),
            self.rotation.to(device),
            self.color.to(device),
            self.mass.to(device),
            self.elasticity.to(device),
            self.velocity.to(device),
            device=device
        )

    def render(self, resolution: Tuple[int, int], anti_alias: int = 2) -> torch.Tensor:
        """Render shape to tensor with anti-aliasing."""
        raise NotImplementedError("Subclasses must implement render method")

    def get_bounding_box(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (min_x, min_y, max_x, max_y) for each shape in batch."""
        raise NotImplementedError("Subclasses must implement get_bounding_box method")


class Square(Shape):
    """Square primitive with equal width and height."""

    def __init__(self, position: torch.Tensor, size: torch.Tensor, rotation: torch.Tensor,
                 color: torch.Tensor, mass: torch.Tensor, elasticity: torch.Tensor,
                 velocity: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        # Ensure square has equal width and height
        size = size[:, 0:1].repeat(1, 2)
        super().__init__(position, size, rotation, color, mass, elasticity, velocity, device)

    def render(self, resolution: Tuple[int, int], anti_alias: int = 2) -> torch.Tensor:
        batch_size = self.position.shape[0]
        h, w = resolution
        h_aa, w_aa = h * anti_alias, w * anti_alias

        # Create coordinate grid
        y = torch.linspace(-1, 1, h_aa, device=self.device)
        x = torch.linspace(-1, 1, w_aa, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Apply rotation and translation
        cos = torch.cos(self.rotation)
        sin = torch.sin(self.rotation)
        rot_mat = torch.stack([cos, -sin, sin, cos], dim=-1).view(batch_size, 2, 2)

        # Transform grid to shape local coordinates
        local_grid = grid - self.position.view(batch_size, 1, 1, 2)
        local_grid = torch.einsum('bij,bhwj->bhwi', rot_mat, local_grid)
        local_grid = local_grid / self.size.view(batch_size, 1, 1, 2)

        # SDF for square
        dist = torch.max(torch.abs(local_grid), dim=-1)[0]
        mask = dist <= 1.0

        # Apply color
        output = torch.zeros(batch_size, h_aa, w_aa, 3, device=self.device)
        output[mask] = self.color.unsqueeze(1).unsqueeze(1).repeat(1, h_aa, w_aa, 1)[mask]

        # Anti-alias downsampling
        if anti_alias > 1:
            output = output.permute(0, 3, 1, 2)
            output = F.avg_pool2d(output, kernel_size=anti_alias, stride=anti_alias)
            output = output.permute(0, 2, 3, 1)

        return output


class Rectangle(Shape):
    """Rectangle primitive with variable aspect ratio."""

    def render(self, resolution: Tuple[int, int], anti_alias: int = 2) -> torch.Tensor:
        batch_size = self.position.shape[0]
        h, w = resolution
        h_aa, w_aa = h * anti_alias, w * anti_alias

        y = torch.linspace(-1, 1, h_aa, device=self.device)
        x = torch.linspace(-1, 1, w_aa, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        cos = torch.cos(self.rotation)
        sin = torch.sin(self.rotation)
        rot_mat = torch.stack([cos, -sin, sin, cos], dim=-1).view(batch_size, 2, 2)

        local_grid = grid - self.position.view(batch_size, 1, 1, 2)
        local_grid = torch.einsum('bij,bhwj->bhwi', rot_mat, local_grid)
        local_grid = local_grid / self.size.view(batch_size, 1, 1, 2)

        dist = torch.max(torch.abs(local_grid), dim=-1)[0]
        mask = dist <= 1.0

        output = torch.zeros(batch_size, h_aa, w_aa, 3, device=self.device)
        output[mask] = self.color.unsqueeze(1).unsqueeze(1).repeat(1, h_aa, w_aa, 1)[mask]

        if anti_alias > 1:
            output = output.permute(0, 3, 1, 2)
            output = F.avg_pool2d(output, kernel_size=anti_alias, stride=anti_alias)
            output = output.permute(0, 2, 3, 1)

        return output


class Triangle(Shape):
    """Equilateral triangle primitive."""

    def render(self, resolution: Tuple[int, int], anti_alias: int = 2) -> torch.Tensor:
        batch_size = self.position.shape[0]
        h, w = resolution
        h_aa, w_aa = h * anti_alias, w * anti_alias

        y = torch.linspace(-1, 1, h_aa, device=self.device)
        x = torch.linspace(-1, 1, w_aa, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        cos = torch.cos(self.rotation)
        sin = torch.sin(self.rotation)
        rot_mat = torch.stack([cos, -sin, sin, cos], dim=-1).view(batch_size, 2, 2)

        local_grid = grid - self.position.view(batch_size, 1, 1, 2)
        local_grid = torch.einsum('bij,bhwj->bhwi', rot_mat, local_grid)
        local_grid = local_grid / self.size.view(batch_size, 1, 1, 2)

        # SDF for equilateral triangle
        px = local_grid[..., 0]
        py = local_grid[..., 1]

        k = 0.5773502691896257  # 1/sqrt(3)
        d = torch.abs(px + k * py) - k
        d = torch.maximum(d, -2 * k * py - k)
        d = torch.maximum(d, py - 1.0)
        dist = d * 0.8660254037844386  # sqrt(3)/2

        mask = dist <= 0.0

        output = torch.zeros(batch_size, h_aa, w_aa, 3, device=self.device)
        output[mask] = self.color.unsqueeze(1).unsqueeze(1).repeat(1, h_aa, w_aa, 1)[mask]

        if anti_alias > 1:
            output = output.permute(0, 3, 1, 2)
            output = F.avg_pool2d(output, kernel_size=anti_alias, stride=anti_alias)
            output = output.permute(0, 2, 3, 1)

        return output


class Hexagon(Shape):
    """Regular hexagon primitive."""

    def render(self, resolution: Tuple[int, int], anti_alias: int = 2) -> torch.Tensor:
        batch_size = self.position.shape[0]
        h, w = resolution
        h_aa, w_aa = h * anti_alias, w * anti_alias

        y = torch.linspace(-1, 1, h_aa, device=self.device)
        x = torch.linspace(-1, 1, w_aa, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        cos = torch.cos(self.rotation)
        sin = torch.sin(self.rotation)
        rot_mat = torch.stack([cos, -sin, sin, cos], dim=-1).view(batch_size, 2, 2)

        local_grid = grid - self.position.view(batch_size, 1, 1, 2)
        local_grid = torch.einsum('bij,bhwj->bhwi', rot_mat, local_grid)
        local_grid = local_grid / self.size.view(batch_size, 1, 1, 2)

        # SDF for hexagon
        px = torch.abs(local_grid[..., 0])
        py = torch.abs(local_grid[..., 1])

        d = torch.maximum(px * 0.8660254037844386 + py * 0.5, py) - 0.8660254037844386
        mask = d <= 0.0

        output = torch.zeros(batch_size, h_aa, w_aa, 3, device=self.device)
        output[mask] = self.color.unsqueeze(1).unsqueeze(1).repeat(1, h_aa, w_aa, 1)[mask]

        if anti_alias > 1:
            output = output.permute(0, 3, 1, 2)
            output = F.avg_pool2d(output, kernel_size=anti_alias, stride=anti_alias)
            output = output.permute(0, 2, 3, 1)

        return output


class Trapezoid(Shape):
    """Trapezoid primitive with variable top width."""

    def render(self, resolution: Tuple[int, int], anti_alias: int = 2) -> torch.Tensor:
        batch_size = self.position.shape[0]
        h, w = resolution
        h_aa, w_aa = h * anti_alias, w * anti_alias

        y = torch.linspace(-1, 1, h_aa, device=self.device)
        x = torch.linspace(-1, 1, w_aa, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        cos = torch.cos(self.rotation)
        sin = torch.sin(self.rotation)
        rot_mat = torch.stack([cos, -sin, sin, cos], dim=-1).view(batch_size, 2, 2)

        local_grid = grid - self.position.view(batch_size, 1, 1, 2)
        local_grid = torch.einsum('bij,bhwj->bhwi', rot_mat, local_grid)
        local_grid = local_grid / self.size.view(batch_size, 1, 1, 2)

        # SDF for trapezoid (top width = 0.5 * bottom width)
        px = local_grid[..., 0]
        py = local_grid[..., 1]

        top_width = 0.5
        dx = 1.0 - top_width
        d = torch.maximum(
            torch.abs(px) - (1.0 - dx * (py + 1.0) / 2.0),
            py - 1.0
        )
        d = torch.maximum(d, -py - 1.0)
        mask = d <= 0.0

        output = torch.zeros(batch_size, h_aa, w_aa, 3, device=self.device)
        output[mask] = self.color.unsqueeze(1).unsqueeze(1).repeat(1, h_aa, w_aa, 1)[mask]

        if anti_alias > 1:
            output = output.permute(0, 3, 1, 2)
            output = F.avg_pool2d(output, kernel_size=anti_alias, stride=anti_alias)
            output = output.permute(0, 2, 3, 1)

        return output


def generate_shape(
    shape_type: str,
    batch_size: int,
    bounds: Tuple[float, float],
    size_range: Tuple[float, float],
    mass_range: Tuple[float, float],
    elasticity_range: Tuple[float, float],
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Shape:
    """Generate a batch of shapes of specified type."""
    if shape_type not in SHAPE_REGISTRY:
        raise ValueError(f"Unknown shape type: {shape_type}. Available types: {list(SHAPE_REGISTRY.keys())}")

    return SHAPE_REGISTRY[shape_type].from_random(
        batch_size=batch_size,
        bounds=bounds,
        size_range=size_range,
        mass_range=mass_range,
        elasticity_range=elasticity_range,
        seed=seed,
        device=device
    )


def batch_generate_shapes(
    shape_types: List[str],
    batch_size: int,
    bounds: Tuple[float, float],
    size_range: Tuple[float, float],
    mass_range: Tuple[float, float],
    elasticity_range: Tuple[float, float],
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> List[Shape]:
    """Generate batch of mixed shape types."""
    shapes = []
    if seed is not None:
        # Derive deterministic per-shape seeds by adding index offset
        seeds = [seed + i for i in range(len(shape_types))]
    else:
        seeds = [None] * len(shape_types)

    for shape_type, s in zip(shape_types, seeds):
        shapes.append(generate_shape(
            shape_type,
            batch_size,
            bounds,
            size_range,
            mass_range,
            elasticity_range,
            seed=s,
            device=device
        ))

    return shapes


SHAPE_REGISTRY: Dict[str, Any] = {
    'square': Square,
    'rectangle': Rectangle,
    'triangle': Triangle,
    'hexagon': Hexagon,
    'trapezoid': Trapezoid
}
