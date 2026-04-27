"""
Tests for coordinate transformation system.
"""

import pytest
import torch
from figureland.physics.transform import CoordinateTransform


class TestCoordinateTransform:
    """Tests for CoordinateTransform class."""

    def test_creation_default(self):
        """Test default CoordinateTransform creation."""
        transform = CoordinateTransform()
        assert transform.physics_min == -1.0
        assert transform.physics_max == 1.0
        assert transform.height == 256
        assert transform.width == 256
        assert transform.flip_y is True

    def test_creation_custom(self):
        """Test custom CoordinateTransform creation."""
        transform = CoordinateTransform(
            physics_bounds=(-2.0, 2.0),
            resolution=(100, 200),
            flip_y=False
        )
        assert transform.physics_min == -2.0
        assert transform.physics_max == 2.0
        assert transform.height == 100
        assert transform.width == 200
        assert transform.flip_y is False

    def test_physics_to_pixel_x_edges(self):
        """Test physics to pixel X conversion at edges."""
        transform = CoordinateTransform(resolution=(100, 100))
        assert transform.physics_to_pixel_x(-1.0) == 0
        assert transform.physics_to_pixel_x(0.0) == 50
        assert transform.physics_to_pixel_x(1.0) == 100

    def test_physics_to_pixel_y_edges(self):
        """Test physics to pixel Y conversion at edges (with flip)."""
        transform = CoordinateTransform(resolution=(100, 100))
        # With flip_y=True, physics +1 should map to pixel 0 (top)
        assert transform.physics_to_pixel_y(1.0) == 0
        assert transform.physics_to_pixel_y(0.0) == 50
        assert transform.physics_to_pixel_y(-1.0) == 100

    def test_physics_to_pixel_y_no_flip(self):
        """Test physics to pixel Y conversion without flip."""
        transform = CoordinateTransform(resolution=(100, 100), flip_y=False)
        # Without flip, physics -1 should map to pixel 0
        assert transform.physics_to_pixel_y(-1.0) == 0
        assert transform.physics_to_pixel_y(0.0) == 50
        assert transform.physics_to_pixel_y(1.0) == 100

    def test_physics_to_pixel_batch(self):
        """Test batch physics to pixel conversion."""
        transform = CoordinateTransform(resolution=(100, 100))
        positions = torch.tensor([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
        pixels = transform.physics_to_pixel(positions)
        
        assert pixels.shape == (3, 2)
        assert pixels[0, 0].item() == 0    # x: -1 -> 0
        assert pixels[0, 1].item() == 0    # y: 1 -> 0 (flipped)
        assert pixels[1, 0].item() == 50   # x: 0 -> 50
        assert pixels[1, 1].item() == 50   # y: 0 -> 50
        assert pixels[2, 0].item() == 100  # x: 1 -> 100
        assert pixels[2, 1].item() == 100  # y: -1 -> 100 (flipped)

    def test_physics_size_to_pixel(self):
        """Test physics size to pixel conversion."""
        transform = CoordinateTransform(resolution=(100, 100))
        assert transform.physics_size_to_pixel(1.0) == 50
        assert transform.physics_size_to_pixel(0.5) == 25
        assert transform.physics_size_to_pixel(0.1) == 5

    def test_pixel_to_physics_x_edges(self):
        """Test pixel to physics X conversion at edges."""
        transform = CoordinateTransform(resolution=(100, 100))
        assert abs(transform.pixel_to_physics_x(0) - (-1.0)) < 0.01
        assert abs(transform.pixel_to_physics_x(50) - 0.0) < 0.01
        assert abs(transform.pixel_to_physics_x(100) - 1.0) < 0.01

    def test_pixel_to_physics_y_edges(self):
        """Test pixel to physics Y conversion at edges (with flip)."""
        transform = CoordinateTransform(resolution=(100, 100))
        assert abs(transform.pixel_to_physics_y(0) - 1.0) < 0.01   # top -> +1
        assert abs(transform.pixel_to_physics_y(50) - 0.0) < 0.01
        assert abs(transform.pixel_to_physics_y(100) - (-1.0)) < 0.01  # bottom -> -1

    def test_pixel_to_physics_batch(self):
        """Test batch pixel to physics conversion."""
        transform = CoordinateTransform(resolution=(100, 100))
        pixels = torch.tensor([[0, 0], [50, 50], [100, 100]], dtype=torch.float)
        physics = transform.pixel_to_physics(pixels)
        
        assert physics.shape == (3, 2)
        assert abs(physics[0, 0].item() - (-1.0)) < 0.01
        assert abs(physics[0, 1].item() - 1.0) < 0.01
        assert abs(physics[1, 0].item() - 0.0) < 0.01
        assert abs(physics[1, 1].item() - 0.0) < 0.01
        assert abs(physics[2, 0].item() - 1.0) < 0.01
        assert abs(physics[2, 1].item() - (-1.0)) < 0.01

    def test_round_trip_x(self):
        """Test round-trip conversion for X coordinate."""
        transform = CoordinateTransform()
        for physics_x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            pixel_x = transform.physics_to_pixel_x(physics_x)
            back = transform.pixel_to_physics_x(pixel_x)
            assert abs(back - physics_x) < 0.02, f"Failed for {physics_x}: got {back}"

    def test_round_trip_y(self):
        """Test round-trip conversion for Y coordinate."""
        transform = CoordinateTransform()
        for physics_y in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            pixel_y = transform.physics_to_pixel_y(physics_y)
            back = transform.pixel_to_physics_y(pixel_y)
            assert abs(back - physics_y) < 0.02, f"Failed for {physics_y}: got {back}"

    def test_round_trip_batch(self):
        """Test round-trip conversion for batch of positions."""
        transform = CoordinateTransform()
        original = torch.tensor([[-0.5, 0.5], [0.0, 0.0], [0.5, -0.5]])
        pixels = transform.physics_to_pixel(original)
        back = transform.pixel_to_physics(pixels)
        
        assert torch.allclose(original, back, atol=0.02)

    def test_create_pixel_grid(self):
        """Test pixel grid creation."""
        transform = CoordinateTransform(resolution=(10, 10))
        grid = transform.create_pixel_grid()
        
        assert grid.shape == (10, 10, 2)
        # Check corners
        assert abs(grid[0, 0, 0].item() - (-1.0)) < 0.01  # top-left x
        assert abs(grid[0, 0, 1].item() - 1.0) < 0.01     # top-left y
        assert abs(grid[-1, -1, 0].item() - 1.0) < 0.01   # bottom-right x
        assert abs(grid[-1, -1, 1].item() - (-1.0)) < 0.01  # bottom-right y

    def test_create_pixel_grid_device(self):
        """Test pixel grid creation with device."""
        transform = CoordinateTransform(resolution=(10, 10))
        grid = transform.create_pixel_grid(device=torch.device('cpu'))
        assert grid.device.type == 'cpu'

    def test_custom_bounds(self):
        """Test with custom physics bounds."""
        transform = CoordinateTransform(
            physics_bounds=(-2.0, 2.0),
            resolution=(100, 100)
        )
        assert transform.physics_to_pixel_x(-2.0) == 0
        assert transform.physics_to_pixel_x(0.0) == 50
        assert transform.physics_to_pixel_x(2.0) == 100

    def test_repr(self):
        """Test string representation."""
        transform = CoordinateTransform()
        repr_str = repr(transform)
        assert 'CoordinateTransform' in repr_str
        assert '-1.0' in repr_str
        assert '1.0' in repr_str
        assert '256' in repr_str
