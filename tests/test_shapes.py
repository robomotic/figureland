"""
Unit tests for shape primitives.
"""

import torch
import pytest
from figureland.shapes import Square, Rectangle, Triangle, Hexagon, Trapezoid, generate_shape


@pytest.mark.parametrize("shape_class", [Square, Rectangle, Triangle, Hexagon, Trapezoid])
def test_shape_creation(shape_class):
    """Test shape creation with random parameters."""
    batch_size = 4
    shape = shape_class.from_random(
        batch_size=batch_size,
        bounds=(-1.0, 1.0),
        size_range=(0.1, 0.2),
        mass_range=(0.1, 10.0),
        elasticity_range=(0.2, 0.9),
        seed=42
    )

    assert shape.position.shape == (batch_size, 2)
    assert shape.size.shape == (batch_size, 2)
    assert shape.rotation.shape == (batch_size,)
    assert shape.color.shape == (batch_size, 3)
    assert shape.mass.shape == (batch_size,)
    assert shape.elasticity.shape == (batch_size,)
    assert shape.velocity.shape == (batch_size, 2)


@pytest.mark.parametrize("shape_type", ['square', 'rectangle', 'triangle', 'hexagon', 'trapezoid'])
def test_generate_shape(shape_type):
    """Test shape generation via factory function."""
    shape = generate_shape(
        shape_type,
        batch_size=2,
        bounds=(-1.0, 1.0),
        size_range=(0.1, 0.2),
        mass_range=(0.1, 10.0),
        elasticity_range=(0.2, 0.9),
        seed=42
    )
    assert shape is not None


@pytest.mark.parametrize("shape_class", [Square, Rectangle, Triangle, Hexagon, Trapezoid])
def test_shape_rendering(shape_class):
    """Test shape rendering produces valid tensor output."""
    shape = shape_class.from_random(
        batch_size=2,
        bounds=(-1.0, 1.0),
        size_range=(0.1, 0.2),
        mass_range=(0.1, 10.0),
        elasticity_range=(0.2, 0.9),
        seed=42
    )

    frames = shape.render((64, 64), anti_alias=1)
    assert frames.shape == (2, 64, 64, 3)
    assert torch.all(frames >= 0.0)
    assert torch.all(frames <= 1.0)


def test_square_force_equal_size():
    """Test square forces equal width and height."""
    position = torch.tensor([[0.0, 0.0]])
    size = torch.tensor([[0.2, 0.3]])
    rotation = torch.tensor([0.0])
    color = torch.tensor([[1.0, 0.0, 0.0]])
    mass = torch.tensor([1.0])
    elasticity = torch.tensor([0.5])

    square = Square(position, size, rotation, color, mass, elasticity)
    assert square.size[0, 0] == square.size[0, 1]


def test_shape_device_transfer():
    """Test shape can be moved between devices."""
    shape = Square.from_random(
        batch_size=2,
        bounds=(-1.0, 1.0),
        size_range=(0.1, 0.2),
        mass_range=(0.1, 10.0),
        elasticity_range=(0.2, 0.9),
        seed=42
    )

    shape_cpu = shape.to(torch.device('cpu'))
    assert shape_cpu.position.device.type == 'cpu'
