"""
Unit tests for physics engine with proper OOP shape management.
"""

import torch
import pytest
from figureland.physics import Environment, PhysicsEngine
from figureland.shapes import Square, Rectangle
from figureland.exceptions import ShapePenetrationError, ShapeOutOfBoundsError


def test_environment_creation():
    """Test environment creation with different types."""
    env_rect = Environment(bounds=(-1.0, 1.0), environment_type='rectangle')
    assert env_rect.environment_type == 'rectangle'
    assert len(env_rect.shapes) == 0

    env_circle = Environment(bounds=(-1.0, 1.0), environment_type='circle')
    assert env_circle.environment_type == 'circle'


def test_physics_engine_step():
    """Test physics engine steps correctly update state."""
    env = Environment(bounds=(-1.0, 1.0), gravity=9.8)
    engine = PhysicsEngine(env, dt=1/60)

    shape = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.1, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.5, 0.5),
        seed=42
    )

    initial_vel = shape.velocity.clone()
    initial_pos = shape.position.clone()

    env.add_shape(shape)
    engine.step()

    # Velocity should change due to gravity
    shape = env.shapes[0]
    assert not torch.allclose(shape.velocity, initial_vel)
    assert not torch.allclose(shape.position, initial_pos)


def test_wall_collision():
    """Test wall collision detection and response."""
    env = Environment(bounds=(-1.0, 1.0), gravity=0.0, friction=0.0, air_resistance=0.0)
    engine = PhysicsEngine(env, dt=1/60)

    shape = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.2, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=42
    )
    # Ensure shape is within bounds (half-size 0.2, so max x = 1 - 0.2 = 0.8)
    shape.position[0, 0] = torch.tensor(0.79)  # Near right wall, will collide after step
    shape.velocity[0, 0] = torch.tensor(1.0)  # Moving right

    env.add_shape(shape)
    initial_vel_x = shape.velocity[0, 0].item()
    engine.step()

    # Velocity should be reversed due to collision and reduced by elasticity
    shape = env.shapes[0]
    assert shape.velocity[0, 0] < 0
    expected = torch.tensor(-initial_vel_x * 0.8)
    assert torch.allclose(shape.velocity[0, 0], expected, atol=0.1)


def test_object_collision():
    """Test object-object collision response."""
    env = Environment(bounds=(-1.0, 1.0), gravity=0.0, friction=0.0, air_resistance=0.0)
    engine = PhysicsEngine(env, dt=1/60, collisions_enabled=True)

    shape1 = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.2, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(1.0, 1.0),
        seed=42
    )

    shape2 = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.2, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(1.0, 1.0),
        seed=43
    )

    # Place them exactly touching (distance = sum of half-sizes = 0.4)
    shape1.position[0] = torch.tensor([-0.2, 0.0])
    shape2.position[0] = torch.tensor([0.2, 0.0])
    shape1.velocity[0] = torch.tensor([1.0, 0.0])
    shape2.velocity[0] = torch.tensor([-1.0, 0.0])

    initial_vel1 = shape1.velocity.clone()
    initial_vel2 = shape2.velocity.clone()

    env.add_shapes([shape1, shape2])
    engine.step()

    # Velocities should be swapped for elastic collision of equal masses
    shape1 = env.shapes[0]
    shape2 = env.shapes[1]
    assert torch.allclose(shape1.velocity[0, 0], initial_vel2[0, 0], atol=0.1)
    assert torch.allclose(shape2.velocity[0, 0], initial_vel1[0, 0], atol=0.1)


def test_shape_penetration_validation():
    """Test that adding overlapping shapes raises error."""
    env = Environment(bounds=(-1.0, 1.0))

    shape1 = Square.from_random(1, (-1,1), (0.2,0.2), (1,1), (0.8,0.8))
    shape2 = Square.from_random(1, (-1,1), (0.2,0.2), (1,1), (0.8,0.8))

    shape1.position[0] = torch.tensor([0.0, 0.0])
    shape2.position[0] = torch.tensor([0.0, 0.0])  # Exact overlap

    env.add_shape(shape1)

    with pytest.raises(ShapePenetrationError):
        env.add_shape(shape2)


def test_shape_out_of_bounds_validation():
    """Test that adding out of bounds shapes raises error."""
    env = Environment(bounds=(-1.0, 1.0))

    shape = Square.from_random(1, (-1,1), (0.2,0.2), (1,1), (0.8,0.8))
    shape.position[0] = torch.tensor([2.0, 0.0])  # Way outside bounds

    with pytest.raises(ShapeOutOfBoundsError):
        env.add_shape(shape)


def test_physics_device_transfer():
    """Test physics engine can be moved between devices."""
    env = Environment(bounds=(-1.0, 1.0))
    engine = PhysicsEngine(env)

    engine_cpu = engine.to(torch.device('cpu'))
    assert engine_cpu.device.type == 'cpu'