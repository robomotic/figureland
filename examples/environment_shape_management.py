#!/usr/bin/env python3
"""
Example demonstrating proper OOP shape management in Environment.

Shows validation, error handling, and proper API usage.
"""

import torch
from figureland.physics import Environment, PhysicsEngine
from figureland.shapes import Square
from figureland.exceptions import (
    ShapeOutOfBoundsError,
    ShapePenetrationError,
    DeviceMismatchError
)


def main():
    print("=== Environment Shape Management Example ===\n")

    # Create environment
    env = Environment(bounds=(-1.0, 1.0), gravity=9.8)
    engine = PhysicsEngine(env)

    print("✓ Environment created")

    # Create 2 shapes
    shape1 = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.2, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=42
    )

    shape2 = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.2, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=43
    )

    # Position correctly
    shape1.position[0] = torch.tensor([-0.5, 0.0])
    shape2.position[0] = torch.tensor([0.5, 0.0])

    # Add shapes with validation
    print("\nAdding shapes with validation:")
    try:
        env.add_shape(shape1)
        print("  ✓ Shape 1 added successfully")

        env.add_shape(shape2)
        print("  ✓ Shape 2 added successfully")
        print(f"  ✓ Total shapes in environment: {len(env.shapes)}")

    except ShapePenetrationError:
        print("  ✗ Shapes are overlapping")
    except ShapeOutOfBoundsError:
        print("  ✗ Shape outside environment")

    # Try to add overlapping shape - will fail
    print("\nAttempting to add overlapping shape:")
    shape3 = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.2, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=44
    )
    shape3.position[0] = torch.tensor([-0.5, 0.0])  # Exact overlap with shape1

    try:
        env.add_shape(shape3)
        print("  ✓ Shape 3 added")
    except ShapePenetrationError as e:
        print(f"  ✗ Correctly rejected: {e}")

    # Try to add out of bounds shape - will fail
    print("\nAttempting to add out of bounds shape:")
    shape4 = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.2, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=45
    )
    shape4.position[0] = torch.tensor([-2.0, 0.0])  # Outside bounds

    try:
        env.add_shape(shape4)
        print("  ✓ Shape 4 added")
    except ShapeOutOfBoundsError as e:
        print(f"  ✗ Correctly rejected: {e}")

    # Run simulation
    print("\nRunning simulation with managed shapes:")
    for frame in range(5):
        positions, collisions = engine.step()
        print(f"  Frame {frame}: Positions = {positions.numpy().round(2)}")

    print("\n✓ Simulation completed successfully")
    print("\n=== Validation Checks ===")
    print("✅ Automatic bounds checking")
    print("✅ Automatic penetration checking")
    print("✅ Device consistency validation")
    print("✅ Batch size validation")
    print("✅ Proper error hierarchy")
    print("✅ Encapsulated shape management")


if __name__ == "__main__":
    main()
