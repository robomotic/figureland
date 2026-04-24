#!/usr/bin/env python3
"""
Example: Two falling balls in 400x100 container.

Demonstrates:
- Physics simulation with gravity
- Collision detection and bouncing
- Video rendering
- Frame export
"""

import torch
import cv2
import numpy as np
import os
from figureland.shapes import batch_generate_shapes
from figureland.physics import PhysicsEngine, Environment


def main():
    # Simulation parameters
    resolution = (100, 400)  # height x width
    total_frames = 200
    fps = 30
    gravity = 9.8
    elasticity = 0.8

    # Initialize physics
    env = Environment(bounds=(-1.0, 1.0), gravity=gravity)
    engine = PhysicsEngine(env, dt=1/fps)

    # Create two shapes at top of container
    shapes = batch_generate_shapes(
        shape_types=['square', 'square'],
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.1, 0.1),
        mass_range=(1.0, 1.0),
        elasticity_range=(elasticity, elasticity),
        seed=42
    )

    # Position balls at top left and right
    shapes[0].position[0] = torch.tensor([-0.5, 0.8])
    shapes[1].position[0] = torch.tensor([0.5, 0.8])

    # Set distinct colors
    shapes[0].color[:] = torch.tensor([1.0, 0.0, 0.0])  # Red
    shapes[1].color[:] = torch.tensor([0.0, 0.0, 1.0])  # Blue


    # Add shapes to environment for physics simulation
    env.add_shapes(shapes)

    # Create output directory
    os.makedirs('./output', exist_ok=True)

    print(f"Generating {total_frames} frame simulation...")

    def render_shapes_to_frame(shapes_list):
        """Render shapes as circles (balls) onto a frame."""
        frame = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
        for shape in shapes_list:
            # Convert physics coordinates (-1 to 1) to pixel coordinates
            x = int((shape.position[0, 0].item() + 1.0) * (resolution[1] / 2))
            y = int((1.0 - shape.position[0, 1].item()) * (resolution[0] / 2))
            radius = int(shape.size[0, 0].item() * (resolution[0] / 2))

            # Convert RGB to BGR for OpenCV
            color = (
                int(shape.color[0, 2].item() * 255),
                int(shape.color[0, 1].item() * 255),
                int(shape.color[0, 0].item() * 255),
            )

            # Draw as a filled circle (ball)
            cv2.circle(frame, (x, y), radius, color, -1)
        return frame

    # Save first frame before simulation
    frame = render_shapes_to_frame(shapes)
    cv2.imwrite('./output/falling_balls_first.png', frame)
    print("✅ Saved first frame")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('./output/falling_balls.mp4', fourcc, fps, (resolution[1], resolution[0]))

    # Run simulation with frame capture
    for _ in range(total_frames):
        frame = render_shapes_to_frame(shapes)
        writer.write(frame)
        engine.step()

    writer.release()

    # Save last frame after simulation
    frame = render_shapes_to_frame(shapes)
    cv2.imwrite('./output/falling_balls_last.png', frame)
    print("✅ Saved last frame")

    print(f"✅ Video saved: ./output/falling_balls.mp4")

    print("\nSimulation complete:")
    print(f"  - Resolution: {resolution[1]}x{resolution[0]}")
    print(f"  - Gravity: {gravity} m/s²")
    print(f"  - Elasticity: {elasticity}")
    print(f"  - Frames: {total_frames}")


if __name__ == "__main__":
    main()