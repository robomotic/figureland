"""
Example demonstrating background image support with different shapes.
Uses an existing image as background and renders various non-overlapping shapes on top.
"""

import torch
from figureland.rendering import Renderer
from figureland.shapes import Square, Rectangle, Triangle, Hexagon, Trapezoid
from figureland.physics import Environment
from pathlib import Path

def main():
    # Use an existing image from the project as background
    background_path = "images/first_frame.png"

    if not Path(background_path).exists():
        print(f"Background image not found: {background_path}")
        print("Using solid color background instead.")
        background_path = None

    # Create renderer with background image (stretched to fit)
    renderer = Renderer(
        resolution=(256, 256),
        anti_alias=2,
        background_image=background_path,
        tile_background=False,  # Set to True for tiled background
        device=torch.device('cpu')
    )

    print(f"Renderer created with background_image={background_path}, tile_background=False")

    # Create environment to manage shapes and check overlaps
    env = Environment(
        bounds=(-1.0, 1.0),
        environment_type='rectangle',
        device=torch.device('cpu')
    )

    # Create different shapes with non-overlapping manual positions
    shapes = []

    # Red square at left center
    square = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.3, 0.3),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=42
    )
    square.position[:] = torch.tensor([[-0.7, 0.0]])  # Left center (adjusted to avoid overlap)
    square.color[:] = torch.tensor([1.0, 0.0, 0.0])  # Red
    shapes.append(square)

    # Blue rectangle at top right
    rect = Rectangle.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.4, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=43
    )
    rect.position[:] = torch.tensor([[0.7, 0.5]])  # Top right
    rect.color[:] = torch.tensor([0.0, 0.0, 1.0])  # Blue
    shapes.append(rect)

    # Green triangle at bottom right
    triangle = Triangle.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.3, 0.3),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=44
    )
    triangle.position[:] = torch.tensor([[0.7, -0.5]])  # Bottom right
    triangle.color[:] = torch.tensor([0.0, 1.0, 0.0])  # Green
    shapes.append(triangle)

    # Yellow hexagon at top left (adjusted position to avoid overlap with square)
    hexagon = Hexagon.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.25, 0.25),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=45
    )
    hexagon.position[:] = torch.tensor([[-0.7, 0.6]])  # Top left (adjusted)
    hexagon.color[:] = torch.tensor([1.0, 1.0, 0.0])  # Yellow
    shapes.append(hexagon)

    # Purple trapezoid at bottom center
    trapezoid = Trapezoid.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.35, 0.35),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.8, 0.8),
        seed=46
    )
    trapezoid.position[:] = torch.tensor([[0.0, -0.7]])  # Bottom center
    trapezoid.color[:] = torch.tensor([0.5, 0.0, 0.5])  # Purple
    shapes.append(trapezoid)

    # Add shapes to environment to validate (strict=True checks for overlaps)
    try:
        env.add_shapes(shapes, strict=True)
        print(f"All {len(env.shapes)} shapes added to environment without overlaps!")
    except Exception as e:
        print(f"Shape overlap detected: {e}")
        # Use only the shapes that were successfully added to the environment
        shapes = env.shapes

    # Render the frame using environment's validated shapes
    print("Rendering frame with background image and non-overlapping shapes...")
    frame = renderer.render(env.shapes if env.shapes else shapes)

    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")

    # Save the output
    import numpy as np
    from PIL import Image

    # Convert to numpy and save
    frame_np = (frame[0].cpu().numpy() * 255).astype(np.uint8)
    output_path = "output/background_demo_single.png"
    Path("output").mkdir(exist_ok=True)
    Image.fromarray(frame_np).save(output_path)
    print(f"Saved single frame to: {output_path}")

    # Demo with tiled background
    print("\n--- Demo with tiled background ---")
    tiled_renderer = Renderer(
        resolution=(256, 256),
        anti_alias=2,
        background_image=background_path,
        tile_background=True,  # Tile the background
        device=torch.device('cpu')
    )

    frame_tiled = tiled_renderer.render(shapes)
    frame_tiled_np = (frame_tiled[0].cpu().numpy() * 255).astype(np.uint8)
    output_tiled = "output/background_demo_tiled.png"
    Image.fromarray(frame_tiled_np).save(output_tiled)
    print(f"Saved tiled background frame to: {output_tiled}")

    print("\nDone! Check the output/ directory for results.")

if __name__ == "__main__":
    main()
