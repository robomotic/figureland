#!/usr/bin/env python3
"""
Example: Tiles-based falling balls simulation.

Loads a single tile from a 4x4 grid PNG image and repeats it across a large background.
Uses PIL and imageio for image/video processing.
"""

import os
import numpy as np
from PIL import Image
import imageio

# Tile configuration
TILES_IMAGE = 'images/tiles.png'
GRID_COLS = 4
GRID_ROWS = 4
BACKGROUND_TILES_WIDE = 16
BACKGROUND_TILES_TALL = 8


def main():
    print(f"Loading tiles from {TILES_IMAGE}...")
    
    # Load all tiles from the 4x4 grid
    img = Image.open(TILES_IMAGE)
    img_width, img_height = img.size
    tile_width = img_width // GRID_COLS
    tile_height = img_height // GRID_ROWS
    
    tiles = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            left = col * tile_width
            top = row * tile_height
            tile = img.crop((left, top, left + tile_width, top + tile_height))
            tiles.append(tile)
    
    print(f"Loaded {len(tiles)} tiles (4x4 grid)")
    
    # Pick ONE random tile to use for the entire background
    tile_idx = np.random.randint(0, len(tiles))
    tile = tiles[tile_idx]
    tile_arr = np.array(tile.convert('RGB'))
    print(f"Selected tile {tile_idx} for background (will repeat)")
    
    print(f"Tile size: {tile_width}x{tile_height}")
    
    # Create large background dimensions
    bg_width = tile_width * BACKGROUND_TILES_WIDE
    bg_height = tile_height * BACKGROUND_TILES_TALL
    print(f"Background: {bg_width}x{bg_height} ({BACKGROUND_TILES_WIDE}x{BACKGROUND_TILES_TALL} tiles)")
    
    # Create output directory
    os.makedirs('./output', exist_ok=True)
    
    # Simulation parameters
    num_balls = 20
    num_frames = 200
    fps = 30
    gravity = 9.8
    elasticity = 0.8
    
    # Initialize ball positions and properties
    np.random.seed(42)
    ball_x = np.random.uniform(0.1, bg_width - 0.1, num_balls)
    ball_y = np.random.uniform(0.8 * bg_height, bg_height - 0.1, num_balls)
    ball_vx = np.random.uniform(-20, 20, num_balls)
    ball_vy = np.random.uniform(0, 100, num_balls)
    ball_radius = np.random.randint(10, 30, num_balls)
    ball_colors = [
        (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        for _ in range(num_balls)
    ]
    
    # Create video writer using imageio (better codec support)
    writer = imageio.get_writer(
        './output/tiles_basic.mp4',
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p'
    )
    
    print(f"Generating {num_frames} frame simulation with {num_balls} balls...")
    
    for frame_idx in range(num_frames):
        # Create background with tiled pattern
        frame = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
        
        # Draw tiled background - repeat the SAME tile everywhere
        for row in range(BACKGROUND_TILES_TALL):
            for col in range(BACKGROUND_TILES_WIDE):
                x = col * tile_width
                y = row * tile_height
                frame[y:y+tile_height, x:x+tile_width] = tile_arr
        
        # Update ball physics
        dt = 1.0 / fps
        for i in range(num_balls):
            # Apply gravity
            ball_vy[i] += gravity * 100 * dt
            
            # Update position
            ball_x[i] += ball_vx[i] * dt
            ball_y[i] += ball_vy[i] * dt
            
            # Bounce off walls
            if ball_x[i] - ball_radius[i] < 0:
                ball_x[i] = ball_radius[i]
                ball_vx[i] *= -elasticity
            elif ball_x[i] + ball_radius[i] > bg_width:
                ball_x[i] = bg_width - ball_radius[i]
                ball_vx[i] *= -elasticity
            
            if ball_y[i] - ball_radius[i] < 0:
                ball_y[i] = ball_radius[i]
                ball_vy[i] *= -elasticity
            elif ball_y[i] + ball_radius[i] > bg_height:
                ball_y[i] = bg_height - ball_radius[i]
                ball_vy[i] *= -elasticity
            
            # Draw ball as circle using numpy mask
            y_coords, x_coords = np.ogrid[:bg_height, :bg_width]
            mask = (x_coords - ball_x[i])**2 + (y_coords - ball_y[i])**2 <= ball_radius[i]**2
            frame[mask] = ball_colors[i]
        
        # Save first and last frames as PNG
        if frame_idx == 0:
            Image.fromarray(frame).save('./output/tiles_basic_first.png')
        if frame_idx == num_frames - 1:
            Image.fromarray(frame).save('./output/tiles_basic_last.png')
        
        writer.append_data(frame)
    
    writer.close()
    
    print(f"✅ Video saved: ./output/tiles_basic.mp4")
    print(f"✅ First frame: ./output/tiles_basic_first.png")
    print(f"✅ Last frame: ./output/tiles_basic_last.png")
    print(f"\nSimulation complete:")
    print(f"  - Background: {bg_width}x{bg_height} pixels")
    print(f"  - Tiles: {BACKGROUND_TILES_WIDE}x{BACKGROUND_TILES_TALL} grid (single tile repeated)")
    print(f"  - Balls: {num_balls}")
    print(f"  - Frames: {num_frames}")


if __name__ == "__main__":
    main()