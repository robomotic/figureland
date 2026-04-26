#!/usr/bin/env python3
"""
Example: Tiles-based falling balls simulation.

Loads tiles from a 4x4 grid PNG image and places them on a large background.
Uses PIL to extract individual tiles and render them as falling circles on a 16x8 tile background.
"""

import os
import cv2
import numpy as np
from PIL import Image

# Tile configuration
TILES_IMAGE = 'images/tiles.png'
GRID_COLS = 4
GRID_ROWS = 4
BACKGROUND_TILES_WIDE = 16
BACKGROUND_TILES_TALL = 8


def load_tiles(image_path, grid_cols, grid_rows):
    """Load tiles from a PNG image and extract individual tiles."""
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    tile_width = img_width // grid_cols
    tile_height = img_height // grid_rows
    
    tiles = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Extract tile region
            left = col * tile_width
            top = row * tile_height
            right = left + tile_width
            bottom = top + tile_height
            
            tile = img.crop((left, top, right, bottom))
            tiles.append(tile)
    
    return tiles


def main():
    print(f"Loading tiles from {TILES_IMAGE}...")
    tiles = load_tiles(TILES_IMAGE, GRID_COLS, GRID_ROWS)
    print(f"Loaded {len(tiles)} tiles (4x4 grid)")
    
    # Pick ONE random tile to use for the entire background
    tile_idx = np.random.randint(0, len(tiles))
    tile = tiles[tile_idx]
    tile_arr = np.array(tile.convert('RGB'))
    print(f"Selected tile {tile_idx} for background (will repeat)")
    
    # Get tile dimensions
    tile_width, tile_height = tile.size
    print(f"Tile size: {tile_width}x{tile_height}")
    
    # Create large background
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
    
    # Initialize ball positions (random x, y at top)
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
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('./output/tiles_basic.mp4', fourcc, fps, (bg_width, bg_height))
    
    print(f"Generating {num_frames} frame simulation with {num_balls} balls...")
    
    for frame_idx in range(num_frames):
        # Create dark background with tiled pattern
        frame = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
        
        # Draw tiled background - repeat the SAME tile everywhere
        for row in range(BACKGROUND_TILES_TALL):
            for col in range(BACKGROUND_TILES_WIDE):
                x = col * tile_width
                y = row * tile_height
                frame[y:y+tile_height, x:x+tile_width] = tile_arr
        
        # Update and draw balls
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
            
            # Draw ball as circle
            center = (int(ball_x[i]), int(ball_y[i]))
            cv2.circle(frame, center, ball_radius[i], ball_colors[i], -1)
        
        # Save first and last frames
        if frame_idx == 0:
            cv2.imwrite('./output/tiles_basic_first.png', frame)
        if frame_idx == num_frames - 1:
            cv2.imwrite('./output/tiles_basic_last.png', frame)
        
        writer.write(frame)
    
    writer.release()
    print(f"✅ Video saved: ./output/tiles_basic.mp4")
    print(f"✅ First frame: ./output/tiles_basic_first.png")
    print(f"✅ Last frame: ./output/tiles_basic_last.png")
    print(f"\nSimulation complete:")
    print(f"  - Background: {bg_width}x{bg_height} pixels")
    print(f"  - Tiles: {BACKGROUND_TILES_WIDE}x{BACKGROUND_TILES_TALL} grid ({len(tiles)} unique tiles)")
    print(f"  - Balls: {num_balls}")
    print(f"  - Frames: {num_frames}")


if __name__ == "__main__":
    main()