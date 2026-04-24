"""
Unit tests for rendering engine.
"""

import torch
import pytest
import tempfile
import os
from PIL import Image
from figureland.rendering import Renderer
from figureland.shapes import Square, Rectangle


def test_renderer_creation():
    """Test renderer initialization."""
    renderer = Renderer(resolution=(64, 64), anti_alias=2)
    assert renderer.resolution == (64, 64)
    assert renderer.anti_alias == 2


def test_single_shape_rendering():
    """Test rendering single shape."""
    renderer = Renderer(resolution=(64, 64), anti_alias=1)

    shape = Square.from_random(
        batch_size=2,
        bounds=(-1.0, 1.0),
        size_range=(0.3, 0.3),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.5, 0.5),
        seed=42
    )

    frames = renderer.render([shape])
    assert frames.shape == (2, 64, 64, 3)
    assert torch.any(frames > 0)  # Shape should be visible


def test_multiple_shape_rendering():
    """Test rendering multiple shapes with proper ordering."""
    renderer = Renderer(resolution=(64, 64), anti_alias=1)

    # Create two shapes with explicit position at center
    shape1 = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.4, 0.4),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.5, 0.5),
        seed=42
    )
    shape1.position[:] = torch.tensor([[0.0, 0.0]])  # Center
    shape1.color[:] = torch.tensor([1.0, 0.0, 0.0])  # Red

    shape2 = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.2, 0.2),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.5, 0.5),
        seed=43
    )
    shape2.position[:] = torch.tensor([[0.0, 0.0]])  # Same center
    shape2.color[:] = torch.tensor([0.0, 1.0, 0.0])  # Green

    # Render shape1 first, then shape2 (shape2 on top)
    frames = renderer.render([shape1, shape2])

    # Center pixel should be green (top shape)
    center_pixel = frames[0, 32, 32]
    assert torch.allclose(center_pixel, torch.tensor([0.0, 1.0, 0.0]), atol=0.1)


def test_anti_aliasing():
    """Test anti-aliasing downsampling works."""
    renderer_aa = Renderer(resolution=(64, 64), anti_alias=2)
    renderer_no_aa = Renderer(resolution=(64, 64), anti_alias=1)

    shape = Square.from_random(
        batch_size=1,
        bounds=(-1.0, 1.0),
        size_range=(0.3, 0.3),
        mass_range=(1.0, 1.0),
        elasticity_range=(0.5, 0.5),
        seed=42
    )

    frames_aa = renderer_aa.render([shape])
    frames_no_aa = renderer_no_aa.render([shape])

    assert frames_aa.shape == frames_no_aa.shape

    # Anti-aliased image should have smoother edges
    edge_aa = frames_aa[0, :, :, 0].std()
    edge_no_aa = frames_no_aa[0, :, :, 0].std()
    assert edge_aa < edge_no_aa


def test_render_episode():
    """Test rendering full episode history."""
    renderer = Renderer(resolution=(32, 32), anti_alias=1)

    # Create 5 frame history
    history = []
    for i in range(5):
        shape = Square.from_random(
            batch_size=2,
            bounds=(-1.0, 1.0),
            size_range=(0.2, 0.2),
            mass_range=(1.0, 1.0),
            elasticity_range=(0.5, 0.5),
            seed=42 + i
        )
        history.append([shape])

    episode = renderer.render_episode(history)
    assert episode.shape == (5, 2, 32, 32, 3)


def test_background_image():
    """Test rendering with a background image."""
    # Create a simple test image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img = Image.new('RGB', (64, 64), color=(255, 0, 0))  # Red image
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        renderer = Renderer(
            resolution=(64, 64),
            background_image=tmp_path,
            tile_background=False
        )

        shape = Square.from_random(
            batch_size=1,
            bounds=(-1.0, 1.0),
            size_range=(0.2, 0.2),
            mass_range=(1.0, 1.0),
            elasticity_range=(0.5, 0.5),
            seed=42
        )

        frames = renderer.render([shape])
        assert frames.shape == (1, 64, 64, 3)

        # Background should be red (with some tolerance for the shape overlay)
        # Check corners where shape shouldn't be
        assert frames[0, 0, 0, 0] > 0.9  # Red channel should be ~1.0
    finally:
        os.unlink(tmp_path)


def test_background_image_tiled():
    """Test rendering with a tiled background image."""
    # Create a small test image for tiling
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img = Image.new('RGB', (16, 16), color=(0, 255, 0))  # Green image
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        renderer = Renderer(
            resolution=(64, 64),
            background_image=tmp_path,
            tile_background=True
        )

        shape = Square.from_random(
            batch_size=1,
            bounds=(-1.0, 1.0),
            size_range=(0.2, 0.2),
            mass_range=(1.0, 1.0),
            elasticity_range=(0.5, 0.5),
            seed=42
        )

        frames = renderer.render([shape])
        assert frames.shape == (1, 64, 64, 3)

        # Background should be green (with some tolerance for the shape overlay)
        assert frames[0, 0, 0, 1] > 0.9  # Green channel should be ~1.0
    finally:
        os.unlink(tmp_path)


def test_set_background_image():
    """Test setting background image after initialization."""
    renderer = Renderer(resolution=(64, 64))

    # Create a simple test image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img = Image.new('RGB', (64, 64), color=(0, 0, 255))  # Blue image
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        renderer.set_background_image(tmp_path, tile=False)

        shape = Square.from_random(
            batch_size=1,
            bounds=(-1.0, 1.0),
            size_range=(0.2, 0.2),
            mass_range=(1.0, 1.0),
            elasticity_range=(0.5, 0.5),
            seed=42
        )

        frames = renderer.render([shape])
        assert frames.shape == (1, 64, 64, 3)

        # Background should be blue
        assert frames[0, 0, 0, 2] > 0.9  # Blue channel should be ~1.0
    finally:
        os.unlink(tmp_path)
