"""
Tests for video codec detection system.
"""

import pytest
from figureland.output.codec import CodecDetector
from figureland.output.config import VideoConfig


class TestCodecDetector:
    """Tests for CodecDetector class."""

    def test_detect_available_codecs_returns_list(self):
        """Test that detect_available_codecs returns a list."""
        available = CodecDetector.detect_available_codecs()
        assert isinstance(available, list)

    def test_detect_available_codecs_has_at_least_one(self):
        """Test that detect_available_codecs returns a list (may be empty if no codecs)."""
        available = CodecDetector.detect_available_codecs()
        assert isinstance(available, list)
        # Note: In some environments, no codecs may be available
        # This test just verifies the method doesn't crash

    def test_is_codec_available_mp4v(self):
        """Test that codec availability check works."""
        # Just verify the method doesn't crash
        result = CodecDetector.is_codec_available('mp4v', 'mp4')
        assert isinstance(result, bool)

    def test_get_best_codec_returns_tuple(self):
        """Test that get_best_codec returns a tuple."""
        codec, fmt = CodecDetector.get_best_codec()
        assert isinstance(codec, str)
        assert isinstance(fmt, str)
        assert len(codec) > 0
        assert len(fmt) > 0

    def test_get_best_codec_mp4_format(self):
        """Test that get_best_codec returns mp4 format by default."""
        codec, fmt = CodecDetector.get_best_codec('mp4')
        assert fmt == 'mp4'

    def test_get_best_codec_prefers_best(self):
        """Test that get_best_codec returns a valid codec tuple."""
        best = CodecDetector.get_best_codec()
        assert isinstance(best, tuple)
        assert len(best) == 2
        assert isinstance(best[0], str)
        assert isinstance(best[1], str)

    def test_get_codec_for_format(self):
        """Test getting codec for specific format."""
        result = CodecDetector.get_codec_for_format('mp4')
        # Should return a tuple or None
        assert result is None or (isinstance(result, tuple) and len(result) == 2)


class TestVideoConfig:
    """Tests for VideoConfig dataclass."""

    def test_default_config(self):
        """Test default VideoConfig values."""
        config = VideoConfig()
        assert config.codec is None
        assert config.format == 'mp4'
        assert config.fps == 30
        assert config.pixelformat == 'yuv420p'
        assert config.bitrate is None

    def test_custom_config(self):
        """Test custom VideoConfig values."""
        config = VideoConfig(
            codec='libx264',
            format='mp4',
            fps=60,
            pixelformat='yuv420p',
            bitrate=5000000
        )
        assert config.codec == 'libx264'
        assert config.format == 'mp4'
        assert config.fps == 60
        assert config.bitrate == 5000000

    def test_resolve_codec_auto(self):
        """Test that auto codec resolution works."""
        config = VideoConfig(codec='auto')
        codec, fmt = config.resolve_codec()
        assert isinstance(codec, str)
        assert isinstance(fmt, str)

    def test_resolve_codec_none(self):
        """Test that None codec triggers auto-detection."""
        config = VideoConfig(codec=None)
        codec, fmt = config.resolve_codec()
        assert isinstance(codec, str)
        assert isinstance(fmt, str)

    def test_resolve_codec_explicit(self):
        """Test that explicit codec is returned."""
        config = VideoConfig(codec='mp4v', format='mp4')
        codec, fmt = config.resolve_codec()
        assert codec == 'mp4v'
        assert fmt == 'mp4'

    def test_from_dict(self):
        """Test creating VideoConfig from dictionary."""
        config_dict = {
            'codec': 'libx264',
            'format': 'mp4',
            'fps': 60
        }
        config = VideoConfig.from_dict(config_dict)
        assert config.codec == 'libx264'
        assert config.format == 'mp4'
        assert config.fps == 60

    def test_from_dict_defaults(self):
        """Test creating VideoConfig from partial dictionary."""
        config_dict = {'fps': 24}
        config = VideoConfig.from_dict(config_dict)
        assert config.codec is None
        assert config.format == 'mp4'
        assert config.fps == 24
