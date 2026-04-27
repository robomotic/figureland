"""
Video export configuration.

Provides a dataclass for configuring video export with automatic
codec detection when codec is not specified.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from .codec import CodecDetector


@dataclass
class VideoConfig:
    """Configuration for video export.
    
    Attributes:
        codec: Video codec to use. If 'auto' or None, auto-detects best codec.
        format: Container format ('mp4', 'avi', 'gif').
        fps: Frames per second.
        pixelformat: Pixel format for video encoding.
        bitrate: Target bitrate in bits per second. None for auto.
    """
    codec: Optional[str] = None
    format: str = 'mp4'
    fps: int = 30
    pixelformat: str = 'yuv420p'
    bitrate: Optional[int] = None
    
    def resolve_codec(self) -> Tuple[str, str]:
        """Resolve the final codec, auto-detecting if not specified.
        
        Returns:
            Tuple of (codec_name, container_format)
        """
        if self.codec is None or self.codec == 'auto':
            return CodecDetector.get_best_codec(self.format)
        return (self.codec, self.format)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'VideoConfig':
        """Create VideoConfig from a dictionary.
        
        Args:
            config_dict: Dictionary with video config keys
            
        Returns:
            VideoConfig instance
        """
        return cls(
            codec=config_dict.get('codec'),
            format=config_dict.get('format', 'mp4'),
            fps=config_dict.get('fps', 30),
            pixelformat=config_dict.get('pixelformat', 'yuv420p'),
            bitrate=config_dict.get('bitrate'),
        )
