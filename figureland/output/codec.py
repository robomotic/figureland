"""
Video codec detection and selection utilities.

Detects available video codecs on the current system and provides
automatic selection of the best available codec for video export.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class CodecDetector:
    """Detects available video codecs and selects the best one.
    
    Tests codecs by attempting to create a VideoWriter with each codec.
    Returns the first codec that successfully opens.
    """
    
    # Priority order for codecs (best quality/compatibility first)
    PREFERRED_CODECS: List[Tuple[str, str]] = [
        ('libx264', 'mp4'),   # H.264 - best quality, widely supported
        ('avc1', 'mp4'),      # AVC1 - good compatibility
        ('h264', 'mp4'),      # Alternative H.264 tag
        ('mp4v', 'mp4'),      # MPEG-4 Visual - reliable fallback
        ('XVID', 'avi'),      # Xvid - AVI fallback
        ('MJPG', 'avi'),      # Motion JPEG - last resort
    ]
    
    # Test frame size for codec detection
    TEST_WIDTH = 64
    TEST_HEIGHT = 64
    
    @classmethod
    def is_codec_available(cls, codec: str, fmt: str = 'mp4') -> bool:
        """Test if a specific codec is available.
        
        Args:
            codec: FourCC codec string (e.g., 'mp4v', 'libx264')
            fmt: Container format ('mp4', 'avi', etc.)
            
        Returns:
            True if codec is available and can be opened
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            # Try to create a writer with the codec
            writer = cv2.VideoWriter(
                '/dev/null',  # Discard output
                fourcc,
                30,
                (cls.TEST_WIDTH, cls.TEST_HEIGHT)
            )
            is_opened = writer.isOpened()
            writer.release()
            return is_opened
        except Exception as e:
            logger.debug(f"Codec {codec} not available: {e}")
            return False
    
    @classmethod
    def detect_available_codecs(cls) -> List[Tuple[str, str]]:
        """Test which codecs are available.
        
        Returns:
            List of (codec, format) tuples for available codecs,
            ordered by preference (best first)
        """
        available = []
        for codec, fmt in cls.PREFERRED_CODECS:
            if cls.is_codec_available(codec, fmt):
                available.append((codec, fmt))
                logger.debug(f"Available codec: {codec} ({fmt})")
            else:
                logger.debug(f"Unavailable codec: {codec} ({fmt})")
        
        if not available:
            logger.warning("No video codecs available! Video export will fail.")
        
        return available
    
    @classmethod
    def get_best_codec(cls, preferred_format: str = 'mp4') -> Tuple[str, str]:
        """Get the best available codec for the preferred format.
        
        Args:
            preferred_format: Preferred container format ('mp4', 'avi', etc.)
            
        Returns:
            Tuple of (codec, format) for the best available codec
        """
        available = cls.detect_available_codecs()
        
        if not available:
            # Return mp4v as fallback even if it might not work
            logger.error("No codecs available, returning mp4v as fallback")
            return ('mp4v', 'mp4')
        
        # Filter by preferred format first
        format_matches = [(c, f) for c, f in available if f == preferred_format]
        if format_matches:
            return format_matches[0]
        
        # Return best available regardless of format
        return available[0]
    
    @classmethod
    def get_codec_for_format(cls, fmt: str) -> Optional[Tuple[str, str]]:
        """Get the best available codec for a specific format.
        
        Args:
            fmt: Container format ('mp4', 'avi', etc.)
            
        Returns:
            Tuple of (codec, format) or None if no codec available for format
        """
        available = cls.detect_available_codecs()
        for codec, codec_fmt in available:
            if codec_fmt == fmt:
                return (codec, codec_fmt)
        return None
