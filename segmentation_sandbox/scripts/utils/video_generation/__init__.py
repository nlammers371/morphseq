"""
Video generation utilities for MorphSeq pipeline.

Provides foundation video creation and overlay management for progressive enhancement
across different pipeline modules.
"""

from .overlay_manager import OverlayConfig, OverlayManager
from .video_config import COLORBLIND_PALETTE, VideoConfig
from .video_generator import VideoGenerator

__all__ = [
    "VideoGenerator",
    "OverlayManager",
    "OverlayConfig",
    "VideoConfig",
    "COLORBLIND_PALETTE",
]
