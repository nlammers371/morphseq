"""Segmentation video-generation utilities (pipeline-scoped)."""

from .models import AnnotationRecord, FrameRecord, VideoRecord
from .overlay_manager import OverlayConfig, OverlayManager
from .results_adapter import detect_source_format, list_videos, load_video_record
from .service import generate_eval_videos_for_experiment
from .video_config import COLORBLIND_PALETTE, OVERLAY_COLORS, VideoConfig
from .video_generator import VideoGenerator

__all__ = [
    "AnnotationRecord",
    "FrameRecord",
    "VideoRecord",
    "OverlayConfig",
    "OverlayManager",
    "VideoConfig",
    "VideoGenerator",
    "COLORBLIND_PALETTE",
    "OVERLAY_COLORS",
    "detect_source_format",
    "list_videos",
    "load_video_record",
    "generate_eval_videos_for_experiment",
]
