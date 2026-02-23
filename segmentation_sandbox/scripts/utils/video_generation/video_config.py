"""Compatibility shim for moved video configuration module.

Canonical implementation now lives in:
`src.data_pipeline.segmentation.video_generation.video_config`.
"""

from pathlib import Path
import sys

try:
    from src.data_pipeline.segmentation.video_generation.video_config import (  # noqa: F401
        COLORBLIND_PALETTE,
        OVERLAY_COLORS,
        VideoConfig,
        get_color_cycle,
    )
except ModuleNotFoundError:
    # Allow execution when cwd is segmentation_sandbox/ (repo root not on sys.path).
    _repo_root = Path(__file__).resolve().parents[4]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from src.data_pipeline.segmentation.video_generation.video_config import (  # noqa: F401
        COLORBLIND_PALETTE,
        OVERLAY_COLORS,
        VideoConfig,
        get_color_cycle,
    )

__all__ = [
    "VideoConfig",
    "COLORBLIND_PALETTE",
    "OVERLAY_COLORS",
    "get_color_cycle",
]
