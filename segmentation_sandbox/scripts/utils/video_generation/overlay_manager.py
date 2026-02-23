"""Compatibility shim for moved overlay manager module.

Canonical implementation now lives in:
`src.data_pipeline.segmentation.video_generation.overlay_manager`.
"""

from pathlib import Path
import sys

try:
    from src.data_pipeline.segmentation.video_generation.overlay_manager import (  # noqa: F401
        OverlayConfig,
        OverlayManager,
    )
except ModuleNotFoundError:
    # Allow execution when cwd is segmentation_sandbox/ (repo root not on sys.path).
    _repo_root = Path(__file__).resolve().parents[4]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from src.data_pipeline.segmentation.video_generation.overlay_manager import (  # noqa: F401
        OverlayConfig,
        OverlayManager,
    )

__all__ = ["OverlayConfig", "OverlayManager"]
