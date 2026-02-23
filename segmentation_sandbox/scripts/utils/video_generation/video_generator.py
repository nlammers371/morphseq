"""Compatibility shim for moved video generator module.

Canonical implementation now lives in:
`src.data_pipeline.segmentation.video_generation.video_generator`.
"""

from pathlib import Path
import sys

try:
    from src.data_pipeline.segmentation.video_generation.video_generator import (  # noqa: F401
        VideoGenerator,
    )
except ModuleNotFoundError:
    # Allow execution when cwd is segmentation_sandbox/ (repo root not on sys.path).
    _repo_root = Path(__file__).resolve().parents[4]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from src.data_pipeline.segmentation.video_generation.video_generator import (  # noqa: F401
        VideoGenerator,
    )

__all__ = ["VideoGenerator"]
