#!/usr/bin/env python3
"""Compatibility shim for eval-video rendering CLI.

Canonical implementation now lives in:
`src.data_pipeline.segmentation.video_generation.render_eval_video`.
"""

from pathlib import Path
import sys

try:
    from src.data_pipeline.segmentation.video_generation.render_eval_video import main
except ModuleNotFoundError:
    # Allow execution when cwd is segmentation_sandbox/ (repo root not on sys.path).
    _repo_root = Path(__file__).resolve().parents[4]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from src.data_pipeline.segmentation.video_generation.render_eval_video import main


if __name__ == "__main__":
    raise SystemExit(main())
