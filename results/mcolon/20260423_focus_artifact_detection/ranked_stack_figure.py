"""
Thin re-export of the motion ranked figure helper.

The focus analysis uses the same column layout and rendering contract,
so this file keeps the new results folder self-contained while avoiding
code duplication.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MOTION_HELPER = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/"
    "results/mcolon/20260421_motion_artifact_detection/ranked_stack_figure.py"
)

spec = importlib.util.spec_from_file_location("motion_ranked_stack_figure", MOTION_HELPER)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load ranked figure helper from {MOTION_HELPER}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

make_ranked_figure = module.make_ranked_figure

