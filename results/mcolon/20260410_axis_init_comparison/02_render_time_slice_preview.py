"""Render a quick condensed time-slice HTML preview for inspection."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260410_axis_centering_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator")
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import analyze.trajectory_condensation as tc


RUN_PATH = SCRIPT_DIR / "results" / "raw_projection" / "run" / "condensed_positions.npz"
OUTPUT_PATH = SCRIPT_DIR / "figures" / "raw_projection_time_slice.html"


def main() -> None:
    run = tc.load_run(RUN_PATH, title="Raw projection")
    tc.time_slice_html(
        run.positions,
        run.mask,
        run.time_values,
        labels=run.labels,
        embryo_ids=run.embryo_ids,
        color_map=run.color_map,
        title="Raw projection time slice",
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
