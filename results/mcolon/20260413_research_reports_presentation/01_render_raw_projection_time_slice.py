"""Render the talk version of the raw projection time-slice figure."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260413_research_reports_cache"
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


RUN_PATH = REPO_ROOT / "results/mcolon/20260410_axis_init_comparison/results/raw_projection/run/condensed_positions.npz"
OUTPUT_PATH = SCRIPT_DIR / "raw_projection_time_slice.html"
DISPLAY_NAME_MAP = {
    "ab": "wildtype",
    "wik-ab": "wildtype",
    "wik_ab": "wildtype",
    "inj_ctrl": "inj. ctrl",
    "pbx1b_crispant": "pbx1b",
    "pbx4_crispant": "pbx4",
    "pbx1b_pbx4_crispant": "pbx1b + pbx4",
}
PRESENTATION_COLOR_MAP = {
    "inj. ctrl": "#2166AC",
    "pbx1b": "#9467BD",
    "pbx4": "#F7B267",
    "pbx1b + pbx4": "#B2182B",
    "wildtype": "#888888",
}
DISPLAY_ORDER = ["wildtype", "inj. ctrl", "pbx1b", "pbx4", "pbx1b + pbx4"]


def _normalize_labels(labels: np.ndarray | None) -> np.ndarray | None:
    if labels is None:
        return None
    return np.asarray([DISPLAY_NAME_MAP.get(str(label), str(label)) for label in labels], dtype=object)


def _presentation_color_map(labels: np.ndarray | None) -> dict[str, str] | None:
    if labels is None:
        return None
    present = {str(label) for label in _normalize_labels(labels).tolist()}
    ordered = [label for label in DISPLAY_ORDER if label in present]
    return {label: PRESENTATION_COLOR_MAP[label] for label in ordered}


def _strip_subplot_titles(fig) -> None:
    fig.layout.annotations = tuple(
        ann for ann in fig.layout.annotations
        if ann.text not in {"3D overview", "Current time slice"}
    )
    for frame in fig.frames:
        if frame.layout and getattr(frame.layout, "annotations", None):
            frame.layout.annotations = tuple(
                ann for ann in frame.layout.annotations
                if ann.text not in {"3D overview", "Current time slice"}
            )


def main() -> None:
    run = tc.load_run(RUN_PATH, title="<b>3D</b> Trajectories")
    labels = _normalize_labels(run.labels)
    color_map = _presentation_color_map(run.labels)
    fig = tc.time_slice_html(
        run.positions,
        run.mask,
        run.time_values,
        labels=labels,
        embryo_ids=run.embryo_ids,
        color_map=color_map,
        title="",
    )
    fig.update_layout(title="")
    for ann in fig.layout.annotations:
        if ann.text == "3D overview":
            ann.text = "<b>3D</b> Trajectories"
    for frame in fig.frames:
        if frame.layout and getattr(frame.layout, "annotations", None):
            for ann in frame.layout.annotations:
                if ann.text == "3D overview":
                    ann.text = "<b>3D</b> Trajectories"
    _strip_subplot_titles(fig)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_PATH), include_plotlyjs="cdn")
    print(f"Saved time-slice HTML: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
