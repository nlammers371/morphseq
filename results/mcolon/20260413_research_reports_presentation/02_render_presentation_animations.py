"""Render presentation-quality static images and animations.

Outputs (written next to this script):
  figures/init_structure.png          — static image of initial (AlignedUMAP) positions
  figures/final_structure.png         — static image of condensed positions
  figures/init_vs_final_rotation.gif  — side-by-side init/final slow orbit
  figures/condensation_iterations.gif — timelapse of the condensation optimization process
  figures/time_slice.gif              — time-slice scrubber (3D overview + 2D slice)

Note: condensation_iterations.gif requires position_history in the NPZ (run with save_every=N).
If missing, re-run condensation with save_every set before running this script.
"""

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import analyze.trajectory_condensation as tc
from analyze.trajectory_condensation.viz.animation import (
    animate_init_final_rotation,
    animate_iterations,
    animate_rotation,
    animate_time_slice,
    _draw_stacked_3d,
    _resolve_color_map,
)

RUN_PATH = (
    REPO_ROOT
    / "results/mcolon/20260410_axis_init_comparison/results/raw_projection/run/condensed_positions.npz"
)
OUT_DIR = SCRIPT_DIR / "figures"

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
    return np.asarray(
        [DISPLAY_NAME_MAP.get(str(l), str(l)) for l in labels], dtype=object
    )


def _color_map(labels: np.ndarray | None) -> dict[str, str] | None:
    if labels is None:
        return None
    present = {str(l) for l in _normalize_labels(labels).tolist()}
    ordered = [l for l in DISPLAY_ORDER if l in present]
    return {l: PRESENTATION_COLOR_MAP[l] for l in ordered}


def _save_static_3d(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None,
    color_map: dict[str, str],
    title: str,
    output_path: Path,
    elev: float = 25.0,
    azim: float = -60.0,
    dpi: int = 150,
) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    _draw_stacked_3d(
        ax, positions, mask, time_values, labels, color_map,
        point_size=10.0, alpha_point=0.65,
        alpha_line=0.25, linewidth=0.7, min_obs=2,
    )
    subtitle = "z = time (hpf)"
    ax.set_title((title + "\n" + subtitle) if title else subtitle, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    run = tc.load_run(RUN_PATH)
    labels = _normalize_labels(run.labels)
    cmap = _color_map(run.labels)
    resolved_cmap = _resolve_color_map(labels, cmap)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Static: initial structure ---
    if run.x0 is not None:
        _save_static_3d(
            run.x0, run.mask, run.time_values,
            labels, resolved_cmap,
            title="Initial structure (AlignedUMAP)",
            output_path=OUT_DIR / "init_structure.png",
        )
    else:
        print("No x0 in run — skipping init_structure.png")

    # --- Static: final condensed structure ---
    _save_static_3d(
        run.positions, run.mask, run.time_values,
        labels, resolved_cmap,
        title="Condensed structure",
        output_path=OUT_DIR / "final_structure.png",
    )

    # --- Animation: init vs final rotation (side-by-side twirl) ---
    if run.x0 is not None:
        animate_init_final_rotation(
            run.x0, run.positions, run.mask, run.time_values,
            labels=labels,
            color_map=resolved_cmap,
            output_path=OUT_DIR / "init_vs_final_rotation.gif",
            n_frames=480,
            elev=25.0, azim_start=-60.0, azim_end=300.0,
            fps=24, dpi=120,
            figsize=(14, 7),
            point_size=10.0,
            alpha_point=0.65, alpha_line=0.2, linewidth=0.7,
            title="",
        )
    else:
        print("No x0 in run — skipping init_vs_final_rotation.gif")

    # --- Animation: final structure slow orbit ---
    animate_rotation(
        run.positions, run.mask, run.time_values,
        labels=labels,
        color_map=resolved_cmap,
        output_path=OUT_DIR / "final_rotation.gif",
        n_frames=120,
        elev=25.0, azim_start=-60.0, azim_end=300.0,
        fps=6, dpi=120,
        figsize=(8, 7),
        point_size=10.0,
        alpha_point=0.65, alpha_line=0.2, linewidth=0.7,
        title="",
    )

    # --- Animation: condensation optimization timelapse ---
    if run.position_history is not None and run.position_history.shape[0] >= 2:
        animate_iterations(
            run.position_history, run.mask, run.time_values,
            snapshot_iters=run.snapshot_iters,
            labels=labels,
            color_map=resolved_cmap,
            output_path=OUT_DIR / "condensation_iterations.mp4",
            elev=25.0, azim=-60.0, azim_end=300.0,
            rotation=True,
            fps=5, dpi=120,
            figsize=(8, 7),
            point_size=10.0,
            alpha_point=0.65, alpha_line=0.2, linewidth=0.7,
            title="",
        )
    else:
        print("No position_history in run — skipping condensation_iterations.mp4")
        print("Re-run condensation with save_every=N to capture snapshots.")

    # --- Animation: time-slice scrubber ---
    animate_time_slice(
        run.positions, run.mask, run.time_values,
        labels=labels,
        color_map=resolved_cmap,
        output_path=OUT_DIR / "time_slice.gif",
        elev=25.0, azim=-60.0,
        fps=8, n_interp=3, hold_frames=6,
        dpi=120, figsize=(14, 6),
        point_size=18.0,
        alpha_point=0.75, alpha_line=0.15,
        alpha_slice_plane=0.18,
        linewidth=0.6,
        title="",
    )


if __name__ == "__main__":
    main()
