"""Render condensation iterations MP4 — presentation quality."""

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
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import analyze.trajectory_condensation as tc
from analyze.trajectory_condensation.viz.animation import _resolve_color_map

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

ELEV = 25.0
AZIM = 60.0
FPS = 6
DPI = 480
FIGSIZE = (8, 7)
POINT_SIZE = 10.0
ALPHA_POINT = 0.7
ALPHA_LINE = 0.1
LINEWIDTH = 1.2
MIN_OBS = 2


def _normalize_labels(labels):
    if labels is None:
        return None
    return np.asarray(
        [DISPLAY_NAME_MAP.get(str(l), str(l)) for l in labels], dtype=object
    )


def _color_map(labels):
    if labels is None:
        return None
    present = {str(l) for l in _normalize_labels(labels).tolist()}
    ordered = [l for l in DISPLAY_ORDER if l in present]
    return {l: PRESENTATION_COLOR_MAP[l] for l in ordered}


def _draw_frame(ax, positions, mask, time_values, labels, color_map):
    N_e = positions.shape[0]
    groups = np.unique(labels) if labels is not None else [None]

    legend_handles = []
    for group in groups:
        if group is not None:
            embryo_idx = np.where(labels == group)[0]
            color = color_map.get(group, "steelblue")
        else:
            embryo_idx = np.arange(N_e)
            color = "steelblue"

        first = True
        for i in embryo_idx:
            obs_t = np.where(mask[i, :])[0]
            if len(obs_t) < MIN_OBS:
                continue
            xs = positions[i, obs_t, 0]
            ys = positions[i, obs_t, 1]
            zs = time_values[obs_t]
            ax.plot(xs, ys, zs, color=color, alpha=ALPHA_LINE, linewidth=LINEWIDTH)
            sc = ax.scatter(xs, ys, zs, color=color, alpha=ALPHA_POINT, s=POINT_SIZE)
            if first and group is not None:
                legend_handles.append((sc, str(group)))
            first = False

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("time (hpf)", fontsize=16, labelpad=10)
    ax.tick_params(axis='both', labelsize=0, length=0)
    ax.tick_params(axis='z', labelsize=9)

    # Legend with scatter dots (no lines), top-right, ordered by DISPLAY_ORDER
    if legend_handles:
        from matplotlib.lines import Line2D
        handle_dict = {g: sc for sc, g in legend_handles}
        ordered_groups = [g for g in DISPLAY_ORDER if g in handle_dict]
        proxies = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get(g, 'steelblue'),
                   markersize=10, label=g)
            for g in ordered_groups
        ]
        ax.legend(handles=proxies, loc="upper right", fontsize=11, framealpha=0.7,
                  bbox_to_anchor=(1.15, 1.0))


def main() -> None:
    run = tc.load_run(RUN_PATH)

    if run.position_history is None or run.position_history.shape[0] < 2:
        print("No position_history — cannot render iterations MP4.")
        return

    n_frames = run.position_history.shape[0]
    print(f"position_history: {n_frames} snapshots")

    labels = _normalize_labels(run.labels)
    cmap = _color_map(run.labels)
    resolved_cmap = _resolve_color_map(labels, cmap)
    snapshot_iters = run.snapshot_iters if run.snapshot_iters is not None else list(range(n_frames))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "condensation_iterations_4k.mp4"

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=ELEV, azim=AZIM)

    def update(frame):
        ax.cla()
        positions = run.position_history[frame]
        _draw_frame(ax, positions, run.mask, run.time_values, labels, resolved_cmap)

        iter_num = snapshot_iters[frame]
        subtitle = "Initialization" if iter_num == 0 else f"iter {iter_num}"
        fig.suptitle("Morphology-Axis Space", fontsize=16, fontweight="bold", y=0.97)
        ax.set_title(subtitle, fontsize=13, pad=6)
        ax.view_init(elev=ELEV, azim=AZIM)
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)
    anim.save(str(out_path), writer="ffmpeg", fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
