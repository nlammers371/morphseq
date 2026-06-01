"""Render init-only, final-only, and side-by-side rotation GIFs.

Outputs:
  figures/init_rotation.gif            — init structure slow orbit
  figures/final_rotation_pres.gif      — final structure slow orbit
  figures/init_vs_final_rotation.gif   — side-by-side slow orbit

All at 480 frames / 24 fps = 20s full rotation, presentation styling.
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
    for n in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
        Path(os.environ[n]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from matplotlib.lines import Line2D
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
    "ab": "wildtype", "wik-ab": "wildtype", "wik_ab": "wildtype",
    "inj_ctrl": "inj. ctrl",
    "pbx1b_crispant": "pbx1b",
    "pbx4_crispant": "pbx4",
    "pbx1b_pbx4_crispant": "pbx1b + pbx4",
}
PRESENTATION_COLOR_MAP = {
    "inj. ctrl": "#2166AC", "pbx1b": "#9467BD",
    "pbx4": "#F7B267", "pbx1b + pbx4": "#B2182B", "wildtype": "#888888",
}
DISPLAY_ORDER = ["wildtype", "inj. ctrl", "pbx1b", "pbx4", "pbx1b + pbx4"]

N_FRAMES = 480
FPS = 24
DPI = 120
ELEV = 25.0
AZIM_START = -60.0
AZIM_END = 300.0
POINT_SIZE = 10.0
ALPHA_POINT = 0.7
ALPHA_LINE = 0.1
LINEWIDTH = 1.2
MIN_OBS = 2


def _normalize_labels(labels):
    return np.asarray([DISPLAY_NAME_MAP.get(str(l), str(l)) for l in labels], dtype=object)


def _draw_pres(ax, positions, mask, time_values, labels, color_map):
    """Draw stacked 3D with presentation styling."""
    N_e = positions.shape[0]
    groups = np.unique(labels) if labels is not None else [None]
    legend_handles = []

    for group in groups:
        embryo_idx = np.where(labels == group)[0] if group is not None else np.arange(N_e)
        color = color_map.get(group, "steelblue") if group is not None else "steelblue"
        first = True
        for i in embryo_idx:
            obs_t = np.where(mask[i, :])[0]
            if len(obs_t) < MIN_OBS:
                continue
            xs, ys, zs = positions[i, obs_t, 0], positions[i, obs_t, 1], time_values[obs_t]
            ax.plot(xs, ys, zs, color=color, alpha=ALPHA_LINE, linewidth=LINEWIDTH)
            ax.scatter(xs, ys, zs, color=color, alpha=ALPHA_POINT, s=POINT_SIZE)
            if first and group is not None:
                legend_handles.append(group)
            first = False

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("time (hpf)", fontsize=16, labelpad=10)
    ax.tick_params(axis='both', labelsize=0, length=0)
    ax.tick_params(axis='z', labelsize=9)

    if legend_handles:
        ordered = [g for g in DISPLAY_ORDER if g in legend_handles]
        proxies = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_map.get(g, 'steelblue'),
                   markersize=10, label=g)
            for g in ordered
        ]
        ax.legend(handles=proxies, loc="upper right", fontsize=11,
                  framealpha=0.7, bbox_to_anchor=(1.15, 1.0))


def _render_single(positions, mask, time_values, labels, color_map, title, output_path):
    azimuths = np.linspace(AZIM_START, AZIM_END, N_FRAMES)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=ELEV, azim=AZIM_START)

    def update(frame):
        ax.cla()
        _draw_pres(ax, positions, mask, time_values, labels, color_map)
        fig.suptitle("Morphology-Axis Space", fontsize=16, fontweight="bold", y=0.97)
        ax.set_title(title, fontsize=13, pad=6)
        ax.view_init(elev=ELEV, azim=azimuths[frame])
        return []

    anim = mpl_animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False)
    anim.save(str(output_path), writer="pillow", fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _render_side_by_side(x0, positions, mask, time_values, labels, color_map, output_path):
    azimuths = np.linspace(AZIM_START, AZIM_END, N_FRAMES)
    fig = plt.figure(figsize=(14, 7))
    ax_init = fig.add_subplot(121, projection="3d")
    ax_final = fig.add_subplot(122, projection="3d")

    def update(frame):
        ax_init.cla()
        ax_final.cla()
        _draw_pres(ax_init, x0, mask, time_values, labels, color_map)
        _draw_pres(ax_final, positions, mask, time_values, labels, color_map)
        fig.suptitle("Morphology-Axis Space", fontsize=16, fontweight="bold", y=0.97)
        ax_init.set_title("Initialization", fontsize=13, pad=6)
        ax_final.set_title("Condensed", fontsize=13, pad=6)
        ax_init.view_init(elev=ELEV, azim=azimuths[frame])
        ax_final.view_init(elev=ELEV, azim=azimuths[frame])
        return []

    anim = mpl_animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False)
    anim.save(str(output_path), writer="pillow", fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    run = tc.load_run(RUN_PATH)
    labels = _normalize_labels(run.labels)
    resolved_cmap = _resolve_color_map(labels, PRESENTATION_COLOR_MAP)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _render_single(run.x0, run.mask, run.time_values, labels, resolved_cmap,
                   "Initialization", OUT_DIR / "init_rotation.gif")

    _render_single(run.positions, run.mask, run.time_values, labels, resolved_cmap,
                   "Condensed", OUT_DIR / "final_rotation_pres.gif")

    _render_side_by_side(run.x0, run.positions, run.mask, run.time_values, labels, resolved_cmap,
                         OUT_DIR / "init_vs_final_rotation.gif")


if __name__ == "__main__":
    main()
