"""
iteration_choice_plots.py
-------------------------
Visualization helpers for iteration ranking and candidate selection.

  plot_iteration_scores(scores, output_path)
      Geometry score curve with selected iterations highlighted.

  render_selected_iteration_bundle(...)
      Write the standard plot bundle for one selected candidate iteration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import plotting
from ..iteration_ranking import select_evenly_distributed


def plot_iteration_scores(scores, output_path: Path, *, title: str = "") -> None:
    if scores.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(scores["iter"], scores["geometry_score"], color="#2166AC", lw=2.0, label="geometry_score")
    top = select_evenly_distributed(scores, min(5, len(scores)))
    axes[0].scatter(top["iter"], top["geometry_score"], color="#B2182B", s=35, zorder=3, label="Selected (distributed)")
    axes[0].set_ylabel("geometry_score", fontsize=9)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    metric_cols = [
        ("score_separation", "sep"),
        ("score_compactness", "compact"),
        ("score_stability", "stable"),
        ("score_density_uniformity", "density_cv"),
    ]
    for col, label in metric_cols:
        if col in scores.columns:
            axes[1].plot(scores["iter"], scores[col], lw=1.5, label=label)
    axes[1].set_xlabel("iteration", fontsize=9)
    axes[1].set_ylabel("component score", fontsize=9)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, ncol=4)

    if title:
        fig.suptitle(title, fontsize=10, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_selected_iteration_bundle(
    *,
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray,
    color_map: dict[str, str],
    output_dir: Path,
    title_prefix: str,
    snapshot_iter: int,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = plotting.plot_trajectories(
        positions, mask, time_values,
        labels=labels, color_map=color_map,
        title=f"{title_prefix} | iter {snapshot_iter}",
    )
    fig.savefig(output_dir / "plot_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    snap_indices = np.linspace(0, len(time_values) - 1, min(6, len(time_values)), dtype=int)
    snapshot_times = [float(time_values[i]) for i in snap_indices]
    fig, _ = plotting.plot_panels(
        positions, mask, time_values,
        labels=labels, color_map=color_map,
        snapshot_times=snapshot_times,
        title=f"{title_prefix} | iter {snapshot_iter}",
    )
    fig.savefig(output_dir / "plot_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_stacked_3d(
        positions, mask, time_values,
        labels=labels, color_map=color_map,
        title=f"{title_prefix} | iter {snapshot_iter}",
    )
    fig.savefig(output_dir / "plot_stacked_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        output_dir / "positions_iter.npz",
        positions=positions,
        mask=mask,
        time_values=time_values,
        labels=labels,
        snapshot_iter=int(snapshot_iter),
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
