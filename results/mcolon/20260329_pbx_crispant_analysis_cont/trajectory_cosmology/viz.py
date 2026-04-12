"""
viz.py
------
Figures from CondensationResult and diagnostic DataFrames.

All plot functions return (fig, axes) so callers can save or adjust.
No file I/O here — saving is the caller's responsibility.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# lazy matplotlib import so this module can be imported without a display
def _mpl():
    import matplotlib.pyplot as plt
    return plt


def plot_trajectories(
    result,
    labels: np.ndarray,
    time_values: np.ndarray,
    color_map: dict[str, str] | None = None,
    alpha_line: float = 0.3,
    alpha_point: float = 0.6,
    figsize: tuple = (10, 8),
    title: str = "Condensed Trajectories",
):
    """Per-embryo trajectory lines colored by genotype.

    Parameters
    ----------
    result : CondensationResult
    labels : (N_e,) genotype strings
    time_values : (T,) hpf
    color_map : genotype → hex color. Uses default palette if None.

    Returns
    -------
    (fig, ax)
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=figsize)

    conditions = np.unique(labels)
    if color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {c: cmap(i) for i, c in enumerate(conditions)}

    positions = result.positions   # (N_e, T, 2)
    mask = result.mask

    for cond in conditions:
        idx = np.where(labels == cond)[0]
        color = color_map[cond]
        first = True
        for i in idx:
            obs_t = np.where(mask[i, :])[0]
            if len(obs_t) < 2:
                continue
            xs = positions[i, obs_t, 0]
            ys = positions[i, obs_t, 1]
            ax.plot(xs, ys, color=color, alpha=alpha_line, linewidth=0.8,
                    label=cond if first else "_nolegend_")
            ax.scatter(xs, ys, color=color, alpha=alpha_point, s=8, linewidths=0)
            first = False

    ax.legend(loc="best", framealpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    return fig, ax


def plot_loss_curve(
    loss_df: pd.DataFrame,
    figsize: tuple = (10, 4),
):
    """Stacked loss decomposition over iterations.

    Returns
    -------
    (fig, ax)
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=figsize)

    for term in ["attract", "repel", "elastic", "fidelity"]:
        if term in loss_df.columns:
            ax.plot(loss_df["iter"], loss_df[term], label=term)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.set_title("Condensation Loss Decomposition")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_bundle_centroids(
    centroids_df: pd.DataFrame,
    color_map: dict[str, str] | None = None,
    figsize: tuple = (10, 5),
):
    """Centroid trajectories per condition over time.

    Returns
    -------
    (fig, axes) — left: x/y over time; right: 2D centroid paths
    """
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    conditions = centroids_df["condition"].unique()
    if color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {c: cmap(i) for i, c in enumerate(conditions)}

    for cond in conditions:
        df = centroids_df[centroids_df["condition"] == cond].sort_values("time_bin_center")
        color = color_map[cond]
        axes[0].plot(df["time_bin_center"], df["x"], label=f"{cond} x", color=color, linestyle="-")
        axes[0].plot(df["time_bin_center"], df["y"], label=f"{cond} y", color=color, linestyle="--")
        axes[1].plot(df["x"], df["y"], color=color, label=cond, marker="o", markersize=4)

    axes[0].set_xlabel("hpf")
    axes[0].set_ylabel("Position")
    axes[0].set_title("Centroid coordinates over time")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title("2D centroid paths")
    axes[1].legend(loc="best", framealpha=0.7)
    fig.tight_layout()
    return fig, axes


def plot_bundle_width(
    width_df: pd.DataFrame,
    color_map: dict[str, str] | None = None,
    figsize: tuple = (8, 4),
):
    """Within-condition spread over time.

    Returns
    -------
    (fig, ax)
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=figsize)

    conditions = width_df["condition"].unique()
    if color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {c: cmap(i) for i, c in enumerate(conditions)}

    for cond in conditions:
        df = width_df[width_df["condition"] == cond].sort_values("time_bin_center")
        ax.plot(df["time_bin_center"], df["spread"], label=cond, color=color_map[cond])

    ax.set_xlabel("hpf")
    ax.set_ylabel("Mean distance from centroid")
    ax.set_title("Within-condition bundle width over time")
    ax.legend()
    fig.tight_layout()
    return fig, ax
