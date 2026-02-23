"""
Visualization functions for classification-based difference detection.

Provides AUROC curve plotting for results from ``run_classification_test``.

Migrated and adapted from
``results/mcolon/20260105_refined_embedding_and_metric_classification/utils/plotting_functions.py``

Key changes from the source:
- Uses ``auroc_obs`` (canonical column from ``run_classification_test`` output)
  instead of ``auroc_observed``
- Accepts ``MulticlassOVRResults`` directly where appropriate
- Adds ``plot_feature_comparison_grid`` for side-by-side feature-type panels
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from analyze.viz.styling import resolve_color_lookup
from .results import MulticlassOVRResults

__all__ = [
    "plot_auroc_with_null",
    "plot_multiple_aurocs",
    "plot_multiclass_ovr_aurocs",
    "plot_feature_comparison_grid",
]

# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

_AUROC_COL_CANDIDATES = ("auroc_obs", "auroc_observed")


def _auroc_col(df: pd.DataFrame) -> str:
    """Return whichever AUROC column is present."""
    for c in _AUROC_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"No AUROC column found. Expected one of {_AUROC_COL_CANDIDATES}. "
        f"Available: {list(df.columns)}"
    )


def _time_col(df: pd.DataFrame, preferred: str = "time_bin_center") -> str:
    if preferred in df.columns:
        return preferred
    if "time_bin" in df.columns:
        return "time_bin"
    raise ValueError("No time column found (expected 'time_bin_center' or 'time_bin')")


# ---------------------------------------------------------------------------
# Core: single AUROC curve
# ---------------------------------------------------------------------------


def plot_auroc_with_null(
    ax: plt.Axes,
    auroc_df: pd.DataFrame,
    color: str,
    label: str,
    style: str = "-",
    time_col: str = "time_bin_center",
    show_null_band: bool = True,
    show_significance: bool = True,
    sig_threshold: float = 0.01,
    sig_marker_size: int = 200,
) -> None:
    """Plot a single AUROC curve with null-distribution band and significance markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    auroc_df : pd.DataFrame
        Must contain an AUROC column (``auroc_obs`` or ``auroc_observed``),
        and optionally ``auroc_null_mean``, ``auroc_null_std``, ``pval``.
    color, label, style : str
    time_col : str
    show_null_band, show_significance : bool
    sig_threshold : float
        P-value threshold for significance markers (default 0.01).
    sig_marker_size : int
    """
    acol = _auroc_col(auroc_df)
    tcol = _time_col(auroc_df, preferred=time_col)

    ax.plot(
        auroc_df[tcol],
        auroc_df[acol],
        f"o{style}",
        label=label,
        color=color,
        linewidth=2,
        markersize=5,
    )

    if (
        show_null_band
        and "auroc_null_mean" in auroc_df.columns
        and "auroc_null_std" in auroc_df.columns
    ):
        ax.fill_between(
            auroc_df[tcol],
            auroc_df["auroc_null_mean"] - auroc_df["auroc_null_std"],
            auroc_df["auroc_null_mean"] + auroc_df["auroc_null_std"],
            color=color,
            alpha=0.25,
            linewidth=0,
        )

    if show_significance and "pval" in auroc_df.columns:
        sig_mask = auroc_df["pval"] <= sig_threshold
        if sig_mask.any():
            ax.scatter(
                auroc_df.loc[sig_mask, tcol],
                auroc_df.loc[sig_mask, acol],
                s=sig_marker_size,
                facecolors="none",
                edgecolors=color,
                linewidths=2.5,
                zorder=5,
            )


# ---------------------------------------------------------------------------
# Multiple AUROC curves on one axis
# ---------------------------------------------------------------------------


def _format_auroc_axis(ax: plt.Axes, title: str, ylim: Tuple[float, float], sig_threshold: float=0.01) -> None:
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance (0.5)")
    ax.scatter(
        [], [], s=200, facecolors="none", edgecolors="black",
        linewidths=2.5, label=f"p ≤ {sig_threshold}",
    )
    ax.set_xlabel("Hours Post Fertilization (hpf)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(title, fontsize=14)
    # ax.legend(loc="upper left", fontsize=9)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def plot_multiple_aurocs(
    auroc_dfs_dict: Dict[str, pd.DataFrame],
    colors_dict: Dict[str, str],
    styles_dict: Optional[Dict[str, str]] = None,
    title: str = "AUROC Comparison",
    figsize: Tuple[int, int] = (14, 7),
    ylim: Tuple[float, float] = (0.3, 1.05),
    time_col: str = "time_bin_center",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    sig_threshold: float = 0.01,
) -> plt.Figure:
    """Overlay multiple AUROC curves on one axis.

    Parameters
    ----------
    auroc_dfs_dict : dict[str, DataFrame]
        ``{label: auroc_df}``
    colors_dict : dict[str, str]
    styles_dict : dict[str, str] | None
    title, figsize, ylim, time_col : display options
    save_path : path-like | None
    ax : Axes | None
        If provided, plot on this axis instead of creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if styles_dict is None:
        styles_dict = {k: "-" for k in auroc_dfs_dict}

    labels = list(auroc_dfs_dict.keys())
    resolved_colors = resolve_color_lookup(
        labels,
        color_lookup=colors_dict,
        enforce_distinct=True,
        warn_on_collision=True,
    )

    for label, auroc_df in auroc_dfs_dict.items():
        if auroc_df is None or auroc_df.empty:
            continue
        plot_auroc_with_null(
            ax=ax,
            auroc_df=auroc_df,
            color=resolved_colors.get(label, "#000000"),
            label=label,
            style=styles_dict.get(label, "-"),
            time_col=time_col,
            sig_threshold=sig_threshold,
        )

    _format_auroc_axis(ax, title, ylim, sig_threshold)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# OvR wrapper (thin convenience)
# ---------------------------------------------------------------------------


def plot_multiclass_ovr_aurocs(
    ovr_results: Union[MulticlassOVRResults, Dict[str, pd.DataFrame]],
    colors_dict: Dict[str, str],
    title: str = "Per-Class OvR AUROC",
    figsize: Tuple[int, int] = (12, 7),
    ylim: Tuple[float, float] = (0.3, 1.05),
    time_col: str = "time_bin_center",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    sig_threshold: float = 0.01,
) -> plt.Figure:
    """Plot One-vs-Rest AUROC curves.

    Accepts either a ``MulticlassOVRResults`` or a plain dict of DataFrames.
    Labels are formatted as ``"{positive} vs {negative}"``.
    """
    if isinstance(ovr_results, MulticlassOVRResults):
        dfs_dict: Dict[str, pd.DataFrame] = {}
        for (pos, neg), df in ovr_results.items():
            dfs_dict[f"{pos} vs {neg}"] = df
    else:
        dfs_dict = ovr_results

    # Build colors for the generated labels
    label_colors = {}
    for lbl in dfs_dict:
        # Try to match the positive class name to colors_dict
        pos_part = lbl.split(" vs ")[0] if " vs " in lbl else lbl
        label_colors[lbl] = colors_dict.get(pos_part, colors_dict.get(lbl, "#000000"))

    return plot_multiple_aurocs(
        auroc_dfs_dict=dfs_dict,
        colors_dict=label_colors,
        title=title,
        figsize=figsize,
        ylim=ylim,
        time_col=time_col,
        save_path=save_path,
        ax=ax,
        sig_threshold=sig_threshold,
    )


# ---------------------------------------------------------------------------
# Feature comparison grid (new)
# ---------------------------------------------------------------------------


def plot_feature_comparison_grid(
    results_by_feature: Dict[str, MulticlassOVRResults],
    feature_labels: Dict[str, str],
    cluster_colors: Dict[str, str],
    title: str = "",
    ylim: Tuple[float, float] = (0.3, 1.05),
    figsize_per_panel: Tuple[float, float] = (6, 5),
    save_path: Optional[Union[str, Path]] = None,
    sig_threshold: float = 0.01,
) -> plt.Figure:
    """Side-by-side panels comparing feature types for the same comparisons.

    Parameters
    ----------
    results_by_feature : dict[str, MulticlassOVRResults]
        ``{feature_key: results}`` — one entry per panel.
    feature_labels : dict[str, str]
        Human-readable label for each feature key (used as panel title).
    cluster_colors : dict[str, str]
        Color for each *positive* class name.
    title : str
        Super-title for the whole figure.
    ylim : tuple
    figsize_per_panel : tuple
        Width × height of each panel.
    save_path : path-like | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(results_by_feature)
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), squeeze=False)

    for idx, (feat_key, results) in enumerate(results_by_feature.items()):
        ax = axes[0, idx]
        panel_title = feature_labels.get(feat_key, feat_key)
        plot_multiclass_ovr_aurocs(
            ovr_results=results,
            colors_dict=cluster_colors,
            title=panel_title,
            ax=ax,
            ylim=ylim,
            sig_threshold=sig_threshold,
        )

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Stubs (future)
# ---------------------------------------------------------------------------


def create_comparison_figure(*args, **kwargs):
    """TODO: 3-panel figure (AUROC + divergence + trajectories).

    Requires ``compare_groups`` to return divergence and trajectory data
    alongside classification results.
    """
    raise NotImplementedError("Planned for future; needs compare_groups update.")


def plot_temporal_emergence(*args, **kwargs):
    """TODO: Bar plot of earliest significant time bin per comparison.

    Requires ``compare_groups`` to return a summary dict with significance
    onset information.
    """
    raise NotImplementedError("Planned for future; needs compare_groups summary dict.")
