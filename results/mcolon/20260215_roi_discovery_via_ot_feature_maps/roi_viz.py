"""
Visualization for ROI discovery results.

Provides:
- Weight map heatmaps (magnitude on canonical grid)
- ROI contour overlays on reference mask
- Sweep Pareto front plots
- Null distribution histograms
- Bootstrap IoU distribution plots
- Objective convergence curves

Follows the plotting patterns from:
- src/analyze/optimal_transport_morphometrics/uot_masks/viz.py (heatmaps, overlays)
- src/analyze/difference_detection/classification_test_viz.py (AUROC plots)
- src/analyze/viz/styling/ (color palettes)

NOTE: Some integration with the existing viz infrastructure
(e.g., resolve_color_lookup, faceting_engine) is deferred until
the ROI results are validated and the pipeline is mature.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import matplotlib.cm as cm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display helpers (match UOT viz convention)
# ---------------------------------------------------------------------------

def _plot_extent(hw: Tuple[int, int]) -> Tuple[list, str]:
    """Return extent and origin for image display mode."""
    h, w = hw
    return [0, w, h, 0], "upper"


# ---------------------------------------------------------------------------
# Weight map visualization
# ---------------------------------------------------------------------------

def plot_weight_map(
    w_full: np.ndarray,
    mask_ref: np.ndarray,
    title: str = "Weight Map Magnitude",
    cmap: str = "inferno",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot the magnitude of the trained weight map on the canonical grid.

    Parameters
    ----------
    w_full : ndarray, (H, W, C)
        Full-resolution weight map.
    mask_ref : ndarray, (H, W)
        Reference embryo mask.
    title : str
    cmap : str
    ax : matplotlib Axes, optional
    save_path : str, optional

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure

    w_mag = np.sqrt(np.sum(w_full ** 2, axis=-1))  # (H, W)

    # Mask outside embryo
    w_display = np.where(mask_ref.astype(bool), w_mag, np.nan)

    H, W = w_display.shape
    extent, origin = _plot_extent((H, W))
    im = ax.imshow(w_display, cmap=cmap, extent=extent, origin=origin,
                   interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="|w|", shrink=0.8)

    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved weight map: {save_path}")

    return ax


def plot_roi_overlay(
    roi_mask: np.ndarray,
    mask_ref: np.ndarray,
    w_full: Optional[np.ndarray] = None,
    title: str = "ROI Overlay",
    roi_color: str = "red",
    roi_alpha: float = 0.4,
    show_contour: bool = True,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot ROI overlaid on the reference mask, optionally with weight heatmap.

    Parameters
    ----------
    roi_mask : ndarray, (H, W), bool
        Binary ROI mask.
    mask_ref : ndarray, (H, W)
        Reference embryo mask.
    w_full : ndarray, (H, W, C), optional
        If provided, show weight magnitude as background heatmap.
    title : str
    roi_color : str
    roi_alpha : float
    show_contour : bool
    ax : matplotlib Axes, optional
    save_path : str, optional
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure

    H, W = mask_ref.shape
    extent, origin = _plot_extent((H, W))

    # Background: either weight map or plain mask
    if w_full is not None:
        w_mag = np.sqrt(np.sum(w_full ** 2, axis=-1))
        bg = np.where(mask_ref.astype(bool), w_mag, np.nan)
        ax.imshow(bg, cmap="gray", extent=extent, origin=origin,
                  interpolation="bilinear", alpha=0.6)
    else:
        ax.imshow(mask_ref.astype(float), cmap="gray", extent=extent,
                  origin=origin, alpha=0.3)

    # ROI overlay
    roi_display = np.zeros((*roi_mask.shape, 4), dtype=float)
    color_rgb = matplotlib.colors.to_rgb(roi_color)
    roi_display[roi_mask, :3] = color_rgb
    roi_display[roi_mask, 3] = roi_alpha
    ax.imshow(roi_display, extent=extent, origin=origin, interpolation="nearest")

    # Contour
    if show_contour:
        ax.contour(roi_mask.astype(float), levels=[0.5],
                   colors=[roi_color], linewidths=1.5,
                   extent=extent, origin=origin)

    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    # Legend
    patches = [
        Patch(facecolor=roi_color, alpha=roi_alpha, label="ROI"),
        Patch(facecolor="gray", alpha=0.3, label="Embryo mask"),
    ]
    ax.legend(handles=patches, loc="upper right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROI overlay: {save_path}")

    return ax


# ---------------------------------------------------------------------------
# Sweep visualization
# ---------------------------------------------------------------------------

def plot_sweep_pareto(
    sweep_df: pd.DataFrame,
    selected_lam: Optional[float] = None,
    selected_mu: Optional[float] = None,
    title: str = "Sweep: AUROC vs ROI Complexity",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot the Pareto front from the λ/μ sweep.

    Parameters
    ----------
    sweep_df : DataFrame
        Sweep table with columns: lam, mu, auroc_mean, area_fraction_mean.
    selected_lam, selected_mu : float, optional
        Highlight the selected (λ, μ) point.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.figure

    scatter = ax.scatter(
        sweep_df["area_fraction_mean"],
        sweep_df["auroc_mean"],
        c=np.log10(sweep_df["lam"]),
        cmap="viridis",
        s=80,
        edgecolors="k",
        linewidths=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="log10(λ)")
    ax.set_xlabel("Area Fraction (complexity)")
    ax.set_ylabel("AUROC")
    ax.set_title(title)

    if selected_lam is not None and selected_mu is not None:
        sel_row = sweep_df[
            (np.isclose(sweep_df["lam"], selected_lam))
            & (np.isclose(sweep_df["mu"], selected_mu))
        ]
        if len(sel_row) > 0:
            ax.scatter(
                sel_row["area_fraction_mean"],
                sel_row["auroc_mean"],
                marker="*", s=300, c="red", zorder=5,
                label=f"Selected (λ={selected_lam:.2e})",
            )
            ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# Null distribution visualization
# ---------------------------------------------------------------------------

def plot_null_distribution(
    null_aurocs: np.ndarray,
    observed_auroc: float,
    pvalue: float,
    title: str = "Label Permutation Null",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Histogram of null AUROC distribution with observed value marked.

    Mirrors the style of plot_auroc_with_null from
    src/analyze/difference_detection/classification_test_viz.py.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.figure

    valid = null_aurocs[np.isfinite(null_aurocs)]
    ax.hist(valid, bins=30, alpha=0.7, color="steelblue", edgecolor="k",
            label="Null distribution")
    ax.axvline(observed_auroc, color="red", linewidth=2, linestyle="--",
               label=f"Observed AUROC={observed_auroc:.4f}\np={pvalue:.4f}")

    ax.set_xlabel("AUROC (selection-aware null)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_bootstrap_iou(
    iou_distribution: np.ndarray,
    title: str = "Bootstrap ROI Stability",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Histogram of bootstrap IoU distribution."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.figure

    mean_iou = np.mean(iou_distribution)
    std_iou = np.std(iou_distribution)

    ax.hist(iou_distribution, bins=30, alpha=0.7, color="darkorange", edgecolor="k")
    ax.axvline(mean_iou, color="red", linewidth=2,
               label=f"Mean IoU={mean_iou:.4f}±{std_iou:.4f}")

    ax.set_xlabel("IoU with Reference ROI")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

def plot_objective_curve(
    objective_log: list,
    title: str = "Training Objective",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot training objective components over steps.

    Shows total, logistic, L1, and TV components.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = ax.figure

    df = pd.DataFrame(objective_log)

    ax.plot(df["step"], df["total_objective"], "k-", linewidth=2, label="Total")
    ax.plot(df["step"], df["logistic_loss_raw"], "b--", label="Logistic")
    ax.plot(df["step"], df["l1_weighted"], "r--", label="λ·L1")
    ax.plot(df["step"], df["tv_weighted"], "g--", label="μ·TV")

    ax.set_xlabel("Step")
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# ---------------------------------------------------------------------------
# Full report figure
# ---------------------------------------------------------------------------

def plot_full_report(
    run_dir: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a multi-panel summary figure for a completed ROI run.

    Panels:
    1. Weight map heatmap
    2. ROI overlay
    3. Sweep Pareto front
    4. Null distribution
    5. Bootstrap IoU
    6. Objective convergence

    This is a convenience wrapper that loads data from the run directory.
    """
    run_path = Path(run_dir)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"ROI Discovery Report: {run_path.name}", fontsize=14, fontweight="bold")

    # Load what's available and fill panels
    panels_filled = 0

    # Panel 1-2: Weight map + ROI (need trained model output — placeholder)
    axes[0, 0].text(0.5, 0.5, "Weight Map\n(requires trained model)",
                    ha="center", va="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("Weight Map |w|")

    axes[0, 1].text(0.5, 0.5, "ROI Overlay\n(requires trained model)",
                    ha="center", va="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("ROI on Reference Mask")

    # Panel 3: Sweep Pareto
    sweep_csv = run_path / "sweep" / "sweep_table.csv"
    sel_json = run_path / "sweep" / "selection.json"
    if sweep_csv.exists():
        sweep_df = pd.read_csv(sweep_csv)
        sel_lam, sel_mu = None, None
        if sel_json.exists():
            with open(sel_json) as f:
                sel = json.load(f)
            sel_lam = sel.get("selected_lam")
            sel_mu = sel.get("selected_mu")
        plot_sweep_pareto(sweep_df, sel_lam, sel_mu, ax=axes[0, 2])
        panels_filled += 1

    # Panel 4: Null distribution
    null_npy = run_path / "nulls" / "null_aurocs.npy"
    nulls_json = run_path / "nulls" / "nulls_summary.json"
    if null_npy.exists() and nulls_json.exists():
        null_aurocs = np.load(null_npy)
        with open(nulls_json) as f:
            ns = json.load(f)
        if "permutation" in ns:
            plot_null_distribution(
                null_aurocs,
                ns["permutation"]["observed_auroc"],
                ns["permutation"]["pvalue"],
                ax=axes[1, 0],
            )
            panels_filled += 1

    # Panel 5: Bootstrap IoU
    iou_npy = run_path / "nulls" / "bootstrap_ious.npy"
    if iou_npy.exists():
        ious = np.load(iou_npy)
        plot_bootstrap_iou(ious, ax=axes[1, 1])
        panels_filled += 1

    # Panel 6: placeholder for convergence
    axes[1, 2].text(0.5, 0.5, "Objective Convergence\n(requires training log)",
                    ha="center", va="center", transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Training Convergence")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved full report: {save_path}")

    return fig


__all__ = [
    "plot_weight_map",
    "plot_roi_overlay",
    "plot_sweep_pareto",
    "plot_null_distribution",
    "plot_bootstrap_iou",
    "plot_objective_curve",
    "plot_full_report",
]
