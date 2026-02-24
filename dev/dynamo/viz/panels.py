"""Three-panel visualization (model spec §12).

Panel 1: Trajectory view — developmental stage vs morphological variation
Panel 2: Prediction fan — forward simulations for a single embryo
Panel 3: Phenotype space — c_e mode loadings in 2D PCA

All functions return matplotlib Figure objects. They work with or without
a trained model by using PC1/PC2 as fallback axes when phi0 is unavailable.

Model spec reference: §12.1–12.4.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from ..data.loading import EmbryoTrajectory, TrajectoryDataset
from ..eval.predictions import PredictionResult


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

# Colorblind-friendly palette
_CLASS_COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
]


def _get_class_colors(class_names: List[str]) -> Dict[str, str]:
    """Assign consistent colors to perturbation classes."""
    return {name: _CLASS_COLORS[i % len(_CLASS_COLORS)] for i, name in enumerate(class_names)}


# ---------------------------------------------------------------------------
# Panel 1: Trajectory View (spec §12.1)
# ---------------------------------------------------------------------------

def trajectory_view(
    dataset: TrajectoryDataset,
    phi0_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    predictions: Optional[Dict[str, List[PredictionResult]]] = None,
    max_trajectories: int = 50,
    figsize: Tuple[float, float] = (10, 7),
) -> Figure:
    """Trajectory view: developmental stage vs first residual PC.

    When phi0_fn is provided, x-axis is phi0(z) and y-axis is first residual PC
    (PCA on displacements orthogonal to grad(phi0)). When phi0_fn is None,
    falls back to PC1 (x) vs PC2 (y) as a proxy.

    Args:
        dataset: TrajectoryDataset with loaded trajectories.
        phi0_fn: Optional function mapping (N, D) arrays to (N,) developmental stage.
            When None, PC1 is used as x-axis proxy.
        predictions: Optional dict of model_name → list of PredictionResult,
            one per trajectory. Predictions are overlaid as dashed lines.
        max_trajectories: Maximum number of trajectories to plot (random subset).
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    trajs = dataset.trajectories
    if len(trajs) > max_trajectories:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(trajs), max_trajectories, replace=False)
        trajs = [trajs[i] for i in sorted(indices)]

    colors = _get_class_colors(dataset.class_names)

    if phi0_fn is not None:
        xlabel = r"$\phi_0(z)$ (developmental stage)"
        ylabel = "First residual PC"
    else:
        xlabel = "PC 1"
        ylabel = "PC 2"

    for traj in trajs:
        if phi0_fn is not None:
            x = phi0_fn(traj.trajectory)
            # Residual PC: project out the phi0 gradient direction
            # For now, use PC2 as a proxy (proper residual PCA requires grad(phi0))
            y = traj.trajectory[:, 1] if traj.trajectory.shape[1] > 1 else np.zeros(len(x))
        else:
            x = traj.trajectory[:, 0]
            y = traj.trajectory[:, 1] if traj.trajectory.shape[1] > 1 else np.zeros(len(x))

        color = colors.get(traj.perturbation_class, "#888888")
        ax.plot(x, y, "-", color=color, alpha=0.5, linewidth=0.8)
        ax.plot(x[0], y[0], "o", color=color, markersize=3, alpha=0.7)
        ax.plot(x[-1], y[-1], "s", color=color, markersize=3, alpha=0.7)

    # Legend for classes
    for name, color in colors.items():
        ax.plot([], [], "-", color=color, label=name, linewidth=2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Trajectory View", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 2: Prediction Fan (spec §12.2)
# ---------------------------------------------------------------------------

def prediction_fan(
    context: np.ndarray,
    target: Optional[np.ndarray] = None,
    forward_samples: Optional[np.ndarray] = None,
    predicted_mean: Optional[np.ndarray] = None,
    predicted_std: Optional[np.ndarray] = None,
    kernel_predictions: Optional[np.ndarray] = None,
    phi0_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    title: str = "Prediction Fan",
    figsize: Tuple[float, float] = (10, 7),
) -> Figure:
    """Prediction fan for a single embryo.

    Shows the observed context fragment, model predictions (samples or
    Gaussian envelope), kernel baseline predictions, and observed target.

    Args:
        context: (L, D) observed context fragment in PC space.
        target: (D,) optional observed future target.
        forward_samples: (n_samples, D) optional SDE forward samples.
        predicted_mean: (D,) predicted target mean.
        predicted_std: (D,) predicted target std per dimension.
        kernel_predictions: (n_points, D) optional kernel baseline point cloud.
        phi0_fn: Optional developmental stage function. Falls back to PC1/PC2.
        title: Plot title.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if phi0_fn is not None:
        get_xy = lambda pts: (phi0_fn(pts), pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts)))
    else:
        get_xy = lambda pts: (pts[:, 0], pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts)))

    # Context fragment
    cx, cy = get_xy(context)
    ax.plot(cx, cy, "k-", linewidth=2, label="Context", zorder=5)
    ax.plot(cx[0], cy[0], "ko", markersize=6, zorder=5)
    ax.plot(cx[-1], cy[-1], "k>", markersize=8, zorder=5)

    # Forward samples (SDE trajectories)
    if forward_samples is not None and len(forward_samples) > 0:
        sx, sy = get_xy(forward_samples)
        ax.scatter(sx, sy, c="#4477AA", alpha=0.15, s=10, label="Model samples", zorder=3)

    # Predicted mean + uncertainty
    if predicted_mean is not None:
        pm = predicted_mean.reshape(1, -1)
        px, py = get_xy(pm)
        ax.plot(px[0], py[0], "*", color="#EE6677", markersize=15,
                markeredgecolor="k", markeredgewidth=0.5, label="Predicted mean", zorder=6)

        if predicted_std is not None:
            # Draw 1σ and 2σ ellipses (approximate in 2D projection)
            for n_sigma, alpha in [(1, 0.3), (2, 0.15)]:
                from matplotlib.patches import Ellipse
                if phi0_fn is None:
                    w = 2 * n_sigma * predicted_std[0]
                    h = 2 * n_sigma * (predicted_std[1] if len(predicted_std) > 1 else predicted_std[0])
                else:
                    # Approximate: use std of first two components
                    w = 2 * n_sigma * predicted_std[0]
                    h = 2 * n_sigma * (predicted_std[1] if len(predicted_std) > 1 else predicted_std[0])
                ellipse = Ellipse(
                    (px[0], py[0]), w, h,
                    facecolor="#EE6677", alpha=alpha, edgecolor="none", zorder=2
                )
                ax.add_patch(ellipse)

    # Kernel baseline predictions
    if kernel_predictions is not None and len(kernel_predictions) > 0:
        kx, ky = get_xy(kernel_predictions)
        ax.scatter(kx, ky, c="#228833", alpha=0.3, s=15, marker="x",
                   label="Kernel baseline", zorder=4)

    # Observed target
    if target is not None:
        t = target.reshape(1, -1)
        tx, ty = get_xy(t)
        ax.plot(tx[0], ty[0], "D", color="#CCBB44", markersize=10,
                markeredgecolor="k", markeredgewidth=0.5, label="Observed target", zorder=7)

    xlabel = r"$\phi_0(z)$" if phi0_fn else "PC 1"
    ylabel = "Residual PC 1" if phi0_fn else "PC 2"

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Panel 3: Phenotype Space (spec §12.3)
# ---------------------------------------------------------------------------

def phenotype_space(
    mode_loadings: np.ndarray,
    class_labels: np.ndarray,
    class_names: List[str],
    novel_mask: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (8, 7),
) -> Figure:
    """Phenotype space: c_e mode loadings projected to 2D PCA.

    Args:
        mode_loadings: (N, M) mode loading vectors c_e for N embryos.
        class_labels: (N,) integer class labels.
        class_names: List of class names (index-aligned with class_labels).
        novel_mask: (N,) boolean mask for novel/held-out perturbation embryos.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure.
    """
    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # PCA on mode loadings
    M = mode_loadings.shape[1]
    n_pcs = min(2, M)
    if M > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(mode_loadings)
        xlabel = f"Loading PC 1 ({pca.explained_variance_ratio_[0]*100:.0f}%)"
        ylabel = f"Loading PC 2 ({pca.explained_variance_ratio_[1]*100:.0f}%)"
    elif M == 2:
        coords = mode_loadings
        xlabel = "$c_1$"
        ylabel = "$c_2$"
    else:
        # M == 1: plot as 1D with jitter
        coords = np.column_stack([mode_loadings[:, 0], np.random.randn(len(mode_loadings)) * 0.1])
        xlabel = "$c_1$"
        ylabel = "(jitter)"

    colors = _get_class_colors(class_names)

    for i, name in enumerate(class_names):
        mask = class_labels == i
        if not mask.any():
            continue
        marker = "o"
        alpha = 0.6
        edgecolors = "none"

        if novel_mask is not None:
            # Plot known and novel separately
            known = mask & ~novel_mask
            novel = mask & novel_mask

            if known.any():
                ax.scatter(coords[known, 0], coords[known, 1],
                           c=colors[name], label=name, alpha=0.6,
                           edgecolors="none", s=40, zorder=3)
            if novel.any():
                ax.scatter(coords[novel, 0], coords[novel, 1],
                           c=colors[name], label=f"{name} (novel)",
                           alpha=0.9, edgecolors="k", linewidths=1.5,
                           s=80, marker="D", zorder=4)
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[name], label=name, alpha=0.6,
                       edgecolors="none", s=40, zorder=3)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Phenotype Space (mode loadings)", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-panel composite
# ---------------------------------------------------------------------------

def three_panel_figure(
    dataset: TrajectoryDataset,
    context: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    predicted_mean: Optional[np.ndarray] = None,
    predicted_std: Optional[np.ndarray] = None,
    forward_samples: Optional[np.ndarray] = None,
    mode_loadings: Optional[np.ndarray] = None,
    class_labels: Optional[np.ndarray] = None,
    phi0_fn: Optional[Callable] = None,
    figsize: Tuple[float, float] = (18, 5.5),
) -> Figure:
    """Combined three-panel figure (spec §12).

    Creates a single figure with trajectory view (left), prediction fan
    (center), and phenotype space (right). Panels with missing data are
    shown with placeholder text.

    Args:
        dataset: TrajectoryDataset for Panel 1.
        context: (L, D) context fragment for Panel 2.
        target: (D,) target for Panel 2.
        predicted_mean: (D,) predicted mean for Panel 2.
        predicted_std: (D,) predicted std for Panel 2.
        forward_samples: (n_samples, D) forward samples for Panel 2.
        mode_loadings: (N, M) mode loadings for Panel 3.
        class_labels: (N,) class labels for Panel 3.
        phi0_fn: Optional developmental stage function.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure with three panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Trajectory View
    _draw_trajectory_view(axes[0], dataset, phi0_fn)

    # Panel 2: Prediction Fan
    if context is not None:
        _draw_prediction_fan(
            axes[1], context, target, forward_samples,
            predicted_mean, predicted_std, phi0_fn
        )
    else:
        axes[1].text(0.5, 0.5, "No prediction data\navailable yet",
                     ha="center", va="center", fontsize=12, color="gray",
                     transform=axes[1].transAxes)
        axes[1].set_title("Prediction Fan", fontsize=13)

    # Panel 3: Phenotype Space
    if mode_loadings is not None and class_labels is not None:
        _draw_phenotype_space(axes[2], mode_loadings, class_labels, dataset.class_names)
    else:
        axes[2].text(0.5, 0.5, "No mode loadings\navailable yet",
                     ha="center", va="center", fontsize=12, color="gray",
                     transform=axes[2].transAxes)
        axes[2].set_title("Phenotype Space", fontsize=13)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal drawing helpers (draw onto existing axes)
# ---------------------------------------------------------------------------

def _draw_trajectory_view(
    ax: plt.Axes,
    dataset: TrajectoryDataset,
    phi0_fn: Optional[Callable] = None,
    max_trajectories: int = 50,
) -> None:
    """Draw trajectory view on an existing axes."""
    trajs = dataset.trajectories
    if len(trajs) > max_trajectories:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(trajs), max_trajectories, replace=False)
        trajs = [trajs[i] for i in sorted(indices)]

    colors = _get_class_colors(dataset.class_names)

    for traj in trajs:
        if phi0_fn is not None:
            x = phi0_fn(traj.trajectory)
            y = traj.trajectory[:, 1] if traj.trajectory.shape[1] > 1 else np.zeros(len(x))
        else:
            x = traj.trajectory[:, 0]
            y = traj.trajectory[:, 1] if traj.trajectory.shape[1] > 1 else np.zeros(len(x))

        color = colors.get(traj.perturbation_class, "#888888")
        ax.plot(x, y, "-", color=color, alpha=0.5, linewidth=0.8)

    for name, color in colors.items():
        ax.plot([], [], "-", color=color, label=name, linewidth=2)
    ax.legend(fontsize=7, framealpha=0.8)

    xlabel = r"$\phi_0(z)$" if phi0_fn else "PC 1"
    ylabel = "Residual PC" if phi0_fn else "PC 2"
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title("Trajectory View", fontsize=12)
    ax.grid(True, alpha=0.3)


def _draw_prediction_fan(
    ax: plt.Axes,
    context: np.ndarray,
    target: Optional[np.ndarray],
    forward_samples: Optional[np.ndarray],
    predicted_mean: Optional[np.ndarray],
    predicted_std: Optional[np.ndarray],
    phi0_fn: Optional[Callable] = None,
) -> None:
    """Draw prediction fan on an existing axes."""
    if phi0_fn is not None:
        get_xy = lambda pts: (phi0_fn(pts), pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts)))
    else:
        get_xy = lambda pts: (pts[:, 0], pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts)))

    cx, cy = get_xy(context)
    ax.plot(cx, cy, "k-", linewidth=2, label="Context", zorder=5)
    ax.plot(cx[-1], cy[-1], "k>", markersize=6, zorder=5)

    if forward_samples is not None and len(forward_samples) > 0:
        sx, sy = get_xy(forward_samples)
        ax.scatter(sx, sy, c="#4477AA", alpha=0.15, s=8, zorder=3, label="Samples")

    if predicted_mean is not None:
        pm = predicted_mean.reshape(1, -1)
        px, py = get_xy(pm)
        ax.plot(px[0], py[0], "*", color="#EE6677", markersize=12,
                markeredgecolor="k", markeredgewidth=0.5, label="Mean", zorder=6)

    if target is not None:
        t = target.reshape(1, -1)
        tx, ty = get_xy(t)
        ax.plot(tx[0], ty[0], "D", color="#CCBB44", markersize=8,
                markeredgecolor="k", markeredgewidth=0.5, label="Target", zorder=7)

    xlabel = r"$\phi_0(z)$" if phi0_fn else "PC 1"
    ylabel = "Residual PC" if phi0_fn else "PC 2"
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title("Prediction Fan", fontsize=12)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.3)


def _draw_phenotype_space(
    ax: plt.Axes,
    mode_loadings: np.ndarray,
    class_labels: np.ndarray,
    class_names: List[str],
) -> None:
    """Draw phenotype space on an existing axes."""
    from sklearn.decomposition import PCA

    M = mode_loadings.shape[1]
    if M > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(mode_loadings)
        xlabel = f"Loading PC 1 ({pca.explained_variance_ratio_[0]*100:.0f}%)"
        ylabel = f"Loading PC 2 ({pca.explained_variance_ratio_[1]*100:.0f}%)"
    elif M == 2:
        coords = mode_loadings
        xlabel = "$c_1$"
        ylabel = "$c_2$"
    else:
        coords = np.column_stack([mode_loadings[:, 0], np.random.randn(len(mode_loadings)) * 0.1])
        xlabel = "$c_1$"
        ylabel = "(jitter)"

    colors = _get_class_colors(class_names)

    for i, name in enumerate(class_names):
        mask = class_labels == i
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[name], label=name, alpha=0.6,
                   edgecolors="none", s=30, zorder=3)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title("Phenotype Space", fontsize=12)
    ax.legend(fontsize=7, framealpha=0.8, loc="best")
    ax.grid(True, alpha=0.3)
