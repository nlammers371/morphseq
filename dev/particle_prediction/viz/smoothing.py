"""Visualization primitives for smoothing diagnostics."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..data.loading import EmbryoTrajectory
from ..data.smoothing import SmoothedTrajectory, smooth_trajectory


def _resolve_dims(n_dims: int, dims: Optional[Sequence[int]], max_dims: int = 3) -> Sequence[int]:
    if dims is not None:
        return dims
    return list(range(min(n_dims, max_dims)))


def plot_raw_vs_smoothed_timeseries(
    raw: EmbryoTrajectory,
    smoothed: SmoothedTrajectory,
    *,
    dims: Optional[Sequence[int]] = None,
    max_dims: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Plot raw and smoothed latent coordinates against time."""

    dims_to_plot = list(_resolve_dims(raw.trajectory.shape[1], dims, max_dims=max_dims))
    if figsize is None:
        figsize = (10, 3.0 * len(dims_to_plot))

    fig, axes = plt.subplots(len(dims_to_plot), 1, figsize=figsize, sharex=True)
    if len(dims_to_plot) == 1:
        axes = [axes]

    for axis, dim in zip(axes, dims_to_plot):
        axis.plot(raw.time_seconds, raw.trajectory[:, dim], color="#9AA0A6", linewidth=1.2, label="Raw")
        axis.plot(smoothed.time_seconds, smoothed.smoothed[:, dim], color="#1B7F79", linewidth=2.0, label="Smoothed")
        axis.set_ylabel(f"PC{dim}")
        axis.grid(alpha=0.3)

    axes[0].set_title(
        f"Raw vs smoothed: {raw.embryo_id} | window={smoothed.window_frames} frames | poly={smoothed.poly_order}"
    )
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_latent_trajectory_before_after_smoothing(
    raw: EmbryoTrajectory,
    smoothed: SmoothedTrajectory,
    *,
    dims: Tuple[int, int] = (0, 1),
    figsize: Tuple[float, float] = (7, 6),
) -> Figure:
    """Overlay one trajectory in latent space before and after smoothing."""

    x_dim, y_dim = dims
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    axis.plot(raw.trajectory[:, x_dim], raw.trajectory[:, y_dim], color="#B3B3B3", linewidth=1.2, label="Raw")
    axis.plot(
        smoothed.smoothed[:, x_dim],
        smoothed.smoothed[:, y_dim],
        color="#D95F02",
        linewidth=2.0,
        label="Smoothed",
    )
    axis.scatter(raw.trajectory[0, x_dim], raw.trajectory[0, y_dim], color="#1B9E77", s=30, label="Start")
    axis.scatter(raw.trajectory[-1, x_dim], raw.trajectory[-1, y_dim], color="#7570B3", s=30, label="End")
    axis.set_xlabel(f"PC{x_dim}")
    axis.set_ylabel(f"PC{y_dim}")
    axis.set_title(f"Latent trajectory smoothing overlay: {raw.embryo_id}")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_sg_parameter_sweep(
    raw: EmbryoTrajectory,
    *,
    dim: int = 0,
    window_frame_options: Sequence[int] = (5, 7, 9),
    poly_order: int = 2,
    figsize: Tuple[float, float] = (10, 5),
) -> Figure:
    """Compare several Savitzky-Golay window lengths on one dimension."""

    fig, axis = plt.subplots(1, 1, figsize=figsize)
    axis.plot(raw.time_seconds, raw.trajectory[:, dim], color="#BDBDBD", linewidth=1.0, label="Raw")

    cmap = plt.get_cmap("viridis")
    for index, window_frames in enumerate(window_frame_options):
        smoothed = smooth_trajectory(raw, window_frames=window_frames, poly_order=poly_order)
        label = f"{smoothed.window_frames} frames"
        axis.plot(
            smoothed.time_seconds,
            smoothed.smoothed[:, dim],
            color=cmap(index / max(1, len(window_frame_options) - 1)),
            linewidth=2.0,
            label=label,
        )

    axis.set_xlabel("Time (s)")
    axis.set_ylabel(f"PC{dim}")
    axis.set_title(f"Savitzky-Golay parameter sweep: {raw.embryo_id}")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig