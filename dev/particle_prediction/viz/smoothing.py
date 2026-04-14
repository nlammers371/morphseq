"""Visualization helpers for smoothing diagnostics."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory, smooth_trajectory


def _resolve_dims(array: np.ndarray, dims: Sequence[int] | None, max_dims: int) -> list[int]:
    n_dims = array.shape[1]
    if dims is None:
        return list(range(min(n_dims, max_dims)))
    return [dim for dim in dims if 0 <= dim < n_dims]


def plot_raw_vs_smoothed_timeseries(
    trajectory: SmoothedTrajectory,
    dims: Sequence[int] | None = None,
    max_dims: int = 3,
) -> plt.Figure:
    """Plot raw and smoothed latent coordinates against time."""

    dims = _resolve_dims(trajectory.smoothed, dims=dims, max_dims=max_dims)
    fig, axes = plt.subplots(len(dims), 1, figsize=(9, 2.8 * len(dims)), sharex=True)
    axes = [axes] if not isinstance(axes, np.ndarray) else list(axes)

    for axis, dim in zip(axes, dims):
        axis.plot(trajectory.time_seconds, trajectory.source.trajectory[:, dim], label="raw", alpha=0.7, linewidth=1.5)
        axis.plot(trajectory.time_seconds, trajectory.smoothed[:, dim], label="smoothed", linewidth=2.0)
        axis.set_ylabel(f"dim {dim}")
        axis.grid(alpha=0.2)

    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Smoothing diagnostics for {trajectory.source.embryo_id}", y=1.02)
    fig.tight_layout()
    return fig


def plot_latent_trajectory_before_after_smoothing(
    trajectory: SmoothedTrajectory,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Overlay raw and smoothed latent trajectories in two dimensions."""

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.plot(
        trajectory.source.trajectory[:, dims[0]],
        trajectory.source.trajectory[:, dims[1]],
        marker="o",
        markersize=3,
        linewidth=1.25,
        alpha=0.6,
        label="raw",
    )
    axis.plot(
        trajectory.smoothed[:, dims[0]],
        trajectory.smoothed[:, dims[1]],
        marker="o",
        markersize=3,
        linewidth=2.0,
        label="smoothed",
    )
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title(f"Latent trajectory before/after smoothing: {trajectory.source.embryo_id}")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_sg_parameter_sweep(
    source: EmbryoTrajectory,
    window_seconds_values: Sequence[float],
    poly_order: int = 2,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Plot a small SG parameter sweep on one trajectory."""

    n_panels = len(window_seconds_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.8), sharex=True, sharey=True)
    axes = [axes] if not isinstance(axes, np.ndarray) else list(axes)

    for axis, window_seconds in zip(axes, window_seconds_values):
        smoothed = smooth_trajectory(source=source, window_seconds=float(window_seconds), poly_order=poly_order)
        axis.plot(source.trajectory[:, dims[0]], source.trajectory[:, dims[1]], alpha=0.25, linewidth=1.0, label="raw")
        axis.plot(smoothed.smoothed[:, dims[0]], smoothed.smoothed[:, dims[1]], linewidth=2.0, label="smoothed")
        axis.set_title(f"window={window_seconds:g}s\nframes={smoothed.window_frames}")
        axis.set_xlabel(f"latent dim {dims[0]}")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel(f"latent dim {dims[1]}")
    axes[0].legend(frameon=False)
    fig.suptitle(f"SG window sweep for {source.embryo_id}", y=1.03)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_latent_trajectory_before_after_smoothing",
    "plot_raw_vs_smoothed_timeseries",
    "plot_sg_parameter_sweep",
]
