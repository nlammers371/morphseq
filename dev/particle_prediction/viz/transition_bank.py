"""Visualization helpers for resampling and transition-bank inspection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dev.particle_prediction.data.resampling import ResampledTrajectory
from dev.particle_prediction.data.transition_bank import TransitionBank
from dev.particle_prediction.data.transition_windows import TransitionWindow


def plot_arc_length_vs_time(trajectory: ResampledTrajectory) -> plt.Figure:
    """Plot resampled arc length against interpolated source time."""

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(trajectory.arc_length, trajectory.source_time_interp, linewidth=2.0)
    axis.set_xlabel("arc length")
    axis.set_ylabel("interpolated source time (s)")
    axis.set_title(f"Arc length vs time: {trajectory.source.embryo_id}")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_resampled_points_on_trajectory(
    trajectory: ResampledTrajectory,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Overlay resampled points on the smoothed latent trajectory."""

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    smoothed = trajectory.smoothed_source.smoothed
    axis.plot(smoothed[:, dims[0]], smoothed[:, dims[1]], linewidth=1.5, alpha=0.45, label="smoothed path")
    axis.scatter(
        trajectory.resampled[:, dims[0]],
        trajectory.resampled[:, dims[1]],
        c=trajectory.arc_length,
        cmap="viridis",
        s=18,
        label="resampled points",
    )
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title(f"Fixed-step resampling: {trajectory.source.embryo_id}")
    axis.legend(frameon=False, loc="best")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_increment_norm_distribution(trajectories: list[ResampledTrajectory]) -> plt.Figure:
    """Plot the distribution of resampled increment norms."""

    increment_norms = np.concatenate(
        [trajectory.increment_norms for trajectory in trajectories if trajectory.increment_norms is not None]
    )
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.hist(increment_norms, bins=30, color="#3a6ea5", alpha=0.85)
    axis.set_xlabel("increment norm")
    axis.set_ylabel("count")
    axis.set_title("Resampled increment norm distribution")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_transition_windows_for_embryo(
    trajectory: ResampledTrajectory,
    windows: list[TransitionWindow],
    dims: tuple[int, int] = (0, 1),
    max_windows: int = 8,
) -> plt.Figure:
    """Show example transition windows overlaid on one embryo trajectory."""

    fig, axis = plt.subplots(figsize=(7.0, 5.5))
    points = trajectory.resampled
    axis.plot(points[:, dims[0]], points[:, dims[1]], linewidth=1.5, alpha=0.35, color="black")

    for window in windows[:max_windows]:
        history_points = np.empty((window.history_segments.shape[0] + 1, window.history_segments.shape[1]), dtype=np.float64)
        history_points[-1] = window.state
        for index in range(window.history_segments.shape[0] - 1, -1, -1):
            history_points[index] = history_points[index + 1] - window.history_segments[index]
        axis.plot(history_points[:, dims[0]], history_points[:, dims[1]], alpha=0.8)
        axis.scatter(window.state[dims[0]], window.state[dims[1]], s=28, color="tab:red")

    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title(f"Transition window examples: {trajectory.source.embryo_id}")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_history_segments_example(
    window: TransitionWindow,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Visualize one canonical ordered history plus its next increment."""

    fig, axis = plt.subplots(figsize=(6.5, 5.0))
    history_points = np.empty((window.history_segments.shape[0] + 1, window.history_segments.shape[1]), dtype=np.float64)
    history_points[-1] = window.state
    for index in range(window.history_segments.shape[0] - 1, -1, -1):
        history_points[index] = history_points[index + 1] - window.history_segments[index]

    axis.plot(history_points[:, dims[0]], history_points[:, dims[1]], marker="o", linewidth=2.0, label="history")
    next_point = window.state + window.increment
    axis.arrow(
        window.state[dims[0]],
        window.state[dims[1]],
        window.increment[dims[0]],
        window.increment[dims[1]],
        length_includes_head=True,
        head_width=0.05,
        color="tab:red",
    )
    axis.scatter(next_point[dims[0]], next_point[dims[1]], color="tab:red", s=10, label="next state")
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title(f"History segments example: {window.embryo_id}")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_bank_state_density(
    bank: TransitionBank,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Scatter the bank state density in two latent dimensions."""

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    scatter = axis.scatter(
        bank.state_matrix[:, dims[0]],
        bank.state_matrix[:, dims[1]],
        c=np.linalg.norm(bank.increment_matrix, axis=1),
        cmap="magma",
        s=10,
        alpha=0.8,
    )
    fig.colorbar(scatter, ax=axis, label="next-step increment norm")
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title("Transition-bank state density")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_arc_length_vs_time",
    "plot_bank_state_density",
    "plot_history_segments_example",
    "plot_increment_norm_distribution",
    "plot_resampled_points_on_trajectory",
    "plot_transition_windows_for_embryo",
]
