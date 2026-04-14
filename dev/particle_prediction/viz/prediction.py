"""Visualization helpers for the local transition kernel."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from dev.particle_prediction.eval.predictions import RolloutPredictionResult
from dev.particle_prediction.models.kernels import construct_tangent_aligned_covariance
from dev.particle_prediction.models.local_transition_pf import LocalPredictionResult


def plot_local_increment_cloud(
    query_state: np.ndarray,
    prediction: LocalPredictionResult,
    dims: tuple[int, int] = (0, 1),
    max_candidates: int = 64,
) -> plt.Figure:
    """Visualize the local candidate increment cloud anchored at the query."""

    increments = np.vstack([window.increment for window in prediction.match_result.candidate_windows[:max_candidates]])
    candidate_next_states = query_state[None, :] + increments

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    scatter = axis.scatter(
        candidate_next_states[:, dims[0]],
        candidate_next_states[:, dims[1]],
        c=prediction.match_result.normalized_weights[: len(candidate_next_states)],
        cmap="viridis",
        s=42,
        alpha=0.85,
    )
    axis.scatter(query_state[dims[0]], query_state[dims[1]], color="tab:red", marker="x", s=70, label="query")
    axis.scatter(
        prediction.predicted_mean[dims[0]],
        prediction.predicted_mean[dims[1]],
        color="black",
        s=65,
        label="weighted mean",
    )
    fig.colorbar(scatter, ax=axis, label="candidate weight")
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title("Local increment cloud")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_sampled_next_steps(
    query_state: np.ndarray,
    prediction: LocalPredictionResult,
    true_next_state: np.ndarray | None = None,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Plot sampled next states from the one-step kernel."""

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.scatter(
        prediction.forward_samples[:, dims[0]],
        prediction.forward_samples[:, dims[1]],
        s=18,
        alpha=0.35,
        color="#3a6ea5",
        label="samples",
    )
    axis.scatter(query_state[dims[0]], query_state[dims[1]], color="tab:red", marker="x", s=70, label="query")
    axis.scatter(
        prediction.predicted_mean[dims[0]],
        prediction.predicted_mean[dims[1]],
        color="black",
        s=65,
        label="predicted mean",
    )
    if true_next_state is not None:
        axis.scatter(true_next_state[dims[0]], true_next_state[dims[1]], color="tab:green", s=55, label="true next")
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title("Sampled next-step predictions")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_jitter_ellipse_or_covariance(
    reference_increment: np.ndarray,
    sigma_parallel: float,
    sigma_perp: float,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Plot the tangent-aligned jitter ellipse for a reference increment."""

    covariance = construct_tangent_aligned_covariance(
        reference_increment=np.asarray(reference_increment, dtype=np.float64),
        sigma_parallel=sigma_parallel,
        sigma_perp=sigma_perp,
    )
    cov_2d = covariance[np.ix_(dims, dims)]
    evals, evecs = np.linalg.eigh(cov_2d)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    width, height = 2.0 * np.sqrt(np.maximum(evals, 1.0e-12))

    fig, axis = plt.subplots(figsize=(6.0, 5.0))
    ellipse = Ellipse(
        xy=(reference_increment[dims[0]], reference_increment[dims[1]]),
        width=width,
        height=height,
        angle=angle,
        facecolor="#d97a3a",
        alpha=0.35,
        edgecolor="#7f3b08",
        linewidth=2.0,
    )
    axis.add_patch(ellipse)
    axis.scatter(reference_increment[dims[0]], reference_increment[dims[1]], color="black", s=45)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.3)
    axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.3)
    axis.set_xlabel(f"increment dim {dims[0]}")
    axis.set_ylabel(f"increment dim {dims[1]}")
    axis.set_title("Tangent-aligned jitter ellipse")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_prediction_fan(
    query_state: np.ndarray,
    rollout: RolloutPredictionResult,
    true_future: np.ndarray | None = None,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Plot rollout particle clouds and predictive means across horizons."""

    query_state = np.asarray(query_state, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.0, 5.8))
    n_steps = rollout.forward_samples.shape[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, n_steps))
    for step_index, color in enumerate(colors):
        axis.scatter(
            rollout.forward_samples[step_index, :, dims[0]],
            rollout.forward_samples[step_index, :, dims[1]],
            s=16,
            alpha=0.18,
            color=color,
        )
    path = np.vstack([query_state[None, :], rollout.predicted_mean])
    axis.plot(path[:, dims[0]], path[:, dims[1]], color="black", linewidth=2.1, label="predicted mean")
    axis.scatter(query_state[dims[0]], query_state[dims[1]], color="tab:red", marker="x", s=80, label="query")
    if true_future is not None:
        true_future = np.asarray(true_future, dtype=np.float64)
        truth_path = np.vstack([query_state[None, :], true_future])
        axis.plot(
            truth_path[:, dims[0]],
            truth_path[:, dims[1]],
            color="tab:green",
            linewidth=2.0,
            linestyle="--",
            label="true future",
        )
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title("Rollout prediction fan")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_rollout_against_truth(
    query_state: np.ndarray,
    rollout: RolloutPredictionResult,
    true_future: np.ndarray,
    context_points: np.ndarray | None = None,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Overlay query context, predictive mean, samples, and observed future."""

    query_state = np.asarray(query_state, dtype=np.float64)
    true_future = np.asarray(true_future, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.scatter(
        rollout.forward_samples[:, :, dims[0]].reshape(-1),
        rollout.forward_samples[:, :, dims[1]].reshape(-1),
        s=10,
        alpha=0.12,
        color="#3a6ea5",
        label="particle cloud",
    )
    if context_points is not None:
        context_points = np.asarray(context_points, dtype=np.float64)
        axis.plot(context_points[:, dims[0]], context_points[:, dims[1]], color="#888888", linewidth=2.0, label="context")
    axis.plot(
        np.vstack([query_state[None, :], rollout.predicted_mean])[:, dims[0]],
        np.vstack([query_state[None, :], rollout.predicted_mean])[:, dims[1]],
        color="black",
        linewidth=2.2,
        label="predicted mean",
    )
    axis.plot(
        np.vstack([query_state[None, :], true_future])[:, dims[0]],
        np.vstack([query_state[None, :], true_future])[:, dims[1]],
        color="tab:green",
        linewidth=2.0,
        linestyle="--",
        label="truth",
    )
    axis.scatter(query_state[dims[0]], query_state[dims[1]], color="tab:red", marker="x", s=75, label="query")
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title("Rollout against truth")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_support_diagnostics_along_rollout(
    rollout: RolloutPredictionResult,
) -> plt.Figure:
    """Plot rollout support diagnostics across horizons."""

    horizons = np.arange(1, len(rollout.step_diagnostics) + 1, dtype=np.int64)
    candidate_count = np.asarray([step.candidate_count for step in rollout.step_diagnostics], dtype=np.float64)
    ess = np.asarray([step.effective_sample_size for step in rollout.step_diagnostics], dtype=np.float64)
    mismatch = np.asarray([step.history_mismatch for step in rollout.step_diagnostics], dtype=np.float64)
    radius = np.asarray([step.search_radius for step in rollout.step_diagnostics], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.8), sharex=True)
    panels = [
        (axes[0, 0], candidate_count, "candidate count"),
        (axes[0, 1], ess, "effective sample size"),
        (axes[1, 0], mismatch, "history mismatch"),
        (axes[1, 1], radius, "search radius"),
    ]
    for axis, values, label in panels:
        axis.plot(horizons, values, marker="o", linewidth=1.8, color="#2b8cbe")
        axis.set_title(label)
        axis.grid(alpha=0.2)
        axis.set_xlabel("horizon")
    fig.suptitle("Support diagnostics along rollout", y=0.99)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_jitter_ellipse_or_covariance",
    "plot_local_increment_cloud",
    "plot_prediction_fan",
    "plot_rollout_against_truth",
    "plot_sampled_next_steps",
    "plot_support_diagnostics_along_rollout",
]
