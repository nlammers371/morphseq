"""Visualization helpers for matching diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, Mapping

from dev.particle_prediction.data.transition_bank import MatchResult, TransitionBank, TransitionWindow


def _coerce_path(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2:
        raise ValueError("trajectory-like inputs must be 2D arrays")
    if len(path) == 0:
        return None
    return path


def _set_local_limits(axis: plt.Axes, dims: tuple[int, int], point_sets: list[np.ndarray]) -> None:
    if not point_sets:
        return

    focus_points = np.vstack([points[:, list(dims)] for points in point_sets if len(points) > 0])
    mins = np.min(focus_points, axis=0)
    maxs = np.max(focus_points, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    padding = np.maximum(0.18 * span, 0.35)
    axis.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
    axis.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])


def _history_distance_sq_all_offsets(
    query_history_segments: np.ndarray,
    candidate_history_segments: np.ndarray,
    dims: tuple[int, ...] | None = None,
) -> float:
    query_history_segments = np.asarray(query_history_segments, dtype=np.float64)
    candidate_history_segments = np.asarray(candidate_history_segments, dtype=np.float64)

    if dims is not None:
        dim_index = list(dims)
        query_history_segments = query_history_segments[:, dim_index]
        candidate_history_segments = candidate_history_segments[:, dim_index]

    if query_history_segments.ndim != 2 or candidate_history_segments.ndim != 2:
        raise ValueError("history arrays must be 2D")
    if query_history_segments.shape[1] != candidate_history_segments.shape[1]:
        raise ValueError("query and candidate histories must share the same feature dimension")
    if candidate_history_segments.shape[0] < query_history_segments.shape[0]:
        raise ValueError("candidate history is shorter than query history")

    history_length = query_history_segments.shape[0]
    max_start = candidate_history_segments.shape[0] - history_length
    distances = [
        float(np.sum((query_history_segments - candidate_history_segments[start : start + history_length]) ** 2))
        for start in range(max_start + 1)
    ]
    return float(min(distances))


def _compute_reference_distance_values(
    bank: TransitionBank,
    query_state: np.ndarray,
    query_history_segments: np.ndarray | None,
    metric: Literal["full_context", "context_2d", "state_only"],
    dims: tuple[int, int],
) -> tuple[np.ndarray, str]:
    query_state = np.asarray(query_state, dtype=np.float64)

    if metric == "state_only":
        state_sq = np.sum((bank.state_matrix - query_state[None, :]) ** 2, axis=1)
        return np.sqrt(state_sq), "state-only distance"

    if query_history_segments is None:
        raise ValueError("query_history_segments is required for context-based metrics")

    query_history_segments = np.asarray(query_history_segments, dtype=np.float64)
    if metric == "full_context":
        state_sq = np.sum((bank.state_matrix - query_state[None, :]) ** 2, axis=1)
        history_sq = np.asarray(
            [
                _history_distance_sq_all_offsets(query_history_segments, candidate_history)
                for candidate_history in bank.history_tensor
            ],
            dtype=np.float64,
        )
        return np.sqrt(state_sq + history_sq), "full context distance"

    if metric == "context_2d":
        state_sq = np.sum((bank.state_matrix[:, list(dims)] - query_state[None, list(dims)]) ** 2, axis=1)
        history_sq = np.asarray(
            [
                _history_distance_sq_all_offsets(query_history_segments, candidate_history, dims=dims)
                for candidate_history in bank.history_tensor
            ],
            dtype=np.float64,
        )
        return np.sqrt(state_sq + history_sq), "2D context distance"

    raise ValueError("metric must be one of {'full_context', 'context_2d', 'state_only'}")


def plot_query_and_candidate_neighbors(
    query_state: np.ndarray,
    match_result: MatchResult,
    query_trajectory: np.ndarray | None = None,
    query_history_points: np.ndarray | None = None,
    reference_trajectories: Mapping[str, np.ndarray] | None = None,
    dims: tuple[int, int] = (0, 1),
    max_candidates: int = 20,
) -> plt.Figure:
    """Plot the query state and nearby bank candidates."""

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    query_state = np.asarray(query_state, dtype=np.float64)
    query_trajectory = _coerce_path(query_trajectory)
    query_history_points = _coerce_path(query_history_points)
    candidates = match_result.candidate_windows[:max_candidates]
    candidate_states = np.vstack([window.state for window in candidates])

    if reference_trajectories is not None:
        plotted_embryos: set[str] = set()
        for window in candidates:
            if window.embryo_id in plotted_embryos:
                continue
            trajectory = _coerce_path(reference_trajectories.get(window.embryo_id))
            if trajectory is None:
                continue
            axis.plot(
                trajectory[:, dims[0]],
                trajectory[:, dims[1]],
                color="0.55",
                linewidth=1.1,
                alpha=0.25,
                zorder=1,
            )
            plotted_embryos.add(window.embryo_id)

    if query_trajectory is not None:
        axis.plot(
            query_trajectory[:, dims[0]],
            query_trajectory[:, dims[1]],
            color="0.2",
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
            label="query trajectory",
        )

    if query_history_points is not None:
        history_mask = ~np.all(np.isclose(query_history_points, query_state[None, :]), axis=1)
        history_points = query_history_points[history_mask]
        if len(history_points) > 0:
            axis.scatter(
                history_points[:, dims[0]],
                history_points[:, dims[1]],
                s=26,
                facecolors="white",
                edgecolors="0.15",
                linewidths=0.9,
                alpha=0.95,
                zorder=4,
                label="query history",
            )

    scatter = axis.scatter(
        candidate_states[:, dims[0]],
        candidate_states[:, dims[1]],
        c=match_result.normalized_weights[: len(candidates)],
        cmap="viridis",
        s=45,
        alpha=0.9,
        zorder=3,
    )
    axis.scatter(query_state[dims[0]], query_state[dims[1]], color="tab:red", s=80, marker="x", zorder=5, label="query")
    focus_point_sets = [query_state[None, :], candidate_states]
    if query_history_points is not None:
        focus_point_sets.append(query_history_points)
    _set_local_limits(axis, dims=dims, point_sets=focus_point_sets)
    fig.colorbar(scatter, ax=axis, label="candidate weight")
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title("Query and candidate neighbors")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_reference_distance_landscape(
    query_state: np.ndarray,
    bank: TransitionBank,
    query_trajectory: np.ndarray | None = None,
    query_history_points: np.ndarray | None = None,
    query_history_segments: np.ndarray | None = None,
    reference_trajectories: Mapping[str, np.ndarray] | None = None,
    metric: Literal["full_context", "context_2d", "state_only"] = "full_context",
    dims: tuple[int, int] = (0, 1),
    focus_top_k: int = 64,
) -> plt.Figure:
    """Plot all valid bank states colored by a selectable distance-to-query metric."""

    fig, axis = plt.subplots(figsize=(6.8, 5.8))
    query_state = np.asarray(query_state, dtype=np.float64)
    query_trajectory = _coerce_path(query_trajectory)
    query_history_points = _coerce_path(query_history_points)
    distance_values, distance_label = _compute_reference_distance_values(
        bank=bank,
        query_state=query_state,
        query_history_segments=query_history_segments,
        metric=metric,
        dims=dims,
    )

    if reference_trajectories is not None:
        for embryo_id in sorted(reference_trajectories):
            trajectory = _coerce_path(reference_trajectories.get(embryo_id))
            if trajectory is None:
                continue
            axis.plot(
                trajectory[:, dims[0]],
                trajectory[:, dims[1]],
                color="0.55",
                linewidth=1.0,
                alpha=0.18,
                zorder=1,
            )

    if query_trajectory is not None:
        axis.plot(
            query_trajectory[:, dims[0]],
            query_trajectory[:, dims[1]],
            color="0.2",
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
            label="query trajectory",
        )

    if query_history_points is not None:
        history_mask = ~np.all(np.isclose(query_history_points, query_state[None, :]), axis=1)
        history_points = query_history_points[history_mask]
        if len(history_points) > 0:
            axis.scatter(
                history_points[:, dims[0]],
                history_points[:, dims[1]],
                s=26,
                facecolors="white",
                edgecolors="0.15",
                linewidths=0.9,
                alpha=0.95,
                zorder=4,
                label="query history",
            )

    scatter = axis.scatter(
        bank.state_matrix[:, dims[0]],
        bank.state_matrix[:, dims[1]],
        c=distance_values,
        cmap="viridis_r",
        s=26,
        alpha=0.9,
        linewidths=0.0,
        zorder=3,
    )
    axis.scatter(query_state[dims[0]], query_state[dims[1]], color="tab:red", s=80, marker="x", zorder=5, label="query")

    focus_count = max(1, min(int(focus_top_k), len(distance_values)))
    focus_indices = np.argsort(distance_values)[:focus_count]
    focus_point_sets = [query_state[None, :], bank.state_matrix[focus_indices]]
    if query_history_points is not None:
        focus_point_sets.append(query_history_points)
    _set_local_limits(axis, dims=dims, point_sets=focus_point_sets)

    fig.colorbar(scatter, ax=axis, label=distance_label)
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    scatter.set_clim(0, np.percentile(distance_values, 10))
    axis.set_title(f"All valid reference points: {metric}")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_history_reranking(
    position_only_match: MatchResult,
    history_match: MatchResult,
    max_candidates: int = 20,
) -> plt.Figure:
    """Compare candidate scores before and after history reranking."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [
        ("position only", position_only_match),
        ("history reranking", history_match),
    ]
    for axis, (title, match_result) in zip(axes, panels):
        candidate_labels = [window.embryo_id for window in match_result.candidate_windows[:max_candidates]]
        axis.barh(
            np.arange(len(candidate_labels)),
            match_result.normalized_weights[: len(candidate_labels)],
            color="#3a6ea5",
            alpha=0.9,
        )
        axis.set_yticks(np.arange(len(candidate_labels)))
        axis.set_yticklabels(candidate_labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("normalized weight")
        axis.grid(axis="x", alpha=0.2)

    axes[0].set_ylabel("candidate")
    fig.tight_layout()
    return fig


def plot_history_offset_heatmap(
    query_history_segments: np.ndarray,
    candidate_window: TransitionWindow,
    offset_radius: int,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Plot offset-tolerant history distances across the allowed offset band."""

    history = np.asarray(candidate_window.history_segments, dtype=np.float64)
    query_history_segments = np.asarray(query_history_segments, dtype=np.float64)
    history_length = query_history_segments.shape[0]
    center_start = (history.shape[0] - history_length) // 2

    offsets = []
    segment_errors = []
    distances = []
    for offset in range(-offset_radius, offset_radius + 1):
        start = center_start + offset
        stop = start + history_length
        if start < 0 or stop > history.shape[0]:
            continue
        candidate_slice = history[start:stop]
        offsets.append(offset)
        diff = query_history_segments - candidate_slice
        distances.append(float(np.sum(diff ** 2)))
        segment_errors.append(np.linalg.norm(diff[:, list(dims)], axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(offsets, distances, marker="o")
    axes[0].set_xlabel("offset")
    axes[0].set_ylabel("history distance")
    axes[0].set_title("Offset-tolerant history distance")
    axes[0].grid(alpha=0.2)

    heatmap = np.vstack(segment_errors)
    image = axes[1].imshow(heatmap, aspect="auto", cmap="magma", interpolation="nearest")
    axes[1].set_xlabel("history segment index")
    axes[1].set_ylabel("offset")
    axes[1].set_yticks(np.arange(len(offsets)))
    axes[1].set_yticklabels(offsets)
    axes[1].set_title("Per-segment mismatch")
    fig.colorbar(image, ax=axes[1], label="segment error")
    fig.tight_layout()
    return fig


def compare_default_vs_fast_matching(
    default_match: MatchResult,
    fast_match: MatchResult,
    max_candidates: int = 15,
) -> plt.Figure:
    """Compare candidate weights from default and fast matching modes."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [
        ("default", default_match),
        ("fast summary", fast_match),
    ]

    for axis, (title, match_result) in zip(axes, panels):
        labels = [window.embryo_id for window in match_result.candidate_windows[:max_candidates]]
        axis.barh(np.arange(len(labels)), match_result.normalized_weights[: len(labels)], color="#d97a3a", alpha=0.9)
        axis.set_title(title)
        axis.set_xlabel("normalized weight")
        axis.set_yticks(np.arange(len(labels)))
        axis.set_yticklabels(labels)
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()

    axes[0].set_ylabel("candidate")
    fig.tight_layout()
    return fig


__all__ = [
    "compare_default_vs_fast_matching",
    "plot_history_offset_heatmap",
    "plot_history_reranking",
    "plot_query_and_candidate_neighbors",
    "plot_reference_distance_landscape",
]
