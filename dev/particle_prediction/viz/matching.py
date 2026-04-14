"""Visualization helpers for matching diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dev.particle_prediction.data.transition_bank import MatchResult, TransitionWindow


def plot_query_and_candidate_neighbors(
    query_state: np.ndarray,
    match_result: MatchResult,
    dims: tuple[int, int] = (0, 1),
    max_candidates: int = 20,
) -> plt.Figure:
    """Plot the query state and nearby bank candidates."""

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    candidates = match_result.candidate_windows[:max_candidates]
    candidate_states = np.vstack([window.state for window in candidates])
    scatter = axis.scatter(
        candidate_states[:, dims[0]],
        candidate_states[:, dims[1]],
        c=match_result.normalized_weights[: len(candidates)],
        cmap="viridis",
        s=45,
        alpha=0.9,
    )
    axis.scatter(query_state[dims[0]], query_state[dims[1]], color="tab:red", s=80, marker="x", label="query")
    fig.colorbar(scatter, ax=axis, label="candidate weight")
    axis.set_xlabel(f"latent dim {dims[0]}")
    axis.set_ylabel(f"latent dim {dims[1]}")
    axis.set_title("Query and candidate neighbors")
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
]
