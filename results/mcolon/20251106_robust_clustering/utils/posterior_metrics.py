"""Posterior and information-theoretic scores derived from aligned bootstrap histories."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from .bootstrap_alignment import stack_histories_to_matrix


def compute_membership_posteriors(
    aligned_histories: Iterable[np.ndarray],
    reference_labels: Sequence[int],
    dirichlet_alpha: float = 0.5,
) -> Dict[str, np.ndarray]:
    histories = [np.asarray(h, dtype=int) for h in aligned_histories]
    if not histories:
        raise ValueError("aligned_histories must contain at least one bootstrap run.")

    matrix_data = stack_histories_to_matrix(histories)

    return membership_posteriors_from_matrix(
        labels_matrix=matrix_data["labels"],
        sample_mask=matrix_data["sample_mask"],
        reference_labels=reference_labels,
        dirichlet_alpha=dirichlet_alpha,
    )


def membership_posteriors_from_matrix(
    labels_matrix: np.ndarray,
    sample_mask: np.ndarray,
    reference_labels: Sequence[int],
    dirichlet_alpha: float = 0.5,
) -> Dict[str, np.ndarray]:
    if labels_matrix.shape != sample_mask.shape:
        raise ValueError("labels_matrix and sample_mask must share the same shape.")

    cluster_ids = np.array(sorted(np.unique(reference_labels)))
    if cluster_ids.size == 0:
        raise ValueError("reference_labels must contain at least one cluster ID.")

    match = (
        (labels_matrix[:, :, None] == cluster_ids[None, None, :])
        & sample_mask[:, :, None]
    )
    counts = match.sum(axis=0).astype(float)
    frequency = sample_mask.sum(axis=0).astype(int)

    posterior = counts + dirichlet_alpha
    posterior /= posterior.sum(axis=1, keepdims=True)

    return {
        "counts": counts,
        "frequency": frequency,
        "posterior": posterior,
        "cluster_ids": cluster_ids,
    }


def max_assignment_probability(posterior: np.ndarray) -> np.ndarray:
    return posterior.max(axis=1)


def shannon_entropy(posterior: np.ndarray, base: float = np.e) -> np.ndarray:
    eps = np.finfo(float).eps
    log_post = np.log(posterior + eps) / np.log(base)
    return -(posterior * log_post).sum(axis=1)


def log_odds_gap(posterior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    sorted_probs = np.sort(posterior, axis=1)[:, ::-1]
    top1 = sorted_probs[:, 0] + eps
    top2 = sorted_probs[:, 1] + eps
    return np.log(top1 / top2)


def marginal_core_probability(posterior: np.ndarray, threshold: float) -> np.ndarray:
    return (posterior.max(axis=1) >= threshold).astype(float)
