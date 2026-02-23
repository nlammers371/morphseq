"""Graph-based scores derived from the co-association matrix."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _prep_matrix(C: np.ndarray, min_weight: float) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    if C.shape[0] != C.shape[1]:
        raise ValueError("Co-association matrix must be square.")

    C = (C + C.T) / 2.0
    C[C < min_weight] = 0.0
    np.fill_diagonal(C, 0.0)
    return C


def within_module_degree_z(C: np.ndarray, labels: Sequence[int], min_weight: float = 0.0) -> np.ndarray:
    C = _prep_matrix(C, min_weight)
    labels = np.asarray(labels)
    z_scores = np.zeros_like(labels, dtype=float)

    for cluster in np.unique(labels):
        mask = labels == cluster
        idx = np.where(mask)[0]
        if idx.size <= 1:
            continue
        intra_strength = C[np.ix_(idx, idx)].sum(axis=1)
        mean = intra_strength.mean()
        std = intra_strength.std(ddof=1)
        if std == 0:
            continue
        z_scores[idx] = (intra_strength - mean) / std

    return z_scores


def participation_coefficient(
    C: np.ndarray,
    labels: Sequence[int],
    min_weight: float = 0.0,
) -> np.ndarray:
    C = _prep_matrix(C, min_weight)
    labels = np.asarray(labels)
    participation = np.zeros_like(labels, dtype=float)

    total_strength = C.sum(axis=1)
    zero_strength = total_strength <= 0
    total_strength[zero_strength] = 1.0

    for cluster in np.unique(labels):
        mask = labels == cluster
        cluster_strength = C[:, mask].sum(axis=1)
        participation += (cluster_strength / total_strength) ** 2

    participation = 1.0 - participation
    participation[zero_strength] = 0.0
    return participation


def compute_graph_scores(
    C: np.ndarray,
    labels: Sequence[int],
    min_weight: float = 0.0,
) -> Dict[str, np.ndarray]:
    return {
        "within_module_z": within_module_degree_z(C, labels, min_weight=min_weight),
        "participation_coef": participation_coefficient(C, labels, min_weight=min_weight),
    }
