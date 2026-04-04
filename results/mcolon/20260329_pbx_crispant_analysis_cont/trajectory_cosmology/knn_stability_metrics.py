"""
knn_stability_metrics.py
------------------------
Pure math helpers for neighborhood-stability analysis across saved iterations.
"""
from __future__ import annotations

import numpy as np


def finite_common_indices(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    mask: np.ndarray,
    time_index: int,
) -> np.ndarray:
    common = np.where(mask[:, time_index])[0]
    if len(common) == 0:
        return common
    keep = np.isfinite(positions_a[common, time_index]).all(axis=1)
    keep &= np.isfinite(positions_b[common, time_index]).all(axis=1)
    return common[keep]


def finite_points_at_time(
    positions: np.ndarray,
    mask: np.ndarray,
    time_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.where(mask[:, time_index])[0]
    if len(obs) == 0:
        return obs, np.zeros((0, positions.shape[-1]), dtype=float)
    pts = positions[obs, time_index, :]
    keep = np.isfinite(pts).all(axis=1)
    return obs[keep], pts[keep]


def knn_edge_index(points: np.ndarray, k: int) -> np.ndarray:
    n = len(points)
    if n <= 1:
        return np.zeros((0, 2), dtype=int)
    k_eff = max(1, min(k, n - 1))
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    order = np.argsort(dists, axis=1)[:, :k_eff]
    edges: set[tuple[int, int]] = set()
    for src, nbrs in enumerate(order):
        for dst in nbrs:
            a, b = sorted((int(src), int(dst)))
            if a != b:
                edges.add((a, b))
    if not edges:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(sorted(edges), dtype=int)


def _knn_sets(points: np.ndarray, k: int) -> list[set[int]]:
    edges = knn_edge_index(points, k)
    n = len(points)
    nbrs = [set() for _ in range(n)]
    for a, b in edges:
        nbrs[a].add(int(b))
        nbrs[b].add(int(a))
    return nbrs


def jaccard_similarity_matrix(
    positions_history: np.ndarray,
    mask: np.ndarray,
    ks: list[int],
) -> dict[int, np.ndarray]:
    n_snap = positions_history.shape[0]
    out: dict[int, np.ndarray] = {k: np.eye(n_snap, dtype=float) for k in ks}

    for i in range(n_snap):
        for j in range(i + 1, n_snap):
            sim_by_k = compare_iteration_pair(positions_history[i], positions_history[j], mask, ks)
            for k, score in sim_by_k.items():
                out[k][i, j] = score
                out[k][j, i] = score
    return out


def compare_iteration_pair(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    mask: np.ndarray,
    ks: list[int],
) -> dict[int, float]:
    scores: dict[int, list[float]] = {k: [] for k in ks}

    for t_idx in range(mask.shape[1]):
        common = finite_common_indices(positions_a, positions_b, mask, t_idx)
        if len(common) <= 1:
            continue
        pts_a = positions_a[common, t_idx, :]
        pts_b = positions_b[common, t_idx, :]
        for k in ks:
            if len(common) <= k:
                continue
            neigh_a = _knn_sets(pts_a, k)
            neigh_b = _knn_sets(pts_b, k)
            per_node = []
            for sa, sb in zip(neigh_a, neigh_b):
                denom = len(sa | sb)
                if denom == 0:
                    continue
                per_node.append(len(sa & sb) / denom)
            if per_node:
                scores[k].append(float(np.mean(per_node)))

    return {
        k: float(np.mean(vals)) if vals else float("nan")
        for k, vals in scores.items()
    }


def previous_iteration_similarity(
    similarity_matrix: np.ndarray,
    snapshot_iters: np.ndarray,
) -> np.ndarray:
    rows = []
    for idx in range(1, len(snapshot_iters)):
        rows.append([
            int(snapshot_iters[idx]),
            int(snapshot_iters[idx - 1]),
            float(similarity_matrix[idx - 1, idx]),
        ])
    return np.asarray(rows, dtype=float)
