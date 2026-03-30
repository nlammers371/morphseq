"""
init_embedding.py
-----------------
Produces initial 2D positions x0 from canonical feature tensors.

Condensation never imports from here — it only sees x0 and mask.

Functions take explicit (features, mask) arguments rather than the full
CosmologyData, keeping the dependency surface minimal and the boundary sharp.
"""
from __future__ import annotations

import numpy as np


def aligned_umap_init(
    features: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    alignment_regularisation: float = 1e-2,
    alignment_window_size: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Fit AlignedUMAP on the feature tensor and return initial positions.

    Parameters
    ----------
    features : (N_e, T, K)
    mask : (N_e, T) bool — True where embryo is observed

    Returns
    -------
    x0 : (N_e, T, 2) float array — NaN where mask is False
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for aligned_umap_init")

    N_e, T, K = features.shape
    slice_indices = _slice_embryo_indices(mask)
    slices, relations = _build_aligned_umap_inputs(features, mask, slice_indices)

    model = umap.AlignedUMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        n_components=2,
        random_state=random_state,
    ).fit(slices, relations=relations)

    x0 = np.full((N_e, T, 2), np.nan)
    for t, (emb, idx) in enumerate(zip(model.embeddings_, slice_indices)):
        x0[idx, t, :] = emb

    return x0


def pca_init(
    features: np.ndarray,
    mask: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Simple PCA fallback initialization — no temporal alignment.

    Useful for smoke tests when umap-learn is unavailable or slow.

    Parameters
    ----------
    features : (N_e, T, K)
    mask : (N_e, T) bool

    Returns
    -------
    x0 : (N_e, T, 2) float array
    """
    from sklearn.decomposition import PCA

    N_e, T, K = features.shape
    x0 = np.full((N_e, T, 2), np.nan)

    all_rows = features[mask]  # (n_obs, K)
    pca = PCA(n_components=2, random_state=random_state).fit(all_rows)

    for t in range(T):
        observed = mask[:, t]
        if observed.sum() == 0:
            continue
        x0[observed, t, :] = pca.transform(features[observed, t, :])

    return x0


def _build_aligned_umap_inputs(
    features: np.ndarray,
    mask: np.ndarray,
    slice_indices: list[np.ndarray],
) -> tuple[list[np.ndarray], list[dict[int, int]]]:
    """Build per-time slices and consecutive-time embryo relations for AlignedUMAP."""
    T = features.shape[1]
    slices = [features[idx, t, :] for t, idx in enumerate(slice_indices)]
    relations = []

    for t in range(T - 1):
        idx_t = slice_indices[t]
        idx_t1 = slice_indices[t + 1]
        shared = set(idx_t) & set(idx_t1)
        pos_in_t = {e: i for i, e in enumerate(idx_t)}
        pos_in_t1 = {e: i for i, e in enumerate(idx_t1)}
        relations.append({pos_in_t[e]: pos_in_t1[e] for e in shared})

    return slices, relations


def _slice_embryo_indices(mask: np.ndarray) -> list[np.ndarray]:
    T = mask.shape[1]
    return [np.where(mask[:, t])[0] for t in range(T)]
