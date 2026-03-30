"""
init_embedding.py
-----------------
Produces initial 2D positions x0 from canonical feature tensors.

Condensation never imports from here — it only sees x0 and mask.
"""
from __future__ import annotations

import numpy as np

from .schema import CosmologyData


def aligned_umap_init(
    data: CosmologyData,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    alignment_regularisation: float = 1e-2,
    alignment_window_size: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Fit AlignedUMAP on the feature tensor and return initial positions.

    Parameters
    ----------
    data
        Canonical CosmologyData from schema.py.

    Returns
    -------
    x0 : (N_e, T, 2) float array
        Initial 2D positions. NaN where mask is False.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for aligned_umap_init")

    N_e, T, K = data.features.shape
    slices, relations = _build_aligned_umap_inputs(data)

    model = umap.AlignedUMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        n_components=2,
        random_state=random_state,
    ).fit(slices, relations=relations)

    x0 = np.full((N_e, T, 2), np.nan)
    slice_embryo_indices = _slice_embryo_indices(data)

    for t, (emb, idx) in enumerate(zip(model.embeddings_, slice_embryo_indices)):
        x0[idx, t, :] = emb

    return x0


def pca_init(data: CosmologyData, random_state: int = 42) -> np.ndarray:
    """Simple PCA fallback initialization — no temporal alignment.

    Useful for smoke tests when umap-learn is unavailable or slow.

    Returns
    -------
    x0 : (N_e, T, 2) float array
    """
    from sklearn.decomposition import PCA

    N_e, T, K = data.features.shape
    x0 = np.full((N_e, T, 2), np.nan)

    all_rows = data.features[data.mask]  # (n_obs, K)
    pca = PCA(n_components=2, random_state=random_state).fit(all_rows)

    for t in range(T):
        observed = data.mask[:, t]
        if observed.sum() == 0:
            continue
        x0[observed, t, :] = pca.transform(data.features[observed, t, :])

    return x0


def _build_aligned_umap_inputs(
    data: CosmologyData,
) -> tuple[list[np.ndarray], list[dict[int, int]]]:
    """Build per-time slices and consecutive-time embryo relations for AlignedUMAP."""
    N_e, T, K = data.features.shape
    slices = []
    relations = []
    slice_embryo_indices = _slice_embryo_indices(data)

    for t in range(T):
        idx = slice_embryo_indices[t]
        slices.append(data.features[idx, t, :])

    for t in range(T - 1):
        idx_t = slice_embryo_indices[t]
        idx_t1 = slice_embryo_indices[t + 1]
        set_t = set(idx_t)
        set_t1 = set(idx_t1)
        shared = set_t & set_t1
        pos_in_t = {e: i for i, e in enumerate(idx_t)}
        pos_in_t1 = {e: i for i, e in enumerate(idx_t1)}
        rel = {pos_in_t[e]: pos_in_t1[e] for e in shared}
        relations.append(rel)

    return slices, relations


def _slice_embryo_indices(data: CosmologyData) -> list[np.ndarray]:
    T = data.features.shape[1]
    return [np.where(data.mask[:, t])[0] for t in range(T)]
