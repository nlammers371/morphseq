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
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import nan_euclidean_distances


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
    if _has_missing_features(features, mask):
        return nan_aware_aligned_umap_init(
            features,
            mask,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for aligned_umap_init")

    N_e, T, _ = features.shape
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


def nan_aware_aligned_umap_init(
    features: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    min_shared_features: int = 1,
) -> np.ndarray:
    """Fit a NaN-aware per-bin UMAP init and align consecutive bins with Procrustes.

    This path is intended for sparse pairwise coordinates where missing values mean
    "off-support", not "neutral".
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for nan_aware_aligned_umap_init")

    N_e, T, _ = features.shape
    slice_indices = _slice_embryo_indices(mask)
    x0 = np.full((N_e, T, 2), np.nan)
    prev_embedding: np.ndarray | None = None
    prev_index: np.ndarray | None = None

    for t, idx in enumerate(slice_indices):
        if len(idx) == 0:
            continue

        emb = _nan_aware_umap_slice(
            features[idx, t, :],
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state + t,
            min_shared_features=min_shared_features,
        )
        if prev_embedding is not None and prev_index is not None:
            emb = _align_with_procrustes(
                current=emb,
                current_index=idx,
                reference=prev_embedding,
                reference_index=prev_index,
            )

        x0[idx, t, :] = emb
        prev_embedding = emb
        prev_index = idx

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

    N_e, T, _ = features.shape
    x0 = np.full((N_e, T, 2), np.nan)

    all_rows = features[mask]  # (n_obs, K)
    pca = PCA(n_components=2, random_state=random_state).fit(all_rows)

    for t in range(T):
        observed = mask[:, t]
        if observed.sum() == 0:
            continue
        x0[observed, t, :] = pca.transform(features[observed, t, :])

    return x0


def _has_missing_features(features: np.ndarray, mask: np.ndarray) -> bool:
    observed = features[mask]
    return bool(observed.size and np.isnan(observed).any())


def _nan_aware_umap_slice(
    X: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    min_shared_features: int,
) -> np.ndarray:
    n_obs = X.shape[0]
    if n_obs == 1:
        return np.zeros((1, 2), dtype=float)
    if n_obs == 2:
        dist = _nan_aware_distance_matrix(X, min_shared_features=min_shared_features)
        span = float(dist[0, 1]) if np.isfinite(dist[0, 1]) and dist[0, 1] > 0 else 1.0
        return np.array([[-0.5 * span, 0.0], [0.5 * span, 0.0]], dtype=float)

    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for _nan_aware_umap_slice")

    dist = _nan_aware_distance_matrix(X, min_shared_features=min_shared_features)
    neighbors = max(2, min(n_neighbors, n_obs - 1))
    return umap.UMAP(
        metric="precomputed",
        n_neighbors=neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    ).fit_transform(dist)


def _nan_aware_distance_matrix(X: np.ndarray, *, min_shared_features: int) -> np.ndarray:
    dist = nan_euclidean_distances(X)
    overlap = _shared_feature_counts(X)
    valid = overlap >= int(min_shared_features)
    valid[np.diag_indices_from(valid)] = True

    finite = np.isfinite(dist) & valid
    if finite.any():
        fill_value = float(np.nanmax(dist[finite]))
        if not np.isfinite(fill_value) or fill_value <= 0:
            fill_value = 1.0
    else:
        fill_value = 1.0

    out = np.array(dist, copy=True)
    out[~finite] = fill_value * 1.25
    np.fill_diagonal(out, 0.0)
    return out


def _shared_feature_counts(X: np.ndarray) -> np.ndarray:
    valid = np.isfinite(X)
    return valid @ valid.T


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


def _align_with_procrustes(
    *,
    current: np.ndarray,
    current_index: np.ndarray,
    reference: np.ndarray,
    reference_index: np.ndarray,
) -> np.ndarray:
    ref_pos = {int(e): i for i, e in enumerate(reference_index)}
    shared_pairs = [(i_cur, ref_pos[int(e)]) for i_cur, e in enumerate(current_index) if int(e) in ref_pos]
    if len(shared_pairs) < 2:
        return current

    cur_idx = np.array([pair[0] for pair in shared_pairs], dtype=int)
    ref_idx = np.array([pair[1] for pair in shared_pairs], dtype=int)
    cur_anchor = current[cur_idx]
    ref_anchor = reference[ref_idx]

    cur_mean = cur_anchor.mean(axis=0)
    ref_mean = ref_anchor.mean(axis=0)
    cur_centered = cur_anchor - cur_mean
    ref_centered = ref_anchor - ref_mean
    if np.allclose(cur_centered, 0.0) or np.allclose(ref_centered, 0.0):
        return current - cur_mean + ref_mean

    rotation, _ = orthogonal_procrustes(cur_centered, ref_centered)
    return (current - cur_mean) @ rotation + ref_mean


def _slice_embryo_indices(mask: np.ndarray) -> list[np.ndarray]:
    T = mask.shape[1]
    return [np.where(mask[:, t])[0] for t in range(T)]
