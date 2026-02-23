# Consensus Trajectory Analysis Pipeline - Implementation Plan

## Quick Start for Next Agent

This plan implements a two-stage outlier filtering system with evidence accumulation consensus dendrograms.

### Key Files to Read First

```
src/analyze/trajectory_analysis/
├── dtw_distance.py          # compute_md_dtw_distance_matrix() - existing
├── bootstrap_clustering.py  # run_bootstrap_hierarchical() - existing, MODIFY
├── cluster_posteriors.py    # analyze_bootstrap_results() - existing
├── dendrogram.py            # generate_dendrograms() - existing, MODIFY
├── config.py                # parameters - MODIFY
├── __init__.py              # exports - MODIFY
├── distance_filtering.py    # NEW FILE
└── consensus_pipeline.py    # NEW FILE
```

### Test Data Location

Use b9d2 phenotype extraction notebook as reference:
- `results/mcolon/20251219_b9d2_phenotype_extraction/b8d2_phenotype_extraction.ipynb`

---

## Pipeline Flow

```
1. Compute MD-DTW distance matrix
2. STAGE 1: k-NN IQR filtering → Remove global outliers
3. Bootstrap clustering on filtered data
4. Build consensus dendrogram (evidence accumulation)
5. Posterior analysis (Core/Uncertain/Outlier classification)
6. STAGE 2: Within-cluster IQR + posterior filtering → Remove cluster outliers
7. Build final consensus dendrogram
8. Return all results + filtering log
```

---

## Files to Create

### 1. `distance_filtering.py`

**Path:** `src/analyze/trajectory_analysis/distance_filtering.py`

```python
"""
Two-stage outlier filtering for trajectory analysis.

Stage 1: k-NN IQR filtering (before clustering)
Stage 2: Within-cluster IQR + posterior filtering (after clustering)
"""
import numpy as np
from typing import Dict, List, Any, Tuple


def identify_embryo_outliers_iqr(
    D: np.ndarray,
    embryo_ids: List[str],
    iqr_multiplier: float = 4.0,
    k_neighbors: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Stage 1: Identify outliers using k-NN distances with IQR threshold.

    CRITICAL: Uses k-Nearest Neighbors (not global mean) to protect rare phenotypes.
    A small cluster of mutants won't be flagged as outliers.

    Outliers: knn_mean_distance > Q3 + iqr_multiplier × IQR

    Parameters
    ----------
    D : np.ndarray (n x n)
        Distance matrix
    embryo_ids : List[str]
        Embryo identifiers
    iqr_multiplier : float, default=4.0
        IQR multiplier threshold
    k_neighbors : int, default=5
        Number of nearest neighbors

    Returns
    -------
    dict with:
        outlier_indices, outlier_ids, kept_indices, kept_ids,
        knn_distances, threshold, q1, q3, iqr
    """
    n = len(D)
    k = min(k_neighbors, n - 1)

    # k-NN mean distance per embryo
    sorted_D = np.sort(D, axis=1)
    knn_distances = sorted_D[:, 1:k+1].mean(axis=1)  # Skip self (column 0)

    # IQR threshold
    q1 = np.percentile(knn_distances, 25)
    q3 = np.percentile(knn_distances, 75)
    iqr = q3 - q1
    threshold = q3 + iqr_multiplier * iqr

    # Identify outliers
    outlier_mask = knn_distances > threshold
    outlier_indices = np.where(outlier_mask)[0]
    kept_indices = np.where(~outlier_mask)[0]

    outlier_ids = [embryo_ids[i] for i in outlier_indices]
    kept_ids = [embryo_ids[i] for i in kept_indices]

    if verbose:
        print(f"Stage 1 IQR Filtering (k-NN distances, k={k}):")
        print(f"  Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
        print(f"  Threshold: {threshold:.2f} (Q3 + {iqr_multiplier}×IQR)")
        print(f"  Outliers removed: {len(outlier_ids)} / {n}")
        if len(outlier_ids) > 0:
            print(f"  Outlier IDs: {outlier_ids}")

    return {
        'outlier_indices': outlier_indices,
        'outlier_ids': outlier_ids,
        'kept_indices': kept_indices,
        'kept_ids': kept_ids,
        'knn_distances': knn_distances,
        'threshold': threshold,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }


def filter_data_and_ids(
    D: np.ndarray,
    embryo_ids: List[str],
    indices_to_keep: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    CRITICAL: Safely filter matrix AND IDs together to prevent index drift.

    The "single source of truth" for all filtering operations.
    Never manually slice D and embryo_ids separately.

    Parameters
    ----------
    D : np.ndarray (n x n)
        Distance matrix
    embryo_ids : List[str]
        Embryo IDs
    indices_to_keep : np.ndarray
        Indices to keep

    Returns
    -------
    D_filtered, ids_filtered
    """
    D_filtered = D[np.ix_(indices_to_keep, indices_to_keep)]
    ids_filtered = [embryo_ids[i] for i in indices_to_keep]

    assert len(ids_filtered) == len(D_filtered), \
        f"Index drift detected! IDs ({len(ids_filtered)}) != Matrix ({len(D_filtered)})"

    return D_filtered, ids_filtered


def identify_cluster_outliers_combined(
    D: np.ndarray,
    cluster_labels: np.ndarray,
    posterior_results: Dict[str, Any],
    embryo_ids: List[str],
    iqr_multiplier: float = 4.0,
    posterior_threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Stage 2: Remove embryo if EITHER:
    1. Within-cluster mean distance > Q3 + iqr_multiplier × IQR (per cluster)
    2. Posterior max_p < posterior_threshold

    Parameters
    ----------
    D : np.ndarray (n x n)
        Distance matrix (already filtered from Stage 1)
    cluster_labels : np.ndarray (n,)
        Cluster assignments
    posterior_results : dict
        Output from analyze_bootstrap_results()
    embryo_ids : List[str]
        Embryo IDs
    iqr_multiplier : float, default=4.0
        IQR multiplier
    posterior_threshold : float, default=0.5
        Minimum max_p to keep

    Returns
    -------
    dict with:
        outlier_indices, outlier_ids, kept_indices, kept_ids,
        outlier_reason (dict: ID -> 'iqr'/'posterior'/'both'),
        within_cluster_mean_distances
    """
    n = len(D)

    # Compute within-cluster mean distances per embryo
    within_cluster_mean = np.zeros(n)
    for i in range(n):
        cluster_i = cluster_labels[i]
        cluster_members = np.where(cluster_labels == cluster_i)[0]
        cluster_members = cluster_members[cluster_members != i]  # Exclude self
        if len(cluster_members) > 0:
            within_cluster_mean[i] = D[i, cluster_members].mean()
        else:
            within_cluster_mean[i] = 0.0

    # IQR threshold per cluster
    outlier_iqr = set()
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_dists = within_cluster_mean[cluster_mask]

        if len(cluster_dists) > 3:
            q1 = np.percentile(cluster_dists, 25)
            q3 = np.percentile(cluster_dists, 75)
            iqr = q3 - q1
            threshold = q3 + iqr_multiplier * iqr

            cluster_indices = np.where(cluster_mask)[0]
            for idx in cluster_indices:
                if within_cluster_mean[idx] > threshold:
                    outlier_iqr.add(idx)

    # Posterior threshold
    max_p_array = posterior_results['max_p']
    outlier_posterior = set(np.where(max_p_array < posterior_threshold)[0])

    # Combined: EITHER condition
    outlier_indices = np.array(sorted(outlier_iqr | outlier_posterior))
    kept_indices = np.array([i for i in range(n) if i not in outlier_indices])

    outlier_ids = [embryo_ids[i] for i in outlier_indices]
    kept_ids = [embryo_ids[i] for i in kept_indices]

    # Track reason
    outlier_reason = {}
    for idx in outlier_indices:
        in_iqr = idx in outlier_iqr
        in_post = idx in outlier_posterior
        if in_iqr and in_post:
            outlier_reason[embryo_ids[idx]] = 'both'
        elif in_iqr:
            outlier_reason[embryo_ids[idx]] = 'iqr'
        else:
            outlier_reason[embryo_ids[idx]] = 'posterior'

    if verbose:
        print(f"Stage 2 Combined Filtering (IQR + Posterior):")
        print(f"  IQR outliers: {len(outlier_iqr)}")
        print(f"  Posterior outliers (max_p < {posterior_threshold}): {len(outlier_posterior)}")
        print(f"  Total removed: {len(outlier_ids)} / {n}")

    return {
        'outlier_indices': outlier_indices,
        'outlier_ids': outlier_ids,
        'kept_indices': kept_indices,
        'kept_ids': kept_ids,
        'outlier_reason': outlier_reason,
        'within_cluster_mean_distances': within_cluster_mean
    }
```

---

### 2. Add to `bootstrap_clustering.py`

```python
def compute_coassociation_matrix(
    bootstrap_results_dict: Dict[str, Any],
    verbose: bool = True
) -> np.ndarray:
    """
    Evidence Accumulation: M[i,j] = fraction of times i and j co-clustered.
    Uses RAW bootstrap labels (no Hungarian alignment).

    Returns
    -------
    M : np.ndarray (n x n)
        Co-association matrix, M[i,j] ∈ [0, 1], symmetric, diagonal = 1
    """
    n_embryos = len(bootstrap_results_dict['embryo_ids'])
    bootstrap_results = bootstrap_results_dict['bootstrap_results']

    coassoc_count = np.zeros((n_embryos, n_embryos), dtype=int)
    cosample_count = np.zeros((n_embryos, n_embryos), dtype=int)

    for boot_result in bootstrap_results:
        labels = boot_result['labels']  # RAW labels (no alignment!)

        for i in range(n_embryos):
            for j in range(i, n_embryos):
                if labels[i] >= 0 and labels[j] >= 0:  # Both sampled?
                    cosample_count[i, j] += 1
                    cosample_count[j, i] += 1

                    if labels[i] == labels[j]:  # Same cluster?
                        coassoc_count[i, j] += 1
                        coassoc_count[j, i] += 1

    # Compute frequency
    M = np.zeros((n_embryos, n_embryos), dtype=float)
    for i in range(n_embryos):
        for j in range(n_embryos):
            if cosample_count[i, j] > 0:
                M[i, j] = coassoc_count[i, j] / cosample_count[i, j]
            else:
                M[i, j] = 0.5 if i != j else 1.0

    np.fill_diagonal(M, 1.0)

    if verbose:
        print(f"Co-association matrix computed (Evidence Accumulation):")
        print(f"  Mean: {M[np.triu_indices(n_embryos, k=1)].mean():.3f}")

    return M


def coassociation_to_distance(M: np.ndarray) -> np.ndarray:
    """D = 1 - M (consensus distance)"""
    D = 1.0 - M
    np.fill_diagonal(D, 0.0)
    return D
```

---

### 3. Modify `dendrogram.py`

Add parameter to `generate_dendrograms()`:

```python
def generate_dendrograms(
    D: np.ndarray,
    embryo_ids: List[str],
    *,
    coassociation_matrix: Optional[np.ndarray] = None,  # NEW
    # ... existing parameters ...
):
    # At start of function, add:
    if coassociation_matrix is not None:
        D_for_linkage = coassociation_to_distance(coassociation_matrix)
    else:
        D_for_linkage = D
    # Use D_for_linkage for linkage computation
```

---

## Key Design Decisions

### 1. k-NN Filtering (not global mean)
- **Problem**: Global mean would flag rare mutant clusters as outliers
- **Solution**: k-NN distance - embryo is outlier only if it has no nearby friends
- **Threshold**: Q3 + 4.0×IQR (conservative)

### 2. Index Tracking (CRITICAL)
- `filter_data_and_ids()` always filters matrix AND IDs together
- Prevents index drift when removing embryos

### 3. Evidence Accumulation (no Hungarian alignment)
- Raw bootstrap labels - simpler and alignment-agnostic
- Merge height = 1 - co-clustering frequency
- Height 0.0 = always together, Height 0.8 = rarely together

### 4. Filtering Log (Chain of Custody)
- DataFrame tracking every embryo: Kept/Stage1_Reject/Stage2_Reject
- Purpose: Detect bias in filtering

---

## Test Plan

1. **Create test notebook** in this directory
2. Load b9d2 data (or similar)
3. Test each function individually:
   - `identify_embryo_outliers_iqr()`
   - `filter_data_and_ids()`
   - `compute_coassociation_matrix()`
   - `identify_cluster_outliers_combined()`
4. Test full pipeline flow
5. Validate consensus heatmap visualization

---

## Configuration

```python
# In config.py
ENABLE_IQR_FILTERING = True
IQR_MULTIPLIER = 4.0
KNN_K = 5
POSTERIOR_OUTLIER_THRESHOLD = 0.5
RANDOM_SEED = 42
```
