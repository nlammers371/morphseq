"""
MD-DTW Prototype Functions for b9d2 Phenotype Analysis

This module implements multivariate Dynamic Time Warping (MD-DTW) for distinguishing
HTA (Head-Trunk Angle) vs CE (Convergent Extension) phenotypes in b9d2 mutants.

Core Implementation:
- Pure Python/NumPy/SciPy - no tslearn dependency required
- Handles multivariate time series with Euclidean distance in feature space
- Sakoe-Chiba band constraint for efficient computation

Functions:
- prepare_multivariate_array(): Convert long-format DataFrame to 3D array
- compute_md_dtw_distance_matrix(): Compute multivariate DTW distances
- plot_dendrogram(): Visualize hierarchical clustering for K-selection
- _dtw_multivariate_pair(): Helper for pairwise DTW computation

Created: 2025-12-18
Location: results/mcolon/20251218_MD-DTW-morphseq_analysis/
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import existing trajectory utilities
from src.analyze.trajectory_analysis.trajectory_utils import interpolate_to_common_grid_multi_df, GRID_STEP
from src.analyze.trajectory_analysis.genotype_styling import get_color_for_genotype

# Pastel color palette for pairs (softer, distinguishable colors)
PASTEL_COLORS = [
    '#FFB6C1',  # Light pink
    '#B0E0E6',  # Powder blue
    '#98FB98',  # Pale green
    '#FFE4B5',  # Moccasin
    '#DDA0DD',  # Plum
    '#F0E68C',  # Khaki
    '#AFEEEE',  # Pale turquoise
    '#FFA07A',  # Light salmon
    '#D8BFD8',  # Thistle
    '#F5DEB3',  # Wheat
    '#B0C4DE',  # Light steel blue
    '#FFDAB9',  # Peach puff
    '#E0BBE4',  # Mauve
    '#C1FFC1',  # Pale spring green
    '#FFE4E1',  # Misty rose
]


def prepare_multivariate_array(
    df: pd.DataFrame,
    metrics: List[str],
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    time_grid: Optional[np.ndarray] = None,
    normalize: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Convert long-format DataFrame to 3D array for multivariate DTW.

    Takes a DataFrame with multiple metric columns and converts to a 3D numpy array
    suitable for multivariate DTW computation. Handles interpolation to common time
    grid and optional Z-score normalization.

    Args:
        df: Long-format DataFrame with columns [embryo_id, time_col, metric1, metric2, ...]
        metrics: List of metric column names (e.g., ['baseline_deviation_normalized', 'total_length_um'])
        time_col: Name of time column (default: 'predicted_stage_hpf')
        embryo_id_col: Name of embryo ID column (default: 'embryo_id')
        time_grid: Optional pre-defined time grid. If None, auto-computed from data
        normalize: Whether to Z-score normalize each metric globally (default: True)
        verbose: Print progress information (default: True)

    Returns:
        X: np.ndarray with shape (n_embryos, n_timepoints, n_metrics)
        embryo_ids: List[str] with embryo identifiers (same order as X rows)
        time_grid: np.ndarray with time values (same for all embryos)

    Example:
        >>> df = load_experiment_dataframe('20251121')
        >>> X, embryo_ids, time_grid = prepare_multivariate_array(
        ...     df,
        ...     metrics=['baseline_deviation_normalized', 'total_length_um']
        ... )
        >>> print(X.shape)  # (n_embryos, n_timepoints, 2)

    Notes:
        - All embryos are interpolated to the same common time grid
        - Missing values (NaN) are handled by linear interpolation
        - Z-score normalization ensures equal weight for all metrics in DTW
    """
    if verbose:
        print(f"Preparing multivariate array for {len(metrics)} metrics...")
        print(f"  Metrics: {metrics}")
        print(f"  Normalization: {normalize}")

    # Step 1: Get embryo IDs in sorted order (for consistency)
    embryo_ids = sorted(df[embryo_id_col].unique())
    n_embryos = len(embryo_ids)
    n_metrics = len(metrics)

    if verbose:
        print(f"  Embryos: {n_embryos}")

    # Step 2: Interpolate each metric for all embryos using the trajectory utility
    # If time_grid is provided, pass it through; otherwise let the utility derive the grid
    df_interp = interpolate_to_common_grid_multi_df(
        df,
        metrics,
        grid_step=(time_grid[1] - time_grid[0]) if time_grid is not None and len(time_grid) > 1 else GRID_STEP,
        time_col=time_col,
        time_grid=time_grid,
        fill_edges=False,
        verbose=verbose,
    )

    # Extract actual grid from interpolation results (safe in case bounds differ)
    time_grid = np.sort(df_interp[time_col].unique())
    n_timepoints = len(time_grid)

    if verbose:
        print(f"  Time points: {n_timepoints} ({time_grid.min():.1f} - {time_grid.max():.1f} hpf)")

    # Step 3: Initialize 3D array
    X = np.zeros((n_embryos, n_timepoints, n_metrics))

    for i, embryo_id in enumerate(embryo_ids):
        emb_df = df_interp[df_interp[embryo_id_col] == embryo_id].set_index(time_col)

        if emb_df.empty:
            if verbose:
                print(f"  WARNING: Embryo {embryo_id} has no interpolated rows, using zeros")
            continue

        for j, metric in enumerate(metrics):
            # Reindex onto full grid and fill missing values
            ser = emb_df[metric].reindex(time_grid)
            # Fill any remaining NaNs using interpolation then zeros
            ser = ser.interpolate(limit_direction='both').fillna(0)
            X[i, :, j] = ser.values

    # Step 5: Handle remaining NaNs (e.g., at edges due to interpolation bounds)
    mask = np.isnan(X)
    if mask.any():
        for i in range(n_embryos):
            for j in range(n_metrics):
                series = X[i, :, j]
                nans = np.isnan(series)

                if nans.all():
                    # All NaNs - set to 0
                    X[i, :, j] = 0
                else:
                    # Use pandas to fill NaNs with interpolation
                    filled = pd.Series(series).interpolate(limit_direction='both').fillna(0)
                    X[i, :, j] = filled.values

    if verbose:
        print(f"  Array shape: {X.shape}")
        print(f"  Before normalization:")
        for j, metric in enumerate(metrics):
            print(f"    {metric}: mean={X[:, :, j].mean():.3f}, std={X[:, :, j].std():.3f}")

    # Step 6: Global Z-score normalization (if enabled)
    if normalize:
        original_shape = X.shape
        X_reshaped = X.reshape(-1, n_metrics)

        # Z-score each metric globally
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_reshaped)

        X = X_normalized.reshape(original_shape)

        if verbose:
            print(f"  After normalization:")
            for j, metric in enumerate(metrics):
                print(f"    {metric}: mean={X[:, :, j].mean():.6f}, std={X[:, :, j].std():.6f}")

    if verbose:
        print(f"✓ Multivariate array prepared successfully")

    return X, embryo_ids, time_grid


def _dtw_multivariate_pair(
    ts_a: np.ndarray,
    ts_b: np.ndarray,
    window: Optional[int] = 3,
) -> float:
    """
    Compute DTW distance between two multivariate time series.

    Uses Euclidean distance in feature space as the local distance metric.
    Implements dynamic programming with optional Sakoe-Chiba band constraint.

    Args:
        ts_a: 2D array with shape (T_a, n_features) - time series A
        ts_b: 2D array with shape (T_b, n_features) - time series B
        window: Sakoe-Chiba band width (None for unconstrained DTW)

    Returns:
        float: DTW distance between ts_a and ts_b

    Notes:
        - The "multivariate" part is handled by computing Euclidean distance
          between feature vectors at each timepoint pair
        - ts_a[i] and ts_b[j] are vectors in feature space
        - Distance is computed as sqrt(sum((ts_a[i] - ts_b[j])^2))
    """
    # Step 1: Compute local cost matrix
    # dist_matrix[i, j] = Euclidean distance between ts_a[i] and ts_b[j]
    # This naturally handles multivariate data
    dist_matrix = cdist(ts_a, ts_b, metric='euclidean')

    n, m = dist_matrix.shape

    # Step 2: Initialize accumulated cost matrix
    # Set all to infinity initially
    acc_cost = np.full((n + 1, m + 1), np.inf)
    acc_cost[0, 0] = 0

    # Step 3: Dynamic Programming with Sakoe-Chiba Constraint
    # The window parameter limits how far we can warp in time
    if window is None:
        # Unconstrained DTW - compute all pairs
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i - 1, j - 1]
                acc_cost[i, j] = cost + min(
                    acc_cost[i - 1, j],      # Insertion (skip in ts_a)
                    acc_cost[i, j - 1],      # Deletion (skip in ts_b)
                    acc_cost[i - 1, j - 1]   # Match
                )
    else:
        # Constrained DTW - only compute within band
        # The band width is determined by the window parameter
        w = max(window, abs(n - m))

        for i in range(1, n + 1):
            # Determine valid range for j
            start_j = max(1, i - w)
            end_j = min(m + 1, i + w + 1)

            for j in range(start_j, end_j):
                cost = dist_matrix[i - 1, j - 1]
                acc_cost[i, j] = cost + min(
                    acc_cost[i - 1, j],
                    acc_cost[i, j - 1],
                    acc_cost[i - 1, j - 1]
                )

    return acc_cost[n, m]


def compute_md_dtw_distance_matrix(
    X: np.ndarray,
    sakoe_chiba_radius: Optional[int] = 3,
    n_jobs: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute multivariate DTW distance matrix.

    Computes pairwise multivariate DTW distances between all embryos using
    pure Python/NumPy implementation (no tslearn required).

    Args:
        X: 3D array with shape (n_embryos, n_timepoints, n_metrics)
        sakoe_chiba_radius: Sakoe-Chiba band constraint width.
                           Limits warping to within ±radius timepoints.
                           None for unconstrained DTW (slower).
                           Default: 3 (good balance of flexibility and speed)
        n_jobs: Kept for API compatibility (single-threaded implementation)
        verbose: Print progress and diagnostics (default: True)

    Returns:
        distance_matrix: np.ndarray with shape (n_embryos, n_embryos)
                        Symmetric matrix where D[i,j] = DTW distance between embryo i and j

    Example:
        >>> X, embryo_ids, time_grid = prepare_multivariate_array(df, metrics=['curvature', 'length'])
        >>> D = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3)
        >>> print(D.shape)  # (n_embryos, n_embryos)
        >>> # Use D for hierarchical clustering
        >>> from src.analyze.trajectory_analysis import run_bootstrap_hierarchical
        >>> results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids)

    Notes:
        - This is a pure Python implementation using NumPy/SciPy
        - No external dependencies like tslearn required
        - Time complexity: O(N^2 * T^2) where N=embryos, T=timepoints
        - For N~50, T~50: ~25M operations (seconds to complete)
        - Output is symmetric by construction: D[i,j] == D[j,i]
        - Diagonal is zero: D[i,i] == 0
    """
    n_embryos = X.shape[0]

    if verbose:
        print(f"Computing MD-DTW distance matrix (Pure Python/NumPy)...")
        print(f"  Embryos: {n_embryos}")
        print(f"  Array shape: {X.shape}")
        print(f"  Sakoe-Chiba radius: {sakoe_chiba_radius}")

    # Initialize distance matrix
    D = np.zeros((n_embryos, n_embryos))

    # Total pairs to compute
    total_pairs = (n_embryos * (n_embryos + 1)) // 2
    pair_count = 0

    # Compute pairwise distances (only upper triangle due to symmetry)
    for i in range(n_embryos):
        for j in range(i, n_embryos):
            if i == j:
                # Diagonal is always 0 (distance from embryo to itself)
                dist = 0.0
            else:
                # Compute DTW distance between embryos i and j
                dist = _dtw_multivariate_pair(X[i], X[j], window=sakoe_chiba_radius)

            D[i, j] = dist
            D[j, i] = dist  # Symmetric

            pair_count += 1
            if verbose and pair_count % max(1, total_pairs // 10) == 0:
                print(f"  Processed {pair_count}/{total_pairs} pairs ({100*pair_count//total_pairs}%)...", end='\r')

    if verbose:
        # Verify properties
        diagonal_max = np.max(np.abs(np.diag(D)))
        asymmetry = np.max(np.abs(D - D.T))

        print(f"\n✓ Distance matrix computed")
        print(f"  Shape: {D.shape}")
        print(f"  Distance range: [{D[D > 0].min():.4f}, {D.max():.4f}]")
        print(f"  Max diagonal value: {diagonal_max:.2e} (should be ~0)")
        print(f"  Max asymmetry: {asymmetry:.2e} (should be ~0)")

        if diagonal_max > 1e-10:
            print(f"  WARNING: Diagonal not zero (max={diagonal_max:.2e})")
        if asymmetry > 1e-10:
            print(f"  WARNING: Matrix not symmetric (max asymmetry={asymmetry:.2e})")

    return D


def identify_outliers(
    D: np.ndarray,
    embryo_ids: List[str],
    method: str = 'median_distance',
    threshold: Optional[float] = None,
    percentile: float = 95,
    verbose: bool = True,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Identify outlier embryos based on distance matrix.

    Outliers are embryos that are consistently far from all other embryos,
    which often create singleton clusters that inflate k in hierarchical clustering.

    Args:
        D: Distance matrix (n_embryos, n_embryos)
        embryo_ids: List of embryo identifiers (same order as D rows)
        method: Outlier detection method:
            - 'median_distance': Flag embryos with median distance > threshold
            - 'percentile': Flag embryos with median distance > percentile of all medians
            - 'mad': Median Absolute Deviation (robust to outliers)
        threshold: Manual threshold for median distance (used with 'median_distance' method)
        percentile: Percentile cutoff for 'percentile' method (default: 95)
        verbose: Print diagnostic information

    Returns:
        outlier_ids: List of outlier embryo IDs
        inlier_ids: List of non-outlier embryo IDs
        info: Dict with diagnostic information:
            - 'median_distances': Median distance for each embryo
            - 'threshold': Threshold used for outlier detection
            - 'outlier_indices': Indices of outliers in original array
            - 'inlier_indices': Indices of inliers in original array

    Example:
        >>> # Detect outliers using 95th percentile
        >>> outliers, inliers, info = identify_outliers(
        ...     D, embryo_ids, method='percentile', percentile=95
        ... )
        >>> print(f"Found {len(outliers)} outliers: {outliers}")

        >>> # Remove outliers and re-cluster
        >>> D_clean = D[np.ix_(info['inlier_indices'], info['inlier_indices'])]
        >>> embryo_ids_clean = inliers

    Notes:
        - median_distance: Good when you know approximate scale of your data
        - percentile: Adaptive to your data distribution (recommended)
        - mad: Most robust to extreme outliers, but can be conservative
    """
    n = len(embryo_ids)

    if verbose:
        print(f"\nIdentifying outliers using '{method}' method...")
        print(f"  Total embryos: {n}")

    # Compute median distance for each embryo (to all others)
    # Exclude diagonal (distance to self = 0)
    median_distances = np.zeros(n)
    for i in range(n):
        # Get distances to all other embryos (exclude self)
        dists_to_others = np.concatenate([D[i, :i], D[i, i+1:]])
        median_distances[i] = np.median(dists_to_others)

    # Determine threshold based on method
    if method == 'median_distance':
        if threshold is None:
            raise ValueError("threshold must be provided for 'median_distance' method")
        thresh = threshold

    elif method == 'percentile':
        thresh = np.percentile(median_distances, percentile)
        if verbose:
            print(f"  {percentile}th percentile of median distances: {thresh:.3f}")

    elif method == 'mad':
        # Median Absolute Deviation (MAD)
        median_of_medians = np.median(median_distances)
        mad = np.median(np.abs(median_distances - median_of_medians))
        # Use 3 * MAD as threshold (robust outlier detection)
        thresh = median_of_medians + 3 * mad
        if verbose:
            print(f"  Median of median distances: {median_of_medians:.3f}")
            print(f"  MAD: {mad:.3f}")
            print(f"  Threshold (median + 3*MAD): {thresh:.3f}")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'median_distance', 'percentile', or 'mad'")

    # Identify outliers
    outlier_mask = median_distances > thresh
    outlier_indices = np.where(outlier_mask)[0]
    inlier_indices = np.where(~outlier_mask)[0]

    outlier_ids = [embryo_ids[i] for i in outlier_indices]
    inlier_ids = [embryo_ids[i] for i in inlier_indices]

    if verbose:
        print(f"  Threshold: {thresh:.3f}")
        print(f"  Outliers detected: {len(outlier_ids)}")
        print(f"  Inliers retained: {len(inlier_ids)}")

        if len(outlier_ids) > 0:
            print(f"\n  Outlier embryos:")
            for embryo_id, med_dist in zip(
                [embryo_ids[i] for i in outlier_indices],
                median_distances[outlier_indices]
            ):
                print(f"    {embryo_id}: median_dist = {med_dist:.3f}")

    # Package info
    info = {
        'median_distances': median_distances,
        'threshold': thresh,
        'outlier_indices': outlier_indices,
        'inlier_indices': inlier_indices,
        'method': method,
    }

    return outlier_ids, inlier_ids, info


def remove_outliers_from_distance_matrix(
    D: np.ndarray,
    embryo_ids: List[str],
    outlier_detection_method: str = 'percentile',
    outlier_threshold: Optional[float] = None,
    outlier_percentile: float = 95,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Remove outlier embryos from distance matrix.

    Convenience wrapper around identify_outliers() that returns a cleaned
    distance matrix ready for clustering.

    Args:
        D: Distance matrix (n_embryos, n_embryos)
        embryo_ids: List of embryo identifiers
        outlier_detection_method: Method for outlier detection ('median_distance', 'percentile', 'mad')
        outlier_threshold: Manual threshold (for 'median_distance' method)
        outlier_percentile: Percentile cutoff (for 'percentile' method)
        verbose: Print diagnostic information

    Returns:
        D_clean: Distance matrix with outliers removed
        embryo_ids_clean: List of non-outlier embryo IDs
        info: Dict with outlier detection information (from identify_outliers)

    Example:
        >>> D_clean, embryo_ids_clean, info = remove_outliers_from_distance_matrix(
        ...     D, embryo_ids, method='percentile', percentile=95
        ... )
        >>> print(f"Removed {len(info['outlier_indices'])} outliers")
        >>> print(f"Clean distance matrix shape: {D_clean.shape}")
    """
    # Identify outliers
    outlier_ids, inlier_ids, info = identify_outliers(
        D,
        embryo_ids,
        method=outlier_detection_method,
        threshold=outlier_threshold,
        percentile=outlier_percentile,
        verbose=verbose,
    )

    # Extract clean distance matrix (inliers only)
    inlier_idx = info['inlier_indices']
    D_clean = D[np.ix_(inlier_idx, inlier_idx)]

    if verbose:
        print(f"\n✓ Outliers removed")
        print(f"  Original size: {D.shape}")
        print(f"  Clean size: {D_clean.shape}")

    return D_clean, inlier_ids, info


def plot_dendrogram(
    D: np.ndarray,
    embryo_ids: List[str],
    *,
    linkage_method: str = 'average',
    k_highlight: Optional[List[int]] = None,
    color_threshold: Optional[float] = None,
    truncate_mode: Optional[str] = None,
    truncate_p: int = 30,
    orientation: str = 'top',
    figsize: Tuple[float, float] = (14, 8),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    verbose: bool = True,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Plot hierarchical clustering dendrogram from distance matrix.

    Visualizes the hierarchical structure of embryo clustering to help select
    optimal K. Supports highlighting different K cutoffs and provides cluster
    assignments for selected K values.

    Args:
        D: Distance matrix (n_embryos, n_embryos) - output from compute_md_dtw_distance_matrix()
        embryo_ids: List of embryo identifiers (same order as D rows)
        linkage_method: Linkage method for hierarchical clustering.
                       Options: 'average' (UPGMA), 'single', 'complete', 'ward' (ward requires euclidean)
                       Default: 'average' (best for DTW distances)
        k_highlight: List of K values to show as horizontal cutoff lines.
                    Example: [2, 3, 4] shows lines where dendrogram would be cut for each K.
        color_threshold: Height at which to color branches (clusters below this are same color).
                        If None, uses scipy default (0.7 * max height).
        truncate_mode: Truncate dendrogram for large N. Options: 'lastp', 'level', None
        truncate_p: Parameter for truncation (e.g., show last p clusters)
        orientation: Dendrogram orientation: 'top', 'bottom', 'left', 'right'
        figsize: Figure size (width, height)
        title: Plot title. If None, auto-generated.
        save_path: Path to save figure. If None, not saved.
        dpi: Resolution for saved figure.
        verbose: Print diagnostic information.

    Returns:
        fig: matplotlib Figure object
        info: Dict with useful outputs:
            - 'linkage_matrix': scipy linkage matrix Z
            - 'dendrogram_data': scipy dendrogram output dict
            - 'cluster_assignments': Dict[k, np.ndarray] with cluster labels for each k in k_highlight
            - 'k_cutoff_heights': Dict[k, float] with cutoff heights for each k

    Example:
        >>> D = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3)
        >>> fig, info = plot_dendrogram(
        ...     D, embryo_ids,
        ...     k_highlight=[2, 3, 4],
        ...     title='b9d2 Mutant Clustering'
        ... )
        >>> # Access cluster assignments for k=3
        >>> labels_k3 = info['cluster_assignments'][3]

    Notes:
        - Uses condensed distance format internally (scipy requirement)
        - Average linkage (UPGMA) is recommended for DTW distances
        - Ward linkage requires Euclidean distances (not recommended for DTW)
    """
    n = len(D)
    
    if verbose:
        print(f"Generating dendrogram...")
        print(f"  Embryos: {n}")
        print(f"  Linkage method: {linkage_method}")

    # Convert square distance matrix to condensed form for scipy
    # Ensure matrix is exactly symmetric (handle floating point)
    D_sym = (D + D.T) / 2
    np.fill_diagonal(D_sym, 0)
    D_condensed = squareform(D_sym)

    # Compute linkage
    Z = linkage(D_condensed, method=linkage_method)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set color threshold
    if color_threshold is None:
        # Default: color at 70% of max height
        color_threshold = 0.7 * Z[:, 2].max()

    # Plot dendrogram
    dendro_kwargs = {
        'Z': Z,
        'labels': embryo_ids,
        'ax': ax,
        'orientation': orientation,
        'color_threshold': color_threshold,
        'above_threshold_color': 'gray',
        'leaf_rotation': 90 if orientation in ['top', 'bottom'] else 0,
        'leaf_font_size': max(6, min(10, 200 // n)),  # Scale font with N
    }

    if truncate_mode:
        dendro_kwargs['truncate_mode'] = truncate_mode
        dendro_kwargs['p'] = truncate_p

    dendro_data = dendrogram(**dendro_kwargs)

    # Prepare output info
    info = {
        'linkage_matrix': Z,
        'dendrogram_data': dendro_data,
        'cluster_assignments': {},
        'k_cutoff_heights': {},
    }

    # Add horizontal lines for k_highlight values
    if k_highlight:
        # Sort merge heights
        heights = Z[:, 2]

        # For k clusters, we need n-k merges, so cutoff is between merge n-k-1 and n-k
        for k in sorted(k_highlight):
            if k < 2 or k > n:
                if verbose:
                    print(f"  WARNING: k={k} out of range [2, {n}], skipping")
                continue

            # Height to cut at: midpoint between merge that creates k clusters and k-1 clusters
            # After n-k merges we have k clusters
            merge_idx_before = n - k - 1  # Last merge before having k clusters
            merge_idx_after = n - k       # First merge after having k clusters

            if merge_idx_before >= 0:
                h_before = heights[merge_idx_before]
            else:
                h_before = 0

            if merge_idx_after < len(heights):
                h_after = heights[merge_idx_after]
            else:
                h_after = heights[-1] * 1.1

            # Cut at midpoint
            cutoff_height = (h_before + h_after) / 2
            info['k_cutoff_heights'][k] = cutoff_height

            # Get cluster assignments at this k
            labels = fcluster(Z, k, criterion='maxclust')
            # Convert to 0-indexed
            labels = labels - 1
            info['cluster_assignments'][k] = labels

            # Draw horizontal line
            color = plt.cm.tab10(k % 10)
            ax.axhline(y=cutoff_height, color=color, linestyle='--', alpha=0.7,
                      label=f'k={k} (h={cutoff_height:.2f})')

            if verbose:
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"  k={k}: cutoff height={cutoff_height:.3f}, cluster sizes: {dict(zip(unique_labels, counts))}")

    # Formatting
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Hierarchical Clustering Dendrogram (n={n}, linkage={linkage_method})',
                    fontsize=14, fontweight='bold')

    if orientation in ['top', 'bottom']:
        ax.set_xlabel('Embryo ID', fontsize=11)
        ax.set_ylabel('MD-DTW Distance', fontsize=11)
    else:
        ax.set_xlabel('MD-DTW Distance', fontsize=11)
        ax.set_ylabel('Embryo ID', fontsize=11)

    if k_highlight:
        ax.legend(loc='upper right', fontsize=10)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"  Saved to: {save_path}")

    if verbose:
        print(f"✓ Dendrogram generated")

    return fig, info


def plot_dendrogram_with_categories(
    D: np.ndarray,
    embryo_ids: List[str],
    category_df: pd.DataFrame,
    category_cols: List[str] = ['pair', 'genotype'],
    *,
    linkage_method: str = 'average',
    k_highlight: Optional[List[int]] = None,
    color_threshold: Optional[float] = None,
    truncate_mode: Optional[str] = None,
    truncate_p: int = 30,
    orientation: str = 'top',
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    spacer_height: float = .7,
    verbose: bool = True,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Plot hierarchical clustering dendrogram with categorical color bars.

    Extended version of plot_dendrogram() that adds colored bars below the dendrogram
    showing categorical groupings (e.g., pair, genotype). Useful for visualizing how
    clusters relate to experimental design or biological categories.

    Args:
        D: Distance matrix (n_embryos, n_embryos) - output from compute_md_dtw_distance_matrix()
        embryo_ids: List of embryo identifiers (same order as D rows)
        category_df: DataFrame with 'embryo_id' column plus categorical columns.
                    Example: ['embryo_id', 'pair', 'genotype']
        category_cols: List of category column names to show as color bars.
                      Default: ['pair', 'genotype']
        linkage_method: Linkage method ('average', 'single', 'complete', 'ward')
        k_highlight: List of K values to show as horizontal cutoff lines
        color_threshold: Height at which to color branches (if None, uses scipy default)
        truncate_mode: Truncate dendrogram for large N ('lastp', 'level', None)
        truncate_p: Parameter for truncation
        orientation: Dendrogram orientation ('top', 'bottom', 'left', 'right')
        figsize: Figure size (width, height). Auto-calculated if None.
        title: Plot title
        save_path: Path to save figure
        dpi: Resolution for saved figure
        spacer_height: Height (in inches) of white space between dendrogram and first category bar.
                      Increase this value if leaf labels overlap with category bars. Default: 1.2
        verbose: Print diagnostic information

    Returns:
        fig: matplotlib Figure object
        info: Dict with:
            - 'linkage_matrix': scipy linkage matrix Z
            - 'dendrogram_data': scipy dendrogram output dict
            - 'cluster_assignments': Dict[k, np.ndarray] with cluster labels for each k
            - 'k_cutoff_heights': Dict[k, float] with cutoff heights for each k
            - 'category_colors': Dict[category_col, Dict[value, color]] color mappings

    Example:
        >>> # Prepare category data
        >>> category_df = pd.DataFrame({
        ...     'embryo_id': embryo_ids,
        ...     'pair': ['pair1', 'pair1', 'pair2', ...],
        ...     'genotype': ['wt', 'mut', 'wt', ...]
        ... })
        >>>
        >>> # Plot with both pair and genotype bars
        >>> fig, info = plot_dendrogram_with_categories(
        ...     D, embryo_ids, category_df,
        ...     category_cols=['pair', 'genotype'],
        ...     k_highlight=[2, 3, 4]
        ... )

    Notes:
        - Color bars are shown below dendrogram (for orientation='top')
        - Each category gets its own colored bar with legend
        - Bars are aligned with dendrogram leaves
        - Missing categories are filled with 'unknown'
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    n = len(D)

    if verbose:
        print(f"Generating dendrogram with category bars...")
        print(f"  Embryos: {n}")
        print(f"  Categories: {category_cols}")
        print(f"  Linkage method: {linkage_method}")

    # Step 1: Validate and prepare category data
    # Create lookup: embryo_id -> {category_col: value}
    category_lookup = {}
    for col in category_cols:
        if col not in category_df.columns:
            raise ValueError(f"Category column '{col}' not found in category_df. Available: {list(category_df.columns)}")

        # Create mapping for this category
        col_map = dict(zip(category_df['embryo_id'], category_df[col]))
        # Fill missing with 'unknown'
        col_map_filled = {eid: col_map.get(eid, 'unknown') for eid in embryo_ids}
        category_lookup[col] = col_map_filled

    if verbose:
        for col in category_cols:
            unique_vals = set(category_lookup[col].values())
            print(f"  '{col}': {len(unique_vals)} unique values: {sorted(unique_vals)}")

    # Step 2: Compute linkage (same as original)
    D_sym = (D + D.T) / 2
    np.fill_diagonal(D_sym, 0)
    D_condensed = squareform(D_sym)
    Z = linkage(D_condensed, method=linkage_method)

    # Step 3: Create figure with GridSpec
    n_category_bars = len(category_cols)
    bar_height_per_category = 0.4  # inches
    # spacer_height is now a parameter - white space between dendrogram and first bar
    total_bar_height = n_category_bars * bar_height_per_category + spacer_height
    extra_label_space = 1.5  # Extra height to keep rotated labels off the category bars

    if figsize is None:
        dendro_height = 8
        fig_width = 14
        fig_height = dendro_height + total_bar_height + 0.5 + extra_label_space
        figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize)
    # Use a spacer row between dendrogram and category bars
    gs = gridspec.GridSpec(
        nrows=2 + n_category_bars,  # dendro + spacer + category bars
        ncols=1,
        height_ratios=[8, spacer_height] + [bar_height_per_category] * n_category_bars,
        hspace=0.15  # More breathing room between dendrogram and category bars
    )

    ax_dendro = fig.add_subplot(gs[0, 0])
    # gs[1, 0] is the spacer - we don't create an axis for it
    ax_categories = [fig.add_subplot(gs[i+2, 0]) for i in range(n_category_bars)]

    # Step 4: Plot dendrogram (same as original)
    if color_threshold is None:
        color_threshold = 0.7 * Z[:, 2].max()

    dendro_kwargs = {
        'Z': Z,
        'labels': embryo_ids,
        'ax': ax_dendro,
        'orientation': orientation,
        'color_threshold': color_threshold,
        'above_threshold_color': 'gray',
        'leaf_rotation': 90 if orientation in ['top', 'bottom'] else 0,
        'leaf_font_size': max(6, min(10, 200 // n)),
    }

    if truncate_mode:
        dendro_kwargs['truncate_mode'] = truncate_mode
        dendro_kwargs['p'] = truncate_p

    dendro_data = dendrogram(**dendro_kwargs)

    # Get leaf order
    leaf_order = dendro_data['leaves']  # Indices into embryo_ids

    # Step 5: Prepare output info
    info = {
        'linkage_matrix': Z,
        'dendrogram_data': dendro_data,
        'cluster_assignments': {},
        'k_cutoff_heights': {},
        'category_colors': {},
    }

    # Add k_highlight lines (same as original)
    if k_highlight:
        heights = Z[:, 2]
        for k in sorted(k_highlight):
            if k < 2 or k > n:
                if verbose:
                    print(f"  WARNING: k={k} out of range [2, {n}], skipping")
                continue

            merge_idx_before = n - k - 1
            merge_idx_after = n - k

            h_before = heights[merge_idx_before] if merge_idx_before >= 0 else 0
            h_after = heights[merge_idx_after] if merge_idx_after < len(heights) else heights[-1] * 1.1

            cutoff_height = (h_before + h_after) / 2
            info['k_cutoff_heights'][k] = cutoff_height

            labels = fcluster(Z, k, criterion='maxclust') - 1
            info['cluster_assignments'][k] = labels

            color = plt.cm.tab10(k % 10)
            ax_dendro.axhline(y=cutoff_height, color=color, linestyle='--', alpha=0.7,
                             label=f'k={k} (h={cutoff_height:.2f})')

    # Step 6: Plot category color bars
    for i, cat_col in enumerate(category_cols):
        ax = ax_categories[i]

        # Get category values in dendrogram leaf order
        ordered_embryo_ids = [embryo_ids[idx] for idx in leaf_order]
        cat_values = [category_lookup[cat_col][eid] for eid in ordered_embryo_ids]

        # Map category values to colors using smart assignment
        unique_cats = sorted(set(cat_values))
        color_map = {}
        
        # Use appropriate color scheme based on category type
        if cat_col in ['genotype', 'Genotype', 'geno']:
            # Use standard genotype colors
            for cat in unique_cats:
                color_map[cat] = get_color_for_genotype(str(cat))
        elif cat_col in ['pair', 'Pair', 'cross', 'Cross']:
            # Use pastel colors for pairs
            for j, cat in enumerate(unique_cats):
                color_map[cat] = PASTEL_COLORS[j % len(PASTEL_COLORS)]
        else:
            # Use standard palette for other categories
            standard_palette = plt.cm.tab20.colors
            for j, cat in enumerate(unique_cats):
                color_map[cat] = standard_palette[j % len(standard_palette)]
        
        info['category_colors'][cat_col] = color_map

        # Create color array for imshow
        color_indices = [unique_cats.index(val) for val in cat_values]
        color_array = np.array([color_indices])

        # Plot as image
        cmap = plt.matplotlib.colors.ListedColormap([color_map[cat] for cat in unique_cats])
        ax.imshow(color_array, aspect='auto', interpolation='nearest', cmap=cmap,
                 vmin=0, vmax=len(unique_cats)-1)

        # Formatting
        ax.set_yticks([0])
        ax.set_yticklabels([cat_col], fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_xlim(-0.5, len(ordered_embryo_ids) - 0.5)  # Match dendrogram width

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add legend
        legend_elements = [Patch(facecolor=color_map[cat], edgecolor='black', label=cat)
                          for cat in unique_cats]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                 fontsize=9, frameon=False)

    # Step 7: Format dendrogram axis
    if title:
        ax_dendro.set_title(title, fontsize=14, fontweight='bold', pad=10)
    else:
        ax_dendro.set_title(f'Hierarchical Clustering Dendrogram with Categories (n={n}, linkage={linkage_method})',
                           fontsize=14, fontweight='bold', pad=10)

    if orientation in ['top', 'bottom']:
        ax_dendro.set_xlabel('', fontsize=11)  # Labels shown by category bars
        ax_dendro.set_ylabel('MD-DTW Distance', fontsize=11)
    else:
        ax_dendro.set_xlabel('MD-DTW Distance', fontsize=11)
        ax_dendro.set_ylabel('', fontsize=11)

    if k_highlight:
        ax_dendro.legend(loc='upper right', fontsize=10, framealpha=0.9)

    ax_dendro.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"  Saved to: {save_path}")

    if verbose:
        print(f"✓ Dendrogram with categories generated")

    return fig, info


if __name__ == "__main__":
    """
    Complete workflow test: MD-DTW → Dendrogram → Cluster Trajectories
    
    Tests the full pipeline:
    1. Create synthetic data with 3 distinct phenotype clusters
    2. Prepare multivariate array (curvature + length)
    3. Compute MD-DTW distance matrix
    4. Generate dendrogram with k cutoffs
    5. Plot multimetric trajectories colored by cluster
    """
    print("=" * 70)
    print("MD-DTW Prototype - Complete Workflow Test")
    print("=" * 70)

    # Create synthetic test data with 3 distinct clusters
    np.random.seed(42)
    n_embryos = 15  # 5 per cluster
    n_timepoints = 30
    time_grid = np.linspace(18, 48, n_timepoints)

    data_rows = []
    for i in range(n_embryos):
        embryo_id = f"embryo_{i:02d}"
        
        # Cluster 0: High curvature, short length (CE-like)
        if i < 5:
            curvature = 3.5 + 0.4 * np.sin(time_grid / 8) + np.random.normal(0, 0.1, n_timepoints)
            length = 250 + 3 * time_grid + np.random.normal(0, 5, n_timepoints)
        # Cluster 1: High curvature, normal length (HTA-like)
        elif i < 10:
            curvature = 3.2 + 0.3 * np.cos(time_grid / 7 + 1) + np.random.normal(0, 0.1, n_timepoints)
            length = 350 + 5 * time_grid + np.random.normal(0, 5, n_timepoints)
        # Cluster 2: Low curvature, long length (WT-like)
        else:
            curvature = 1.8 + 0.2 * np.sin(time_grid / 10 - 1) + np.random.normal(0, 0.1, n_timepoints)
            length = 400 + 6 * time_grid + np.random.normal(0, 5, n_timepoints)

        for t_idx, t in enumerate(time_grid):
            data_rows.append({
                'embryo_id': embryo_id,
                'predicted_stage_hpf': t,
                'baseline_deviation_normalized': curvature[t_idx],
                'total_length_um': length[t_idx],
            })

    df = pd.DataFrame(data_rows)

    print(f"\nTest DataFrame: {len(df)} rows, {df['embryo_id'].nunique()} embryos")
    print("  Expected clusters: 3 (CE-like, HTA-like, WT-like)")

    # Step 1: Prepare multivariate array
    print("\n" + "=" * 70)
    print("Step 1: Preparing Multivariate Array")
    print("=" * 70)

    X, embryo_ids, time_grid_out = prepare_multivariate_array(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        normalize=True,
        verbose=True
    )

    # Step 2: Compute MD-DTW distance matrix
    print("\n" + "=" * 70)
    print("Step 2: Computing MD-DTW Distance Matrix")
    print("=" * 70)

    D = compute_md_dtw_distance_matrix(
        X,
        sakoe_chiba_radius=3,
        verbose=True
    )

    # Step 3: Generate dendrogram and get cluster assignments
    print("\n" + "=" * 70)
    print("Step 3: Generating Dendrogram (k=2,3,4)")
    print("=" * 70)

    fig_dendro, dendro_info = plot_dendrogram(
        D,
        embryo_ids,
        k_highlight=[2, 3, 4],
        title='Synthetic Test: MD-DTW Clustering (Curvature + Length)',
        save_path='test_workflow_dendrogram.png',
        figsize=(16, 8),
        verbose=True
    )
    plt.close(fig_dendro)

    # Step 4: Add cluster labels to DataFrame and plot trajectories
    print("\n" + "=" * 70)
    print("Step 4: Plotting Multimetric Trajectories by Cluster (k=3)")
    print("=" * 70)

    # Import plotting function
    from src.analyze.trajectory_analysis.faceted_plotting import plot_multimetric_trajectories

    # Get cluster assignments for k=3
    k_focus = 3
    cluster_labels = dendro_info['cluster_assignments'][k_focus]
    
    # Create lookup and add to DataFrame
    label_lookup = dict(zip(embryo_ids, cluster_labels))
    df_plot = df.copy()
    df_plot['md_dtw_cluster'] = df_plot['embryo_id'].map(label_lookup)
    
    print(f"  Cluster assignments for k={k_focus}:")
    for c in sorted(df_plot['md_dtw_cluster'].unique()):
        embryo_count = df_plot[df_plot['md_dtw_cluster'] == c]['embryo_id'].nunique()
        print(f"    Cluster {c}: {embryo_count} embryos")

    # Plot multimetric trajectories (metrics in rows, clusters in columns)
    print(f"\n  Generating multimetric plot...")
    try:
        fig_multi = plot_multimetric_trajectories(
            df_plot,
            metrics=['baseline_deviation_normalized', 'total_length_um'],
            col_by='md_dtw_cluster',
            color_by='md_dtw_cluster',
            x_col='predicted_stage_hpf',
            metric_labels={
                'baseline_deviation_normalized': 'Curvature',
                'total_length_um': 'Body Length (μm)',
            },
            title='Synthetic Test: Trajectories by MD-DTW Cluster',
            x_label='Time (hpf)',
            backend='matplotlib',
            bin_width=2.0,
        )
        
        save_path = 'test_workflow_multimetric.png'
        fig_multi.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig_multi)
        print(f"  ✓ Saved: {save_path}")
    except Exception as e:
        print(f"  WARNING: Could not generate multimetric plot: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("✓ Complete Workflow Test Finished!")
    print("=" * 70)
    print(f"Outputs:")
    print(f"  - test_workflow_dendrogram.png (dendrogram with k cutoffs)")
    print(f"  - test_workflow_multimetric.png (trajectories by cluster)")
    print(f"\nThis workflow demonstrates:")
    print(f"  1. MD-DTW clustering on multivariate trajectories")
    print(f"  2. Dendrogram visualization for K selection")
    print(f"  3. Trajectory plots colored by cluster assignment")
    print("=" * 70)
