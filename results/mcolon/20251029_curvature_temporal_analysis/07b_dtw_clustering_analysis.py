#!/usr/bin/env python3
"""
DTW-based clustering analysis of curvature trajectories.

Clusters homozygous mutant embryo curvature trajectories using Dynamic Time Warping (DTW)
distance and tests for anti-correlated early/late curvature patterns.

Specification: 07b_SPECIFICATION.md
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.stats import pearsonr, linregress
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Import load_data from same directory
import importlib.util
load_data_path = SCRIPT_DIR / 'load_data.py'
spec = importlib.util.spec_from_file_location('load_data', load_data_path)
load_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data)

get_analysis_dataframe = load_data.get_analysis_dataframe
GENOTYPE_SHORT = load_data.GENOTYPE_SHORT

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test mode: set to False for full analysis
TEST_MODE = False
N_TEST_EMBRYOS = 10

# Analysis parameters
METRIC_NAME = 'normalized_baseline_deviation'
GENOTYPE_FILTER = 'cep290_homozygous'
EARLY_WINDOW = (44, 50)  # hpf
LATE_WINDOW = (80, 100)  # hpf
DTW_WINDOW = 3  # Sakoe-Chiba band
CLUSTER_K_VALUES = [2, 3, 4, 5, 6, 7]
INTERPOLATION_GRID_STEP = 0.5  # hpf

# Output directory
RESULTS_DIR = Path(__file__).resolve().parent / 'outputs' / '07b_dtw_clustering_analysis'
PLOTS_DIR = RESULTS_DIR / 'plots'
TABLES_DIR = RESULTS_DIR / 'tables'

# ============================================================================
# STEP 1: Data Extraction & Preparation
# ============================================================================

def extract_trajectories(df, genotype_filter, metric_name, min_timepoints=3):
    """
    Extract per-embryo trajectories from long-format dataframe.

    Returns
    -------
    trajectories : list of arrays
        List of trajectory arrays (variable lengths)
    embryo_ids : list of str
        Corresponding embryo IDs
    trajectories_df : pd.DataFrame
        Long-format dataframe (embryo_id, hpf, metric_value)
    """
    print(f"\n{'='*80}")
    print("STEP 1: DATA EXTRACTION & PREPARATION")
    print(f"{'='*80}")

    # Filter for genotype
    df_filtered = df[df['genotype'] == genotype_filter].copy()
    print(f"\n  Filtered to {genotype_filter}: {len(df_filtered)} timepoints")

    # Extract relevant columns
    df_long = df_filtered[['embryo_id', 'predicted_stage_hpf', metric_name]].copy()
    df_long.columns = ['embryo_id', 'hpf', 'metric_value']

    # Drop NaN values
    initial_rows = len(df_long)
    df_long = df_long.dropna(subset=['metric_value'])
    dropped = initial_rows - len(df_long)
    print(f"  Dropped {dropped} rows with NaN metric values: {len(df_long)} remaining")

    # Extract per-embryo trajectories
    trajectories = []
    embryo_ids = []

    for embryo_id, group in df_long.groupby('embryo_id'):
        trajectory = group.sort_values('hpf')['metric_value'].values

        if len(trajectory) >= min_timepoints:
            trajectories.append(trajectory)
            embryo_ids.append(embryo_id)

    print(f"\n  Extracted {len(trajectories)} embryo trajectories (min {min_timepoints} timepoints)")
    print(f"  Mean trajectory length: {np.mean([len(t) for t in trajectories]):.1f} timepoints")

    return trajectories, embryo_ids, df_long


# ============================================================================
# STEP 2: Missing Data Handling
# ============================================================================

def interpolate_trajectories(df_long):
    """
    Apply linear interpolation to handle missing values within trajectories.

    Returns
    -------
    df_long : pd.DataFrame
        Imputed dataframe
    """
    print(f"\n{'='*80}")
    print("STEP 2: MISSING DATA HANDLING (INTERPOLATION)")
    print(f"{'='*80}")

    df_long = df_long.copy()

    interpolated_count = 0

    for embryo_id in df_long['embryo_id'].unique():
        embryo_data = df_long[df_long['embryo_id'] == embryo_id].copy()

        # Check for gaps in timepoints
        hpf_values = embryo_data['hpf'].values

        if len(embryo_data) > 1 and embryo_data['metric_value'].isna().sum() > 0:
            # Interpolate missing values
            f = interpolate.interp1d(
                embryo_data[~embryo_data['metric_value'].isna()]['hpf'].values,
                embryo_data[~embryo_data['metric_value'].isna()]['metric_value'].values,
                kind='linear',
                fill_value='extrapolate'
            )

            # Fill NaN values with interpolated values
            nan_mask = embryo_data['metric_value'].isna()
            df_long.loc[embryo_data[nan_mask].index, 'metric_value'] = f(
                embryo_data[nan_mask]['hpf'].values
            )
            interpolated_count += nan_mask.sum()

    print(f"\n  Interpolated {interpolated_count} missing values")
    print(f"  Remaining NaN values: {df_long['metric_value'].isna().sum()}")

    return df_long


# ============================================================================
# STEP 3: Trajectory Interpolation to Common Timepoints
# ============================================================================

def interpolate_to_common_grid(df_long, grid_step=0.5):
    """
    Interpolate all trajectories to a common timepoint grid.

    NO edge padding - only interpolate within observed range.
    Shorter trajectories will remain truncated.

    Returns
    -------
    interpolated_trajectories : list of arrays
        Trajectories at common timepoint grid
    common_grid : array
        Common timepoint grid (hpf values)
    original_lengths : dict
        Original trajectory length for each embryo
    """
    print(f"\n{'='*80}")
    print("STEP 3: TRAJECTORY INTERPOLATION TO COMMON TIMEPOINTS")
    print(f"{'='*80}")

    # Find min/max hpf across all trajectories
    min_hpf = df_long.groupby('embryo_id')['hpf'].min().min()
    max_hpf = df_long.groupby('embryo_id')['hpf'].max().max()

    print(f"\n  HPF range: {min_hpf:.1f} to {max_hpf:.1f}")

    # Create common timepoint grid
    common_grid = np.arange(min_hpf, max_hpf + grid_step, grid_step)
    print(f"  Common grid: {len(common_grid)} timepoints (step={grid_step} hpf)")

    # Interpolate each trajectory
    interpolated_trajectories = []
    original_lengths = {}
    embryo_ids_ordered = []

    for embryo_id, group in df_long.groupby('embryo_id'):
        group_sorted = group.sort_values('hpf')
        hpf_vals = group_sorted['hpf'].values
        metric_vals = group_sorted['metric_value'].values

        original_lengths[embryo_id] = len(metric_vals)

        # Linear interpolation to common grid
        f = interpolate.interp1d(
            hpf_vals,
            metric_vals,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Interpolate - will have NaN where outside observed range
        interpolated = f(common_grid)

        # Only keep timepoints within observed range (no padding)
        valid_mask = ~np.isnan(interpolated)
        interpolated_trimmed = interpolated[valid_mask]

        if len(interpolated_trimmed) > 0:  # Only keep if we have data
            interpolated_trajectories.append(interpolated_trimmed)
            embryo_ids_ordered.append(embryo_id)

    print(f"\n  Interpolated shape: {len(interpolated_trajectories)} embryos")
    print(f"  Interpolated lengths: min={min([len(t) for t in interpolated_trajectories])}, "
          f"max={max([len(t) for t in interpolated_trajectories])}, "
          f"mean={np.mean([len(t) for t in interpolated_trajectories]):.1f}")

    return interpolated_trajectories, embryo_ids_ordered, original_lengths, common_grid


# ============================================================================
# DTW Distance Computation (Custom Implementation)
# ============================================================================

def compute_dtw_distance(seq1, seq2, window=3, normalize=False):
    """
    Compute Dynamic Time Warping distance between two sequences with Sakoe-Chiba band constraint.

    Parameters
    ----------
    seq1, seq2 : array-like
        Input sequences (1D arrays)
    window : int
        Sakoe-Chiba band width constraint. Automatically expanded to handle length differences.
    normalize : bool
        If True, return per-step distance (length-independent)

    Returns
    -------
    float
        DTW distance
    """
    # Convert to numeric arrays (prevents LAPACK errors with object dtypes)
    seq1 = np.asarray(seq1, dtype=float)
    seq2 = np.asarray(seq2, dtype=float)
    n, m = len(seq1), len(seq2)

    # Expand window to handle sequence length differences
    w = max(window, abs(n - m))

    # Initialize cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill the cost matrix with band constraint
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)
        for j in range(j_start, j_end):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    distance = dtw_matrix[n, m]

    # Normalize by path length if requested (length-independent metric)
    if normalize and not np.isinf(distance):
        distance = distance / (n + m)

    return distance


# ============================================================================
# STEP 4: DTW Distance Matrix Computation
# ============================================================================

def compute_dtw_distance_matrix(trajectories, window=3):
    """
    Compute pairwise DTW distances with Sakoe-Chiba band.

    Returns
    -------
    distance_matrix : np.ndarray
        Symmetric (n_embryos × n_embryos) distance matrix
    """
    print(f"\n{'='*80}")
    print("STEP 4: DTW DISTANCE MATRIX COMPUTATION")
    print(f"{'='*80}")

    n_trajectories = len(trajectories)
    distance_matrix = np.zeros((n_trajectories, n_trajectories))

    print(f"\n  Computing pairwise DTW distances (window={window})...")

    for i in range(n_trajectories):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{n_trajectories}")

        for j in range(i + 1, n_trajectories):
            try:
                dist = compute_dtw_distance(
                    trajectories[i],
                    trajectories[j],
                    window=window
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
            except Exception as e:
                print(f"    Warning: DTW computation failed for pair ({i}, {j}): {e}")
                distance_matrix[i, j] = np.inf
                distance_matrix[j, i] = np.inf

    # Validate
    nan_count = np.isnan(distance_matrix).sum()
    inf_count = np.isinf(distance_matrix).sum()
    diag_check = np.allclose(np.diag(distance_matrix), 0)

    print(f"\n  Validation:")
    print(f"    NaN count: {nan_count}")
    print(f"    Inf count: {inf_count}")
    print(f"    Diagonal ≈ 0: {diag_check}")
    print(f"    Distance stats: min={np.nanmin(distance_matrix):.3f}, "
          f"max={np.nanmax(distance_matrix):.3f}, "
          f"mean={np.nanmean(distance_matrix):.3f}")
    print(f"    Matrix shape: {distance_matrix.shape}")

    return distance_matrix


# ============================================================================
# STEP 5: K-means Clustering
# ============================================================================

def perform_clustering(distance_matrix, k_values=[2, 3, 4]):
    """
    Test multiple k values with hierarchical clustering using precomputed distance matrix.

    Returns
    -------
    clustering_results : dict
        Results for each k value
    """
    print(f"\n{'='*80}")
    print("STEP 5: HIERARCHICAL CLUSTERING")
    print(f"{'='*80}")

    clustering_results = {}

    for k in k_values:
        print(f"\n  Testing k={k}...")

        try:
            # Hierarchical clustering accepts precomputed distance matrix
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric='precomputed',
                linkage='average'
            )

            assignments = clusterer.fit_predict(distance_matrix)

            # Compute silhouette score
            silhouette = silhouette_score(distance_matrix, assignments, metric='precomputed')

            clustering_results[k] = {
                'assignments': assignments,
                'silhouette': silhouette,
                'clusterer_object': clusterer
            }

            print(f"    Silhouette score: {silhouette:.4f}")
            print(f"    Cluster sizes: {np.bincount(assignments)}")

        except Exception as e:
            print(f"    Error: Clustering failed for k={k}: {e}")

    return clustering_results


# ============================================================================
# STEP 6: Select Best K
# ============================================================================

def select_best_k(clustering_results):
    """
    Select k with highest silhouette score.

    Returns
    -------
    best_k : int
    best_assignments : array
    """
    print(f"\n{'='*80}")
    print("STEP 6: SELECT BEST K")
    print(f"{'='*80}")

    best_k = max(
        clustering_results.keys(),
        key=lambda k: clustering_results[k]['silhouette']
    )

    best_silhouette = clustering_results[best_k]['silhouette']
    best_assignments = clustering_results[best_k]['assignments']

    print(f"\n  Best k: {best_k}")
    print(f"  Silhouette score: {best_silhouette:.4f}")
    print(f"  Cluster sizes: {np.bincount(best_assignments)}")

    return best_k, best_assignments


# ============================================================================
# STEP 7: Extract Early/Late Means
# ============================================================================

def extract_early_late_means(df_long, embryo_ids, early_window, late_window):
    """
    Extract mean metric values in early and late windows for each embryo.

    Returns
    -------
    early_means : dict
        {embryo_id: mean_value}
    late_means : dict
        {embryo_id: mean_value}
    """
    print(f"\n{'='*80}")
    print("STEP 7: EXTRACT EARLY/LATE MEANS")
    print(f"{'='*80}")

    print(f"\n  Early window: {early_window[0]}-{early_window[1]} hpf")
    print(f"  Late window: {late_window[0]}-{late_window[1]} hpf")

    early_means = {}
    late_means = {}

    for embryo_id in embryo_ids:
        embryo_data = df_long[df_long['embryo_id'] == embryo_id]

        # Early window
        early_data = embryo_data[
            (embryo_data['hpf'] >= early_window[0]) &
            (embryo_data['hpf'] <= early_window[1])
        ]
        if len(early_data) > 0:
            early_means[embryo_id] = early_data['metric_value'].mean()
        else:
            early_means[embryo_id] = np.nan

        # Late window
        late_data = embryo_data[
            (embryo_data['hpf'] >= late_window[0]) &
            (embryo_data['hpf'] <= late_window[1])
        ]
        if len(late_data) > 0:
            late_means[embryo_id] = late_data['metric_value'].mean()
        else:
            late_means[embryo_id] = np.nan

    # Convert to arrays (in embryo_ids order)
    early_means_arr = np.array([early_means.get(e, np.nan) for e in embryo_ids])
    late_means_arr = np.array([late_means.get(e, np.nan) for e in embryo_ids])

    print(f"\n  Early means: {np.nanmean(early_means_arr):.4f} ± {np.nanstd(early_means_arr):.4f}")
    print(f"  Late means: {np.nanmean(late_means_arr):.4f} ± {np.nanstd(late_means_arr):.4f}")

    return early_means_arr, late_means_arr


# ============================================================================
# STEP 8: Anti-Correlation Test
# ============================================================================

def test_anticorrelation(best_assignments, early_means_arr, late_means_arr, embryo_ids):
    """
    Test for anti-correlation between early and late means within each cluster.

    Returns
    -------
    anticorr_results : dict
    """
    print(f"\n{'='*80}")
    print("STEP 8: ANTI-CORRELATION TEST")
    print(f"{'='*80}")

    anticorr_results = {}

    unique_clusters = np.unique(best_assignments)

    for cluster_id in unique_clusters:
        cluster_mask = best_assignments == cluster_id
        cluster_embryos = np.array(embryo_ids)[cluster_mask]

        early_vals = early_means_arr[cluster_mask]
        late_vals = late_means_arr[cluster_mask]

        # Filter out NaN pairs
        valid_mask = ~(np.isnan(early_vals) | np.isnan(late_vals))
        early_valid = early_vals[valid_mask]
        late_valid = late_vals[valid_mask]

        n_embryos = len(early_valid)

        print(f"\n  Cluster {cluster_id}: {n_embryos} embryos")

        if n_embryos >= 3:
            # Pearson correlation
            r, p_value = pearsonr(early_valid, late_valid)

            # Permutation test
            n_permutations = 1000
            permutation_rs = []

            for _ in range(n_permutations):
                late_shuffled = np.random.permutation(late_valid)
                r_perm, _ = pearsonr(early_valid, late_shuffled)
                permutation_rs.append(r_perm)

            permutation_p = np.mean(np.abs(np.array(permutation_rs)) >= np.abs(r))

            # Classification
            if r < -0.3:
                interpretation = "Anti-correlated"
            elif r > 0.3:
                interpretation = "Correlated"
            else:
                interpretation = "Uncorrelated"

            anticorr_results[cluster_id] = {
                'n_embryos': n_embryos,
                'early_mean': np.mean(early_valid),
                'late_mean': np.mean(late_valid),
                'pearson_r': r,
                'p_value': p_value,
                'permutation_p': permutation_p,
                'interpretation': interpretation
            }

            print(f"    Pearson r: {r:.4f} (p={p_value:.4f})")
            print(f"    Permutation p: {permutation_p:.4f}")
            print(f"    Interpretation: {interpretation}")
        else:
            print(f"    Insufficient valid pairs (need ≥3)")

    return anticorr_results


# ============================================================================
# STEP 9: Create Output Dataframe
# ============================================================================

def create_output_dataframe(df, embryo_ids, best_assignments, early_means_arr, late_means_arr):
    """
    Create output dataframe with cluster assignments and statistics.

    Returns
    -------
    output_df : pd.DataFrame
    """
    print(f"\n{'='*80}")
    print("STEP 9: CREATE OUTPUT DATAFRAME")
    print(f"{'='*80}")

    # Get genotype for each embryo
    genotypes = []
    for embryo_id in embryo_ids:
        genotype = df[df['embryo_id'] == embryo_id]['genotype'].iloc[0]
        genotypes.append(genotype)

    output_df = pd.DataFrame({
        'embryo_id': embryo_ids,
        'cluster': best_assignments,
        'early_mean': early_means_arr,
        'late_mean': late_means_arr,
        'genotype': genotypes
    })

    print(f"\n  Output dataframe shape: {output_df.shape}")
    print(f"  Columns: {list(output_df.columns)}")
    print(f"\n  First 5 rows:")
    print(output_df.head())

    return output_df


# ============================================================================
# STEP 10: Generate Plots
# ============================================================================

def generate_plots(clustering_results, distance_matrix, best_k, best_assignments,
                   anticorr_results, early_means_arr, late_means_arr, embryo_ids,
                   interpolated_trajs, embryo_ids_interp, trajectories_df, df, common_grid):
    """
    Generate all visualization plots.
    """
    print(f"\n{'='*80}")
    print("STEP 10: GENERATE PLOTS")
    print(f"{'='*80}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set up color palette for clusters
    unique_clusters = np.unique(best_assignments)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_clusters), 10)))
    cluster_colors = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}

    # Map timepoint indices to HPF values
    hpf_values = common_grid
    early_start_hpf, early_end_hpf = 44, 50
    late_start_hpf, late_end_hpf = 80, 100

    # Find indices corresponding to early/late windows
    early_indices = np.where((hpf_values >= early_start_hpf) & (hpf_values <= early_end_hpf))[0]
    late_indices = np.where((hpf_values >= late_start_hpf) & (hpf_values <= late_end_hpf))[0]

    early_idx_start = early_indices[0] if len(early_indices) > 0 else 0
    early_idx_end = early_indices[-1] if len(early_indices) > 0 else 0
    late_idx_start = late_indices[0] if len(late_indices) > 0 else len(hpf_values) - 1
    late_idx_end = late_indices[-1] if len(late_indices) > 0 else len(hpf_values) - 1

    # Cluster Selection Metrics
    print(f"\n  Generating: Cluster Selection Metrics...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        k_values = sorted(clustering_results.keys())
        silhouettes = [clustering_results[k]['silhouette'] for k in k_values]

        # Silhouette scores (main metric)
        axes[0, 0].plot(k_values, silhouettes, 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 0].axvline(best_k, color='red', linestyle='--', linewidth=2, label=f'Best k={best_k}')
        axes[0, 0].set_xlabel('k', fontsize=11)
        axes[0, 0].set_ylabel('Silhouette Score', fontsize=11)
        axes[0, 0].set_title('Silhouette Scores (Higher is Better)', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Cluster size distribution for best k
        best_cluster_sizes = np.bincount(best_assignments)
        axes[0, 1].bar(np.arange(len(best_cluster_sizes)), best_cluster_sizes, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Cluster ID', fontsize=11)
        axes[0, 1].set_ylabel('Number of Embryos', fontsize=11)
        axes[0, 1].set_title(f'Cluster Sizes (k={best_k})', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Gap Statistic placeholder
        axes[1, 0].text(0.5, 0.5, 'Gap Statistic (TBD)', ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Gap Statistic')

        # Penetrance placeholder
        axes[1, 1].text(0.5, 0.5, 'Penetrance (TBD)', ha='center', va='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Penetrance')

        plt.tight_layout()
        plot_path = PLOTS_DIR / 'cluster_selection_metrics.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {plot_path.name}")
    except Exception as e:
        print(f"    Error generating cluster selection plot: {e}")

    # Anti-Correlation Scatter
    print(f"\n  Generating: Anti-Correlation Scatter...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set2(np.linspace(0, 1, len(np.unique(best_assignments))))

        for cluster_id in np.unique(best_assignments):
            cluster_mask = best_assignments == cluster_id
            x = early_means_arr[cluster_mask]
            y = late_means_arr[cluster_mask]

            ax.scatter(x, y, s=100, alpha=0.6, label=f'Cluster {cluster_id}',
                      color=colors[cluster_id])

        ax.set_xlabel('Early Mean (44-50 hpf)')
        ax.set_ylabel('Late Mean (80-100 hpf)')
        ax.set_title('Early vs Late Curvature by Cluster')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = PLOTS_DIR / 'anticorrelation_scatter.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {plot_path.name}")
    except Exception as e:
        print(f"    Error generating anticorrelation scatter plot: {e}")

    # DTW Distance Matrix
    print(f"\n  Generating: DTW Distance Matrix...")
    try:
        # Sort by cluster
        sorted_indices = np.argsort(best_assignments)
        sorted_distance_matrix = distance_matrix[np.ix_(sorted_indices, sorted_indices)]

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(sorted_distance_matrix, cmap='viridis', aspect='auto')
        ax.set_xlabel('Embryo Index (sorted by cluster)')
        ax.set_ylabel('Embryo Index (sorted by cluster)')
        ax.set_title('DTW Distance Matrix')
        plt.colorbar(im, ax=ax, label='DTW Distance')

        plt.tight_layout()
        plot_path = PLOTS_DIR / 'dtw_distance_matrix.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {plot_path.name}")
    except Exception as e:
        print(f"    Error generating DTW distance matrix plot: {e}")

    # Temporal Trends by Cluster - Generate for all k values
    print(f"\n  Generating: Temporal Trends by Cluster for all k values...")

    for k_value in sorted(clustering_results.keys()):
        try:
            k_assignments = clustering_results[k_value]['assignments']
            k_unique_clusters = np.unique(k_assignments)
            n_clusters = len(k_unique_clusters)

            fig, axes = plt.subplots(1, n_clusters, figsize=(7*n_clusters, 5.5), dpi=100)
            if n_clusters == 1:
                axes = [axes]

            # First pass: collect all y values to determine shared y-axis limits
            all_y_values = []
            for cluster_id in sorted(k_unique_clusters):
                cluster_mask = k_assignments == cluster_id
                cluster_embryo_indices = np.where(cluster_mask)[0]
                cluster_trajs = [interpolated_trajs[i] for i in cluster_embryo_indices]
                max_len = max([len(t) for t in cluster_trajs])

                padded_trajs = []
                for traj in cluster_trajs:
                    padded = np.full(max_len, np.nan)
                    padded[:len(traj)] = traj
                    padded_trajs.append(padded)
                    all_y_values.extend([v for v in traj if not np.isnan(v)])
                padded_trajs = np.array(padded_trajs)

                mean_traj = np.nanmean(padded_trajs, axis=0)
                std_traj = np.nanstd(padded_trajs, axis=0)
                all_y_values.extend(mean_traj + std_traj)
                all_y_values.extend(mean_traj - std_traj)

            y_min = np.nanmin(all_y_values) if all_y_values else 0
            y_max = np.nanmax(all_y_values) if all_y_values else 1
            y_margin = (y_max - y_min) * 0.1
            y_lim = (y_min - y_margin, y_max + y_margin)

            # Second pass: plot with shared y-axis
            for ax_idx, cluster_id in enumerate(sorted(k_unique_clusters)):
                ax = axes[ax_idx]
                cluster_mask = k_assignments == cluster_id
                cluster_embryo_indices = np.where(cluster_mask)[0]

                # Get trajectories for this cluster
                cluster_trajs = [interpolated_trajs[i] for i in cluster_embryo_indices]
                max_len = max([len(t) for t in cluster_trajs])

                # Plot individual trajectories (increased alpha to 0.3)
                for traj in cluster_trajs:
                    ax.plot(np.arange(len(traj)), traj, color='gray', alpha=0.3, linewidth=0.8)

                # Pad to same length for statistics
                padded_trajs = []
                for traj in cluster_trajs:
                    padded = np.full(max_len, np.nan)
                    padded[:len(traj)] = traj
                    padded_trajs.append(padded)
                padded_trajs = np.array(padded_trajs)

                mean_traj = np.nanmean(padded_trajs, axis=0)
                std_traj = np.nanstd(padded_trajs, axis=0)

                # Plot mean and std band
                ax.plot(np.arange(len(mean_traj)), mean_traj, color='black', linewidth=2.8, label='Mean', zorder=3)
                ax.fill_between(np.arange(len(mean_traj)),
                                mean_traj - std_traj,
                                mean_traj + std_traj,
                                color='blue', alpha=0.18, label='±1 SD', zorder=2)

                # Fit linear regression to mean trajectory
                valid_mask = ~np.isnan(mean_traj)
                if np.sum(valid_mask) > 2:
                    x_valid = np.arange(len(mean_traj))[valid_mask]
                    y_valid = mean_traj[valid_mask]
                    slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
                    fit_line = slope * np.arange(len(mean_traj)) + intercept
                    ax.plot(np.arange(len(mean_traj)), fit_line, 'r--', linewidth=2,
                           label=f'Linear fit (R²={r_value**2:.3f})', zorder=2.5)

                # Get anticorr info if available (note: anticorr_results is for best_k, use generic title for other k)
                title = f'Cluster {cluster_id}  •  n={len(cluster_trajs)}'

                ax.set_xlabel('Timepoint Index', fontsize=11)
                ax.set_ylabel('Normalized Baseline Deviation', fontsize=11)
                ax.set_ylim(y_lim)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.25, linestyle='--')
                ax.set_axisbelow(True)
                if ax_idx == 0:
                    ax.legend(fontsize=9, loc='best', framealpha=0.95)

            plt.tight_layout()
            plot_path = PLOTS_DIR / f'temporal_trends_by_cluster_k{k_value}.png'
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {plot_path.name}")
        except Exception as e:
            print(f"    Error generating temporal trends plot for k={k_value}: {e}")

    # Cluster Trajectories Overlay (Two Panels) - Generate for all k values
    print(f"\n  Generating: Cluster Trajectories Overlays for all k values...")

    for k_value in sorted(clustering_results.keys()):
        try:
            k_assignments = clustering_results[k_value]['assignments']
            k_unique_clusters = np.unique(k_assignments)
            k_colors = plt.cm.tab10(np.linspace(0, 1, min(len(k_unique_clusters), 10)))
            k_cluster_colors = {cluster_id: k_colors[i] for i, cluster_id in enumerate(k_unique_clusters)}

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100)

            # Get maximum length across all clusters and collect all y values for shared axis
            all_max_len = 0
            all_y_values = []
            for cluster_id in k_unique_clusters:
                cluster_mask = k_assignments == cluster_id
                cluster_embryo_indices = np.where(cluster_mask)[0]
                cluster_trajs = [interpolated_trajs[i] for i in cluster_embryo_indices]
                all_max_len = max(all_max_len, max([len(t) for t in cluster_trajs]))

                # Collect y values for limits
                for traj in cluster_trajs:
                    all_y_values.extend([v for v in traj if not np.isnan(v)])

                # Compute mean for limits
                padded_trajs = []
                for traj in cluster_trajs:
                    padded = np.full(all_max_len, np.nan)
                    padded[:len(traj)] = traj
                    padded_trajs.append(padded)
                padded_trajs = np.array(padded_trajs)
                mean_traj = np.nanmean(padded_trajs, axis=0)
                std_traj = np.nanstd(padded_trajs, axis=0)
                all_y_values.extend(mean_traj + std_traj)
                all_y_values.extend(mean_traj - std_traj)

            y_min = np.nanmin(all_y_values) if all_y_values else 0
            y_max = np.nanmax(all_y_values) if all_y_values else 1
            y_margin = (y_max - y_min) * 0.1
            y_lim = (y_min - y_margin, y_max + y_margin)

            # ===== LEFT PANEL: Individual Trajectories =====
            ax_left = axes[0]

            for cluster_id in sorted(k_unique_clusters):
                cluster_mask = k_assignments == cluster_id
                cluster_embryo_indices = np.where(cluster_mask)[0]
                cluster_trajs = [interpolated_trajs[i] for i in cluster_embryo_indices]
                color = k_cluster_colors[cluster_id]

                # Plot individual trajectories (higher alpha, cluster color)
                for traj in cluster_trajs:
                    ax_left.plot(np.arange(len(traj)), traj, color=color, alpha=0.3, linewidth=1)

                # Compute mean trajectory for trend line
                padded_trajs = []
                for traj in cluster_trajs:
                    padded = np.full(all_max_len, np.nan)
                    padded[:len(traj)] = traj
                    padded_trajs.append(padded)
                padded_trajs = np.array(padded_trajs)
                mean_traj = np.nanmean(padded_trajs, axis=0)

                # Plot mean trajectory (no CI)
                n_embryos = len(cluster_trajs)
                ax_left.plot(np.arange(len(mean_traj)), mean_traj,
                            color=color, linewidth=2.8, label=f'Cluster {cluster_id} mean (n={n_embryos})', zorder=3)

            ax_left.set_xlabel('Timepoint Index', fontsize=12)
            ax_left.set_ylabel('Normalized Baseline Deviation', fontsize=12)
            ax_left.set_ylim(y_lim)
            ax_left.set_title(f'Individual Trajectories (k={k_value})', fontsize=13, fontweight='bold')
            ax_left.legend(fontsize=10, loc='best', framealpha=0.95)
            ax_left.grid(True, alpha=0.25, linestyle='--')
            ax_left.set_axisbelow(True)

            # ===== RIGHT PANEL: Mean Trajectories with CI & Fit =====
            ax_right = axes[1]

            for cluster_id in sorted(k_unique_clusters):
                cluster_mask = k_assignments == cluster_id
                cluster_embryo_indices = np.where(cluster_mask)[0]
                cluster_trajs = [interpolated_trajs[i] for i in cluster_embryo_indices]
                color = k_cluster_colors[cluster_id]

                # Pad to same length for statistics
                padded_trajs = []
                for traj in cluster_trajs:
                    padded = np.full(all_max_len, np.nan)
                    padded[:len(traj)] = traj
                    padded_trajs.append(padded)
                padded_trajs = np.array(padded_trajs)

                mean_traj = np.nanmean(padded_trajs, axis=0)
                std_traj = np.nanstd(padded_trajs, axis=0)

                # Plot mean trajectory
                n_embryos = len(cluster_trajs)
                ax_right.plot(np.arange(len(mean_traj)), mean_traj,
                             color=color, linewidth=2.8, label=f'Cluster {cluster_id} (n={n_embryos})', zorder=3)

                # Plot CI band
                ax_right.fill_between(np.arange(len(mean_traj)),
                                     mean_traj - std_traj,
                                     mean_traj + std_traj,
                                     color=color, alpha=0.25, zorder=2)

                # Fit linear regression to mean trajectory
                valid_mask = ~np.isnan(mean_traj)
                if np.sum(valid_mask) > 2:
                    x_valid = np.arange(len(mean_traj))[valid_mask]
                    y_valid = mean_traj[valid_mask]
                    slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
                    fit_line = slope * np.arange(len(mean_traj)) + intercept
                    ax_right.plot(np.arange(len(mean_traj)), fit_line, '--', color=color,
                                 linewidth=1.8, alpha=0.8, zorder=2.5)

            ax_right.set_xlabel('Timepoint Index', fontsize=12)
            ax_right.set_ylabel('Normalized Baseline Deviation', fontsize=12)
            ax_right.set_ylim(y_lim)
            ax_right.set_title(f'Mean Trajectories with ±1 SD & Linear Fit (k={k_value})', fontsize=13, fontweight='bold')
            ax_right.legend(fontsize=11, loc='best', framealpha=0.95)
            ax_right.grid(True, alpha=0.25, linestyle='--')
            ax_right.set_axisbelow(True)

            plt.tight_layout()
            plot_path = PLOTS_DIR / f'cluster_trajectory_overlay_k{k_value}.png'
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {plot_path.name}")
        except Exception as e:
            print(f"    Error generating trajectory overlay plot for k={k_value}: {e}")

    print(f"\n  Plots generated successfully")


# ============================================================================
# STEP 11: Generate Tables
# ============================================================================

def generate_tables(output_df, anticorr_results, best_assignments, clustering_results,
                   distance_matrix, embryo_ids_interp, early_means_arr, late_means_arr, df):
    """
    Generate output CSV tables and save distance matrix.
    """
    print(f"\n{'='*80}")
    print("STEP 11: GENERATE TABLES")
    print(f"{'='*80}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Table 4: Cluster Characteristics
    print(f"\n  Generating Table 4: Cluster Characteristics...")
    try:
        table_4_data = []

        for cluster_id in np.unique(best_assignments):
            cluster_df = output_df[output_df['cluster'] == cluster_id]
            n_embryos = len(cluster_df)
            early_mean = cluster_df['early_mean'].mean()
            late_mean = cluster_df['late_mean'].mean()

            genotype_counts = cluster_df['genotype'].value_counts()
            pct_wt = 100 * genotype_counts.get('cep290_wildtype', 0) / n_embryos
            pct_het = 100 * genotype_counts.get('cep290_heterozygous', 0) / n_embryos
            pct_homo = 100 * genotype_counts.get('cep290_homozygous', 0) / n_embryos

            table_4_data.append({
                'Cluster': cluster_id,
                'n_embryos': n_embryos,
                '%_WT': pct_wt,
                '%_Het': pct_het,
                '%_Homo': pct_homo,
                'Early_mean': early_mean,
                'Late_mean': late_mean
            })

        table_4_df = pd.DataFrame(table_4_data)
        table_4_path = TABLES_DIR / 'table_4_cluster_characteristics.csv'
        table_4_df.to_csv(table_4_path, index=False)
        print(f"    Saved: {table_4_path.name}")
    except Exception as e:
        print(f"    Error generating Table 4: {e}")

    # Table 5: Anti-Correlation Evidence
    print(f"\n  Generating Table 5: Anti-Correlation Evidence...")
    try:
        table_5_data = []

        for cluster_id in sorted(anticorr_results.keys()):
            result = anticorr_results[cluster_id]
            table_5_data.append({
                'Cluster': cluster_id,
                'n_embryos': result['n_embryos'],
                'Early_mean': result['early_mean'],
                'Late_mean': result['late_mean'],
                'Pearson_r': result['pearson_r'],
                'P_value': result['p_value'],
                'Permutation_p_value': result['permutation_p'],
                'Interpretation': result['interpretation']
            })

        table_5_df = pd.DataFrame(table_5_data)
        table_5_path = TABLES_DIR / 'table_5_anticorrelation_evidence.csv'
        table_5_df.to_csv(table_5_path, index=False)
        print(f"    Saved: {table_5_path.name}")
    except Exception as e:
        print(f"    Error generating Table 5: {e}")

    # Table 6: Embryo-Cluster Assignments
    print(f"\n  Generating Table 6: Embryo-Cluster Assignments...")
    try:
        table_6_df = output_df[['embryo_id', 'genotype', 'cluster', 'early_mean', 'late_mean']].copy()
        table_6_df.columns = ['embryo_id', 'genotype', 'cluster_assignment', 'early_curvature', 'late_curvature']
        table_6_path = TABLES_DIR / 'table_6_embryo_cluster_assignments.csv'
        table_6_df.to_csv(table_6_path, index=False)
        print(f"    Saved: {table_6_path.name}")
    except Exception as e:
        print(f"    Error generating Table 6: {e}")

    # Cluster Assignments for Each k Value
    print(f"\n  Generating: Cluster Assignments for all k values...")
    try:
        for k_value in sorted(clustering_results.keys()):
            k_assignments = clustering_results[k_value]['assignments']

            # Create dataframe with assignments and characteristics
            k_df = pd.DataFrame({
                'embryo_id': embryo_ids_interp,
                'cluster': k_assignments,
                'genotype': [df[df['embryo_id'] == e_id]['genotype'].iloc[0] if e_id in df['embryo_id'].values else 'unknown'
                            for e_id in embryo_ids_interp],
                'early_mean': early_means_arr,
                'late_mean': late_means_arr
            })

            k_path = TABLES_DIR / f'cluster_assignments_k{k_value}.csv'
            k_df.to_csv(k_path, index=False)
            print(f"    Saved: {k_path.name}")
    except Exception as e:
        print(f"    Error generating cluster assignments for k values: {e}")

    # Save Raw DTW Distance Matrix
    print(f"\n  Generating: Raw DTW Distance Matrix...")
    try:
        # Create dataframe with embryo IDs as row/column labels
        dist_df = pd.DataFrame(
            distance_matrix,
            index=embryo_ids_interp,
            columns=embryo_ids_interp
        )
        dist_path = TABLES_DIR / 'dtw_distance_matrix.csv'
        dist_df.to_csv(dist_path)
        print(f"    Saved: {dist_path.name}")
    except Exception as e:
        print(f"    Error saving DTW distance matrix: {e}")

    # Save Embryo ID Mapping (Index to Embryo ID)
    print(f"\n  Generating: Embryo ID Mapping...")
    try:
        mapping_df = pd.DataFrame({
            'index': np.arange(len(embryo_ids_interp)),
            'embryo_id': embryo_ids_interp
        })
        mapping_path = TABLES_DIR / 'embryo_id_mapping.csv'
        mapping_df.to_csv(mapping_path, index=False)
        print(f"    Saved: {mapping_path.name}")
    except Exception as e:
        print(f"    Error saving embryo ID mapping: {e}")


# ============================================================================
# STEP 12: Print Summary
# ============================================================================

def print_summary(output_df, results_dir):
    """
    Print final completion summary.
    """
    print(f"\n{'='*80}")
    print("STEP 12: SUMMARY")
    print(f"{'='*80}")

    print(f"\n  ✓ Analysis complete")
    print(f"\n  Output directory: {results_dir}")
    print(f"  - Plots: {results_dir / 'plots'}")
    print(f"  - Tables: {results_dir / 'tables'}")

    print(f"\n  Output dataframe:")
    print(f"    Shape: {output_df.shape}")
    print(f"    Columns: {list(output_df.columns)}")
    print(f"\n  Cluster distribution:")
    print(output_df['cluster'].value_counts().sort_index())


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("07b DTW CLUSTERING ANALYSIS")
    print("="*80)

    if TEST_MODE:
        print(f"\n  TEST MODE ENABLED")
        print(f"  Running on {N_TEST_EMBRYOS} embryos for validation\n")

    try:
        # Load data
        print(f"\nLoading data...")
        df, metadata = get_analysis_dataframe()

        # Test mode: sample embryos
        if TEST_MODE:
            unique_embryos = df[df['genotype'] == GENOTYPE_FILTER]['embryo_id'].unique()
            selected_embryos = np.random.choice(unique_embryos, size=min(N_TEST_EMBRYOS, len(unique_embryos)), replace=False)
            df = df[df['embryo_id'].isin(selected_embryos)]
            print(f"\nTEST MODE: Selected {len(selected_embryos)} embryos")

        # STEP 1: Extract trajectories
        trajectories, embryo_ids, trajectories_df = extract_trajectories(
            df, GENOTYPE_FILTER, METRIC_NAME
        )

        # STEP 2: Interpolate missing data
        trajectories_df = interpolate_trajectories(trajectories_df)

        # STEP 3: Interpolate to common grid
        interpolated_trajs, embryo_ids_interp, orig_lengths, common_grid = interpolate_to_common_grid(
            trajectories_df, grid_step=INTERPOLATION_GRID_STEP
        )

        # STEP 4: Compute DTW distance matrix
        distance_matrix = compute_dtw_distance_matrix(interpolated_trajs, window=DTW_WINDOW)

        # STEP 5: Perform clustering
        clustering_results = perform_clustering(distance_matrix, k_values=CLUSTER_K_VALUES)

        # STEP 6: Select best k
        best_k, best_assignments = select_best_k(clustering_results)

        # STEP 7: Extract early/late means
        early_means_arr, late_means_arr = extract_early_late_means(
            trajectories_df, embryo_ids_interp, EARLY_WINDOW, LATE_WINDOW
        )

        # STEP 8: Test anti-correlation
        anticorr_results = test_anticorrelation(
            best_assignments, early_means_arr, late_means_arr, embryo_ids_interp
        )

        # STEP 9: Create output dataframe
        output_df = create_output_dataframe(
            df, embryo_ids_interp, best_assignments, early_means_arr, late_means_arr
        )

        # STEP 10: Generate plots
        generate_plots(clustering_results, distance_matrix, best_k, best_assignments,
                      anticorr_results, early_means_arr, late_means_arr, embryo_ids_interp,
                      interpolated_trajs, embryo_ids_interp, trajectories_df, df, common_grid)

        # STEP 11: Generate tables
        generate_tables(output_df, anticorr_results, best_assignments, clustering_results,
                       distance_matrix, embryo_ids_interp, early_means_arr, late_means_arr, df)

        # STEP 12: Print summary
        print_summary(output_df, RESULTS_DIR)

        print(f"\n{'='*80}")
        print("SUCCESS")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
