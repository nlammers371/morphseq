#!/usr/bin/env python3
"""
Compare Direct vs Consensus Clustering Methods

Tests hypothesis that consensus clustering (from bootstrapped co-occurrence)
produces more stable, confident clusters than direct clustering.

Methods Compared:
1. K-medoids on DTW distance (direct, baseline)
2. Hierarchical on DTW distance (direct)
3. K-medoids on co-occurrence distance (consensus)
4. Hierarchical on co-occurrence (consensus)

For each method and k value, computes:
- Silhouette scores (cluster quality)
- Bootstrap stability (ARI under resampling)
- Membership classification (core/uncertain/outlier)
- Cross-method agreement (ARI between methods)

Generates (organized by plot type):
SECTION A: Co-association matrices (28 plots: 4 methods × 7 k values)
SECTION B: Temporal trends with membership coloring (28 plots: 4 methods × 7 k values)
SECTION C: Cluster trajectory overlays (28 plots: 4 methods × 7 k values)
SECTION D: Membership vs K plots (4 plots: 1 per method)

Total: 88 plots

Usage
-----
python compare_clustering_methods.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import (
    OUTPUT_DIR, K_VALUES, N_BOOTSTRAP, BOOTSTRAP_FRAC,
    CORE_THRESHOLD, OUTLIER_THRESHOLD, RANDOM_SEED, VERBOSE_OUTPUT
)

# Import pipeline modules (handling hyphenated filenames)
import importlib.util

def load_module(name, filepath):
    """Load a module from a file path (handles hyphens in filenames)."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules
cluster_module = load_module("cluster_module", "cluster-module.py")
select_k_module = load_module("select_k_module", "select-k-simple.py")
membership_module = load_module("membership_module", "membership-module.py")
io_module = load_module("io_module", "io-module.py")
dtw_precompute_module = load_module("dtw_precompute", "0_dtw_precompute.py")

# Extract functions
from sklearn_extra.cluster import KMedoids
run_bootstrap = cluster_module.run_bootstrap
plot_coassoc_matrix = cluster_module.plot_coassoc_matrix
consensus_clustering = select_k_module.consensus_clustering
analyze_membership = membership_module.analyze_membership
plot_membership_vs_k = membership_module.plot_membership_vs_k
load_data = io_module.load_data
save_data = io_module.save_data
save_plot = io_module.save_plot
precompute_dtw = dtw_precompute_module.precompute_dtw

# Import plot utilities
from plot_utils import (
    plot_temporal_trends_by_cluster,
    plot_cluster_trajectory_overlay,
    plot_temporal_trends_with_membership
)
from src.analyze.dtw_time_trend_analysis.trajectory_utils import pad_trajectories_for_plotting

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BOOTSTRAP FUNCTION FOR HIERARCHICAL CLUSTERING
# ============================================================================

def run_bootstrap_hierarchical(D, k, n_bootstrap=100, frac=0.8, random_state=42, verbose=False):
    """
    Bootstrap resampling using HIERARCHICAL clustering (not k-medoids).

    Parallel to run_bootstrap() but uses hierarchical clustering for each iteration.
    """
    n = len(D)
    C = np.zeros((n, n))  # Co-association matrix
    counts = np.zeros((n, n))  # Count matrix
    ari_scores = []
    silhouette_scores = []

    np.random.seed(random_state)

    for iteration in range(n_bootstrap):
        # Sample embryos
        sample_size = int(n * frac)
        sampled_indices = np.random.choice(n, size=sample_size, replace=False)

        # Get submatrix for this sample
        D_sample = D[np.ix_(sampled_indices, sampled_indices)]

        # Cluster with hierarchical (not k-medoids)
        hc = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels_sample = hc.fit_predict(D_sample)

        # Track co-clustering
        for i, global_i in enumerate(sampled_indices):
            for j, global_j in enumerate(sampled_indices):
                counts[global_i, global_j] += 1
                if labels_sample[i] == labels_sample[j]:
                    C[global_i, global_j] += 1

    # Normalize
    C = np.divide(C, counts, where=counts > 0, out=np.zeros_like(C, dtype=float))
    np.fill_diagonal(C, 1.0)

    return {
        'coassoc': C,
        'ari_scores': np.array(ari_scores) if ari_scores else np.ones(n_bootstrap) * 0.5,
        'silhouette_scores': np.array(silhouette_scores) if silhouette_scores else np.ones(n_bootstrap) * 0.5
    }


# ============================================================================
# CLUSTERING METHODS
# ============================================================================

def cluster_kmedoids_dtw(D, k, random_state=42):
    """K-medoids clustering on DTW distance matrix."""
    km = KMedoids(n_clusters=k, metric='precomputed', random_state=random_state)
    labels = km.fit_predict(D)
    return labels, km.medoid_indices_

def cluster_hierarchical_dtw(D, k, linkage_method='average'):
    """Hierarchical clustering on DTW distance matrix."""
    hc = AgglomerativeClustering(n_clusters=k, metric='precomputed',
                                 linkage=linkage_method)
    labels = hc.fit_predict(D)
    return labels, None

def cluster_kmedoids_cooccurrence(C, k, random_state=42):
    """K-medoids clustering on co-occurrence distance (1-C)."""
    D_C = 1 - C
    km = KMedoids(n_clusters=k, metric='precomputed', random_state=random_state)
    labels = km.fit_predict(D_C)
    return labels, km.medoid_indices_

def cluster_hierarchical_cooccurrence(C, k):
    """Hierarchical clustering on co-occurrence matrix (consensus approach)."""
    labels = consensus_clustering(C, k)
    return labels, None


# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

def compare_methods_for_k(D, k, n_bootstrap=100, frac=0.8, core_threshold=CORE_THRESHOLD,
                          verbose=True):
    """
    Compare all 4 clustering methods for a single k value.

    Each method gets its own bootstrap resampling with the same clustering algorithm
    to ensure fair comparison of stability.

    Returns results dict with methods as keys, containing:
    - labels: cluster assignments
    - silhouette: silhouette score
    - membership: core/uncertain/outlier classification
    - bootstrap_ari: mean ARI across bootstrap iterations
    - coassoc: co-occurrence matrix for this method
    - medoids: (for k-medoids methods)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Comparing clustering methods for k={k}")
        print(f"{'='*80}")

    results = {}

    # ===== BOOTSTRAP FOR K-MEDOIDS METHODS =====
    if verbose:
        print(f"\nRunning bootstrap for K-MEDOIDS ({n_bootstrap} iterations, {frac*100:.0f}% sampling)...")

    boot_kmedoids = run_bootstrap(D, k, n_bootstrap=n_bootstrap, frac=frac, verbose=False)
    C_kmedoids = boot_kmedoids['coassoc']
    bootstrap_ari_kmedoids = boot_kmedoids['ari_scores']

    if verbose:
        print(f"  ✓ K-medoids co-occurrence matrix computed")
        print(f"  ✓ Bootstrap mean ARI: {np.mean(bootstrap_ari_kmedoids):.4f}")

    # ===== BOOTSTRAP FOR HIERARCHICAL METHODS =====
    if verbose:
        print(f"\nRunning bootstrap for HIERARCHICAL ({n_bootstrap} iterations, {frac*100:.0f}% sampling)...")

    boot_hierarchical = run_bootstrap_hierarchical(D, k, n_bootstrap=n_bootstrap, frac=frac, verbose=False)
    C_hierarchical = boot_hierarchical['coassoc']
    bootstrap_ari_hierarchical = boot_hierarchical['ari_scores']

    if verbose:
        print(f"  ✓ Hierarchical co-occurrence matrix computed")
        print(f"  ✓ Bootstrap mean ARI: {np.mean(bootstrap_ari_hierarchical):.4f}")

    # ===== METHOD 1: K-medoids on DTW =====
    if verbose:
        print(f"\nMethod 1: K-medoids on DTW distance...")

    labels_km_dtw, medoids_km_dtw = cluster_kmedoids_dtw(D, k)
    sil_km_dtw = silhouette_score(D, labels_km_dtw, metric='precomputed')
    mem_km_dtw = analyze_membership(D, labels_km_dtw, C_kmedoids, core_thresh=core_threshold)

    results['kmedoids_dtw'] = {
        'labels': labels_km_dtw,
        'silhouette': sil_km_dtw,
        'membership': mem_km_dtw,
        'medoids': medoids_km_dtw,
        'n_core': mem_km_dtw['summary']['n_core'],
        'n_uncertain': mem_km_dtw['summary']['n_uncertain'],
        'n_outlier': mem_km_dtw['summary']['n_outlier'],
        'bootstrap_ari': np.mean(bootstrap_ari_kmedoids),
        'coassoc': C_kmedoids
    }

    if verbose:
        print(f"  Silhouette: {sil_km_dtw:.4f}")
        print(f"  Core/Uncertain/Outlier: {results['kmedoids_dtw']['n_core']}/{results['kmedoids_dtw']['n_uncertain']}/{results['kmedoids_dtw']['n_outlier']}")

    # ===== METHOD 2: Hierarchical on DTW =====
    if verbose:
        print(f"\nMethod 2: Hierarchical on DTW distance...")

    labels_hc_dtw, _ = cluster_hierarchical_dtw(D, k)
    sil_hc_dtw = silhouette_score(D, labels_hc_dtw, metric='precomputed')
    mem_hc_dtw = analyze_membership(D, labels_hc_dtw, C_hierarchical, core_thresh=core_threshold)

    results['hierarchical_dtw'] = {
        'labels': labels_hc_dtw,
        'silhouette': sil_hc_dtw,
        'membership': mem_hc_dtw,
        'medoids': None,
        'n_core': mem_hc_dtw['summary']['n_core'],
        'n_uncertain': mem_hc_dtw['summary']['n_uncertain'],
        'n_outlier': mem_hc_dtw['summary']['n_outlier'],
        'bootstrap_ari': np.mean(bootstrap_ari_hierarchical),
        'coassoc': C_hierarchical
    }

    if verbose:
        print(f"  Silhouette: {sil_hc_dtw:.4f}")
        print(f"  Core/Uncertain/Outlier: {results['hierarchical_dtw']['n_core']}/{results['hierarchical_dtw']['n_uncertain']}/{results['hierarchical_dtw']['n_outlier']}")

    # ===== METHOD 3: K-medoids on co-occurrence =====
    if verbose:
        print(f"\nMethod 3: K-medoids on co-occurrence distance...")

    labels_km_c, medoids_km_c = cluster_kmedoids_cooccurrence(C_kmedoids, k)
    sil_km_c = silhouette_score(D, labels_km_c, metric='precomputed')
    mem_km_c = analyze_membership(D, labels_km_c, C_kmedoids, core_thresh=core_threshold)

    results['kmedoids_consensus'] = {
        'labels': labels_km_c,
        'silhouette': sil_km_c,
        'membership': mem_km_c,
        'medoids': medoids_km_c,
        'n_core': mem_km_c['summary']['n_core'],
        'n_uncertain': mem_km_c['summary']['n_uncertain'],
        'n_outlier': mem_km_c['summary']['n_outlier'],
        'bootstrap_ari': np.mean(bootstrap_ari_kmedoids),
        'coassoc': C_kmedoids
    }

    if verbose:
        print(f"  Silhouette: {sil_km_c:.4f}")
        print(f"  Core/Uncertain/Outlier: {results['kmedoids_consensus']['n_core']}/{results['kmedoids_consensus']['n_uncertain']}/{results['kmedoids_consensus']['n_outlier']}")

    # ===== METHOD 4: Hierarchical on co-occurrence (CONSENSUS) =====
    if verbose:
        print(f"\nMethod 4: Hierarchical on co-occurrence (consensus)...")

    labels_consensus, _ = cluster_hierarchical_cooccurrence(C_hierarchical, k)
    sil_consensus = silhouette_score(D, labels_consensus, metric='precomputed')
    mem_consensus = analyze_membership(D, labels_consensus, C_hierarchical, core_thresh=core_threshold)

    results['hierarchical_consensus'] = {
        'labels': labels_consensus,
        'silhouette': sil_consensus,
        'membership': mem_consensus,
        'medoids': None,
        'n_core': mem_consensus['summary']['n_core'],
        'n_uncertain': mem_consensus['summary']['n_uncertain'],
        'n_outlier': mem_consensus['summary']['n_outlier'],
        'bootstrap_ari': np.mean(bootstrap_ari_hierarchical),
        'coassoc': C_hierarchical
    }

    if verbose:
        print(f"  Silhouette: {sil_consensus:.4f}")
        print(f"  Core/Uncertain/Outlier: {results['hierarchical_consensus']['n_core']}/{results['hierarchical_consensus']['n_uncertain']}/{results['hierarchical_consensus']['n_outlier']}")

    # ===== COMPUTE CROSS-METHOD ARI =====
    if verbose:
        print(f"\nComputing cross-method agreement (ARI)...")

    methods = list(results.keys())
    ari_matrix = np.zeros((len(methods), len(methods)))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                ari_matrix[i, j] = 1.0
            else:
                ari_matrix[i, j] = adjusted_rand_score(
                    results[method1]['labels'],
                    results[method2]['labels']
                )

    if verbose:
        print(f"  ARI Matrix:")
        for i, method in enumerate(methods):
            print(f"    {method}: {ari_matrix[i, :].round(3)}")

    results['ari_matrix'] = ari_matrix
    results['methods'] = methods

    return results


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_method_results_for_k(precomp_results, method_results, k, method_name,
                              verbose=True):
    """
    Generate temporal trends plot for a specific method and k value,
    colored by membership (core/uncertain/outlier).
    """
    from plot_utils import plot_temporal_trends_with_membership

    trajectories = precomp_results['trajectories']
    common_grid = precomp_results['common_grid']
    df_long = precomp_results['df_long']
    embryo_ids = precomp_results['embryo_ids']

    labels = method_results['labels']
    membership = method_results['membership']
    membership_classification = membership['classification']

    # Group trajectories by cluster
    trajectories_by_cluster = {}
    cluster_indices_map = {}

    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_trajs = [trajectories[i] for i in cluster_indices]
        cluster_embryo_ids = [embryo_ids[i] for i in cluster_indices]

        # Pad for plotting
        cluster_trajs_padded = pad_trajectories_for_plotting(
            cluster_trajs, common_grid, df_long, cluster_embryo_ids, verbose=False
        )

        trajectories_by_cluster[cluster_id] = cluster_trajs_padded
        cluster_indices_map[cluster_id] = cluster_indices

    # Generate plot with membership coloring
    fig = plot_temporal_trends_with_membership(
        trajectories_by_cluster,
        common_grid,
        k=k,
        membership_classification=membership_classification,
        cluster_indices_map=cluster_indices_map,
        title=f"Temporal Trends by Cluster - {method_name.replace('_', ' ').title()} (k={k})"
    )

    return fig


def main():
    """Main entry point for method comparison."""
    if VERBOSE_OUTPUT:
        print("\n" + "="*80)
        print("CLUSTERING METHOD COMPARISON")
        print("="*80)
        print(f"\nMethods to compare:")
        print(f"  1. K-medoids on DTW distance")
        print(f"  2. Hierarchical on DTW distance")
        print(f"  3. K-medoids on co-occurrence distance")
        print(f"  4. Hierarchical on co-occurrence (consensus)")
        print(f"\nK values: {K_VALUES}")
        print(f"Bootstrap: {N_BOOTSTRAP} iterations, {BOOTSTRAP_FRAC*100:.0f}% sampling")

    # Load data
    if VERBOSE_OUTPUT:
        print(f"\nLoading data...")

    try:
        D = load_data(0, 'distance_matrix', OUTPUT_DIR)
    except FileNotFoundError:
        if VERBOSE_OUTPUT:
            print(f"  DTW matrix not found, precomputing...")
        precomp = precompute_dtw(verbose=VERBOSE_OUTPUT)
        D = precomp['distance_matrix']

    try:
        precomp_results = load_data(0, 'precompute', OUTPUT_DIR)
    except:
        precomp_results = precompute_dtw(verbose=False)

    if VERBOSE_OUTPUT:
        print(f"  ✓ DTW matrix: {D.shape}")
        print(f"  ✓ Trajectories: {len(precomp_results['trajectories'])}")

    # Compare methods for each k
    all_results = {}

    for k in K_VALUES:
        results_k = compare_methods_for_k(
            D, k,
            n_bootstrap=N_BOOTSTRAP,
            frac=BOOTSTRAP_FRAC,
            core_threshold=CORE_THRESHOLD,
            verbose=VERBOSE_OUTPUT
        )

        all_results[k] = results_k

    # ============================================================================
    # SECTION A: CO-ASSOCIATION MATRICES
    # ============================================================================
    if VERBOSE_OUTPUT:
        print(f"\n{'='*80}")
        print(f"SECTION A: Generating Co-association Matrix Heatmaps")
        print(f"{'='*80}")

    for k in K_VALUES:
        results_k = all_results[k]
        if VERBOSE_OUTPUT:
            print(f"\n  k={k}:")

        for method_name, method_result in results_k.items():
            if method_name not in ['ari_matrix', 'methods']:
                try:
                    C = method_result['coassoc']
                    labels = method_result['labels']

                    fig = plot_coassoc_matrix(
                        C, labels=labels, k=k,
                        title=f"Co-association Matrix - {method_name.replace('_', ' ').title()} (k={k})"
                    )

                    plot_name = f"coassoc_{method_name}_k{k}"
                    save_plot(7, plot_name, fig, OUTPUT_DIR)
                    plt.close(fig)

                    if VERBOSE_OUTPUT:
                        print(f"    ✓ Saved {plot_name}")
                except Exception as e:
                    if VERBOSE_OUTPUT:
                        print(f"    Warning: Could not generate co-assoc plot for {method_name}: {e}")

    # ============================================================================
    # SECTION B: TEMPORAL TRENDS WITH MEMBERSHIP COLORING
    # ============================================================================
    if VERBOSE_OUTPUT:
        print(f"\n{'='*80}")
        print(f"SECTION B: Generating Temporal Trends with Membership Coloring")
        print(f"{'='*80}")

    for k in K_VALUES:
        results_k = all_results[k]
        if VERBOSE_OUTPUT:
            print(f"\n  k={k}:")

        for method_name, method_result in results_k.items():
            if method_name not in ['ari_matrix', 'methods', 'coassoc']:
                try:
                    fig = plot_method_results_for_k(
                        precomp_results, method_result, k, method_name, verbose=False
                    )

                    plot_name = f"temporal_trends_{method_name}_k{k}"
                    save_plot(7, plot_name, fig, OUTPUT_DIR)
                    plt.close(fig)

                    if VERBOSE_OUTPUT:
                        print(f"    ✓ Saved {plot_name}")
                except Exception as e:
                    if VERBOSE_OUTPUT:
                        print(f"    Warning: Could not generate plot for {method_name}: {e}")

    # ============================================================================
    # SECTION C: CLUSTER TRAJECTORY OVERLAYS
    # ============================================================================
    if VERBOSE_OUTPUT:
        print(f"\n{'='*80}")
        print(f"SECTION C: Generating Cluster Trajectory Overlay Plots")
        print(f"{'='*80}")

    for k in K_VALUES:
        results_k = all_results[k]
        if VERBOSE_OUTPUT:
            print(f"\n  k={k}:")

        for method_name, method_result in results_k.items():
            if method_name not in ['ari_matrix', 'methods', 'coassoc']:
                try:
                    # Prepare trajectories by cluster
                    trajectories = precomp_results['trajectories']
                    common_grid = precomp_results['common_grid']
                    df_long = precomp_results['df_long']
                    embryo_ids = precomp_results['embryo_ids']
                    labels = method_result['labels']

                    trajectories_by_cluster = {}
                    for cluster_id in np.unique(labels):
                        cluster_indices = np.where(labels == cluster_id)[0]
                        cluster_trajs = [trajectories[i] for i in cluster_indices]
                        cluster_embryo_ids = [embryo_ids[i] for i in cluster_indices]

                        # Pad for plotting
                        cluster_trajs_padded = pad_trajectories_for_plotting(
                            cluster_trajs, common_grid, df_long, cluster_embryo_ids, verbose=False
                        )
                        trajectories_by_cluster[cluster_id] = cluster_trajs_padded

                    # Generate overlay plot
                    fig = plot_cluster_trajectory_overlay(
                        trajectories_by_cluster,
                        common_grid,
                        k=k,
                        title=f"Cluster Trajectory Overlay - {method_name.replace('_', ' ').title()} (k={k})"
                    )

                    plot_name = f"cluster_overlay_{method_name}_k{k}"
                    save_plot(7, plot_name, fig, OUTPUT_DIR)
                    plt.close(fig)

                    if VERBOSE_OUTPUT:
                        print(f"    ✓ Saved {plot_name}")
                except Exception as e:
                    if VERBOSE_OUTPUT:
                        print(f"    Warning: Could not generate overlay plot for {method_name}: {e}")

    # ============================================================================
    # SECTION D: MEMBERSHIP VS K PLOTS (one per method)
    # ============================================================================
    if VERBOSE_OUTPUT:
        print(f"\n{'='*80}")
        print(f"SECTION D: Generating Membership vs K Plots")
        print(f"{'='*80}")

    # Organize membership data by method across all k values
    methods = ['kmedoids_dtw', 'hierarchical_dtw', 'kmedoids_consensus', 'hierarchical_consensus']

    for method_name in methods:
        try:
            # Collect membership data across all k values for this method
            all_k_membership = {}
            for k in K_VALUES:
                method_result = all_results[k][method_name]
                all_k_membership[k] = method_result['membership']

            # Generate membership vs k plot
            fig = plot_membership_vs_k(
                all_k_membership,
                best_k=None,
                title=f"Membership Distribution Across K Values - {method_name.replace('_', ' ').title()}"
            )

            plot_name = f"membership_vs_k_{method_name}"
            save_plot(7, plot_name, fig, OUTPUT_DIR)
            plt.close(fig)

            if VERBOSE_OUTPUT:
                print(f"  ✓ Saved {plot_name}")
        except Exception as e:
            if VERBOSE_OUTPUT:
                print(f"  Warning: Could not generate membership vs k plot for {method_name}: {e}")

    # Save results
    if VERBOSE_OUTPUT:
        print(f"\nSaving results...")

    save_data(7, 'method_comparison_all_k', all_results, OUTPUT_DIR)

    # Print summary
    if VERBOSE_OUTPUT:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        for k in K_VALUES:
            print(f"\nk={k}:")
            results_k = all_results[k]

            # Print silhouette scores
            print(f"  Silhouette scores:")
            for method in results_k['methods']:
                sil = results_k[method]['silhouette']
                print(f"    {method:30s}: {sil:.4f}")

            # Print core membership
            print(f"  Core membership %:")
            for method in results_k['methods']:
                n_core = results_k[method]['n_core']
                n_total = n_core + results_k[method]['n_uncertain'] + results_k[method]['n_outlier']
                pct = 100 * n_core / n_total if n_total > 0 else 0
                print(f"    {method:30s}: {pct:5.1f}% ({n_core}/{n_total})")

    return all_results


if __name__ == '__main__':
    results = main()
