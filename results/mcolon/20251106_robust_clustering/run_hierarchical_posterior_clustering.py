#!/usr/bin/env python3
"""
Multi-Experiment Hierarchical POSTERIOR Clustering

Runs hierarchical clustering with bootstrap-based posterior probability analysis.
Uses label alignment (Hungarian algorithm) to compute per-embryo assignment posteriors.

Key Changes from Original:
- Stores bootstrap LABELS (not co-association matrix)
- Computes assignment posteriors p_i(c) via Hungarian alignment
- Classifies embryos using 2D gating (max_p + log_odds_gap)
- Generates posterior-weighted trajectory plots

Output Structure:
-----------------
output/
├── {experiment_id}/
│   ├── {genotype}/
│   │   ├── posterior_heatmaps/
│   │   ├── posterior_scatters/
│   │   ├── temporal_trends_posterior/  # Continuous alpha
│   │   ├── temporal_trends_category/   # Category colors
│   │   └── membership_vs_k.png
│   └── data/
│       └── posteriors_{genotype}_k{k}.pkl

Usage:
------
python run_hierarchical_posterior_clustering.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import (
    OUTPUT_DIR, EXPERIMENTS, METRIC_NAME, MIN_TIMEPOINTS,
    DTW_WINDOW, GRID_STEP, K_VALUES, N_BOOTSTRAP, BOOTSTRAP_FRAC,
    CORE_THRESHOLD, OUTLIER_THRESHOLD, RANDOM_SEED, VERBOSE_OUTPUT,
    BUILD_DIR
)

# Import functions from previous analysis
sys.path.insert(0, str(Path(__file__).parent.parent / "20251103_DTW_analysis"))
import importlib.util

def load_module(name, filepath):
    """Load a module from a file path (handles hyphens in filenames)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent.parent / "20251103_DTW_analysis" / filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load required modules
cluster_module = load_module("cluster_module", "cluster-module.py")
select_k_module = load_module("select_k_module", "select-k-simple.py")
membership_module = load_module("membership_module", "membership-module.py")

# Extract functions
consensus_clustering = select_k_module.consensus_clustering
analyze_membership = membership_module.analyze_membership
plot_membership_vs_k = membership_module.plot_membership_vs_k
plot_coassoc_matrix = cluster_module.plot_coassoc_matrix

# Import src utilities for trajectory processing and DTW
from src.analyze.dtw_time_trend_analysis import (
    extract_trajectories,
    interpolate_to_common_grid,
    compute_dtw_distance_matrix,
)
from src.analyze.dtw_time_trend_analysis.trajectory_utils import pad_trajectories_for_plotting

# Import plot utilities
from plot_utils import (
    plot_temporal_trends_with_membership,
    plot_cluster_trajectory_overlay
)
from src.analyze.utils.plotting import plot_embryos_metric_over_time

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MULTI-EXPERIMENT DATA LOADING
# ============================================================================

def load_experiment_data(experiment_id, curv_dir, meta_dir, verbose=True):
    """
    Load and merge curvature metrics with metadata for a single experiment.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., '20250305')
    curv_dir : Path
        Directory containing curvature CSV files
    meta_dir : Path
        Directory containing metadata CSV files
    verbose : bool
        Print loading progress

    Returns
    -------
    pd.DataFrame
        Merged dataframe with curvature metrics, metadata, and normalized metrics
    """
    # Load curvature metrics
    curv_file = curv_dir / f"curvature_metrics_{experiment_id}.csv"
    if not curv_file.exists():
        raise FileNotFoundError(f"Curvature file not found: {curv_file}")

    curv_df = pd.read_csv(curv_file)

    # Load metadata
    meta_file = meta_dir / f"df03_final_output_with_latents_{experiment_id}.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    meta_df = pd.read_csv(meta_file)

    # Merge on snip_id (metadata is the source of truth for genotype/embryo_id)
    # Only need snip_id and predicted_stage_hpf from metadata for trajectory extraction
    required_cols = ['snip_id', 'embryo_id', 'genotype', 'predicted_stage_hpf', 'baseline_deviation_normalized']
    meta_subset = meta_df[required_cols].copy()

    df = curv_df.merge(meta_subset, on='snip_id', how='inner')

    if len(df) == 0:
        raise ValueError(f"Merge resulted in empty dataframe (curv: {len(curv_df)}, meta: {len(meta_df)})")

    # Rename baseline_deviation_normalized to match expected column name
    df['normalized_baseline_deviation'] = df['baseline_deviation_normalized']

    # Add source experiment
    df['source_experiment'] = experiment_id

    if verbose:
        print(f"  Loaded {experiment_id}: {len(df)} rows")
        if 'genotype' in df.columns:
            print(f"    Genotypes: {df['genotype'].value_counts().to_dict()}")

    return df


def load_all_experiments(experiment_ids, verbose=True):
    """
    Load and combine data from multiple experiments.

    Parameters
    ----------
    experiment_ids : list of str
        List of experiment identifiers
    verbose : bool
        Print loading progress

    Returns
    -------
    pd.DataFrame
        Combined dataframe from all experiments
    """
    curv_dir = PROJECT_ROOT / "morphseq_playground" / "metadata" / "body_axis" / "summary"
    meta_dir = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"

    dfs = []

    for exp_id in experiment_ids:
        try:
            df = load_experiment_data(exp_id, curv_dir, meta_dir, verbose=verbose)
            dfs.append(df)
        except FileNotFoundError as e:
            if verbose:
                print(f"  Missing: {exp_id} - {e}")
            continue
        except Exception as e:
            if verbose:
                print(f"  Error loading {exp_id}: {e}")
            continue

    if not dfs:
        raise ValueError("No experiments could be loaded")

    combined_df = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"\nTotal: {len(combined_df)} rows from {len(dfs)} experiments")
        if 'genotype' in combined_df.columns:
            print(f"Overall genotype distribution:")
            print(combined_df['genotype'].value_counts())

    return combined_df


# ============================================================================
# CUSTOM SAVE FUNCTIONS
# ============================================================================

def save_plot_organized(output_dir, experiment_id, genotype, plot_type, name, fig):
    """
    Save plot to organized directory structure.

    Parameters
    ----------
    output_dir : Path
        Base output directory
    experiment_id : str
        Experiment identifier (e.g., '20250305')
    genotype : str
        Genotype (e.g., 'cep290_homozygous')
    plot_type : str
        Plot type subdirectory (e.g., 'coassoc_matrices', 'temporal_trends')
    name : str
        Plot filename without extension
    fig : matplotlib.figure.Figure
        Figure to save
    """
    plot_dir = output_dir / experiment_id / genotype / plot_type
    plot_dir.mkdir(parents=True, exist_ok=True)

    filepath = plot_dir / f"{name}.png"
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    if VERBOSE_OUTPUT:
        print(f"      ✓ Saved {filepath.relative_to(output_dir)}")


def save_data_organized(output_dir, experiment_id, genotype, name, obj):
    """Save data object to organized directory structure."""
    import pickle

    data_dir = output_dir / experiment_id / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    filepath = data_dir / f"{name}_{genotype}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    if VERBOSE_OUTPUT:
        print(f"      ✓ Saved {filepath.relative_to(output_dir)}")


# ============================================================================
# BOOTSTRAP FUNCTION FOR HIERARCHICAL CLUSTERING
# ============================================================================

def run_bootstrap_hierarchical(D, k, n_bootstrap=100, frac=0.8, random_state=42, verbose=False):
    """
    Bootstrap resampling using HIERARCHICAL clustering.

    MODIFIED: Stores per-iteration cluster labels for posterior analysis.
    NO co-association matrix computation.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n x n)
    k : int
        Number of clusters
    n_bootstrap : int
        Number of bootstrap iterations
    frac : float
        Fraction of samples per iteration
    random_state : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    dict with keys:
        - 'reference_labels': Cluster labels from full dataset
        - 'bootstrap_results': List of dicts with 'labels' and 'indices' per iteration
    """
    n = len(D)
    np.random.seed(random_state)
    bootstrap_results = []

    for iteration in range(n_bootstrap):
        if verbose and iteration % 20 == 0:
            print(f"    Bootstrap iteration {iteration}/{n_bootstrap}")

        # Sample embryos
        sample_size = int(n * frac)
        sampled_indices = np.random.choice(n, size=sample_size, replace=False)

        # Get submatrix for this sample
        D_sample = D[np.ix_(sampled_indices, sampled_indices)]

        # Cluster with hierarchical
        hc = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels_sample = hc.fit_predict(D_sample)

        # Map back to full array (-1 for non-sampled)
        labels_full = np.full(n, -1, dtype=int)
        labels_full[sampled_indices] = labels_sample

        # Store iteration results
        bootstrap_results.append({
            'labels': labels_full,
            'indices': sampled_indices
        })

    # Run final clustering on full data as reference
    if verbose:
        print(f"    Running reference clustering on full dataset...")
    hc_ref = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    reference_labels = hc_ref.fit_predict(D)

    return {
        'reference_labels': reference_labels,
        'bootstrap_results': bootstrap_results
    }


def cluster_hierarchical_cooccurrence(C, k):
    """Hierarchical clustering on co-occurrence matrix (consensus approach)."""
    labels = consensus_clustering(C, k)
    return labels, None


# ============================================================================
# MAIN CLUSTERING FUNCTION
# ============================================================================

def run_clustering_for_genotype(df, experiment_id, genotype, verbose=True):
    """
    Run hierarchical consensus clustering for a single experiment/genotype combination.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe (will be filtered)
    experiment_id : str
        Experiment to analyze
    genotype : str
        Genotype to analyze
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results for all k values
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing: {experiment_id} / {genotype}")
        print(f"{'='*80}")

    # ========== STEP 1: Filter and extract trajectories ==========
    if verbose:
        print(f"\nStep 1: Extracting trajectories...")

    # Filter to experiment and genotype
    df_filtered = df[
        (df['source_experiment'] == experiment_id) &
        (df['genotype'] == genotype)
    ].copy()

    if len(df_filtered) == 0:
        print(f"  WARNING: No data for {experiment_id} / {genotype}")
        return None

    trajectories, embryo_ids, df_long = extract_trajectories(
        df_filtered,
        genotype_filter=genotype,
        metric_name=METRIC_NAME,
        verbose=False
    )

    # Filter by minimum timepoints
    timepoints_per_embryo = df_long.groupby('embryo_id').size()
    valid_embryos = timepoints_per_embryo[timepoints_per_embryo >= MIN_TIMEPOINTS].index.tolist()

    if len(valid_embryos) == 0:
        print(f"  WARNING: No valid trajectories for {experiment_id} / {genotype}")
        return None

    df_long = df_long[df_long['embryo_id'].isin(valid_embryos)].copy()
    mask = np.array([eid in valid_embryos for eid in embryo_ids])
    trajectories = [t for t, m in zip(trajectories, mask) if m]
    embryo_ids = [e for e, m in zip(embryo_ids, mask) if m]

    if verbose:
        print(f"  Extracted {len(trajectories)} trajectories")

    # ========== STEP 2: Interpolate and compute DTW ==========
    if verbose:
        print(f"\nStep 2: Computing DTW distance matrix...")

    trajectories_interp, embryo_ids_ordered, orig_lens, common_grid = interpolate_to_common_grid(
        df_long, grid_step=GRID_STEP, verbose=False
    )

    D = compute_dtw_distance_matrix(
        trajectories_interp, window=DTW_WINDOW, verbose=False
    )

    if verbose:
        print(f"  DTW matrix shape: {D.shape}")

    # ========== STEP 3: Run clustering for all k values ==========
    if verbose:
        print(f"\nStep 3: Running hierarchical consensus clustering...")

    all_results = {}

    for k in K_VALUES:
        # Skip k if too high for sample size (need at least k+2 samples)
        n_samples = len(trajectories_interp)
        if k >= n_samples:
            if verbose:
                print(f"\n  k={k}: Skipped (too few samples: {n_samples})")
            continue

        if verbose:
            print(f"\n  k={k}:")

        # Bootstrap to get co-occurrence matrix
        boot_result = run_bootstrap_hierarchical(
            D, k, n_bootstrap=N_BOOTSTRAP, frac=BOOTSTRAP_FRAC,
            random_state=RANDOM_SEED, verbose=False
        )
        C = boot_result['coassoc']

        # Consensus clustering
        labels, _ = cluster_hierarchical_cooccurrence(C, k)

        # Check if clustering resulted in at least 2 clusters
        n_clusters_found = len(np.unique(labels))
        if n_clusters_found < 2:
            if verbose:
                print(f"    Skipped (consensus found only {n_clusters_found} cluster)")
            continue

        # Compute silhouette
        try:
            sil = silhouette_score(D, labels, metric='precomputed')
        except ValueError:
            sil = np.nan
            if verbose:
                print(f"    (skipped silhouette - insufficient clusters)")

        # Membership analysis
        try:
            membership = analyze_membership(D, labels, C, core_thresh=CORE_THRESHOLD)
        except ValueError:
            # Skip this k value if membership analysis fails
            if verbose:
                print(f"    Skipped (membership analysis failed)")
            continue

        all_results[k] = {
            'labels': labels,
            'coassoc': C,
            'silhouette': sil,
            'membership': membership,
            'n_core': membership['summary']['n_core'],
            'n_uncertain': membership['summary']['n_uncertain'],
            'n_outlier': membership['summary']['n_outlier']
        }

        if verbose:
            print(f"    Silhouette: {sil:.4f}")
            print(f"    Core/Uncertain/Outlier: {all_results[k]['n_core']}/{all_results[k]['n_uncertain']}/{all_results[k]['n_outlier']}")

    # Store preprocessing results
    precomp_results = {
        'trajectories': trajectories_interp,
        'common_grid': common_grid,
        'df_long': df_long,
        'embryo_ids': embryo_ids_ordered,
        'distance_matrix': D
    }

    return {
        'all_results': all_results,
        'precomp_results': precomp_results
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def generate_plots_for_genotype(results, experiment_id, genotype, verbose=True):
    """
    Generate all plots for a single experiment/genotype combination.

    Organizes plots by type:
    - coassoc_matrices/
    - temporal_trends/
    - cluster_overlays/
    - membership_vs_k.png (root level)
    """
    if verbose:
        print(f"\nGenerating plots for {experiment_id} / {genotype}...")

    all_results = results['all_results']
    precomp = results['precomp_results']

    # ========== SECTION A: Co-association matrices ==========
    if verbose:
        print(f"\n  Co-association matrices:")

    for k in K_VALUES:
        try:
            C = all_results[k]['coassoc']
            labels = all_results[k]['labels']

            fig = plot_coassoc_matrix(
                C, labels=labels, k=k,
                title=f"Co-association Matrix (k={k})"
            )

            save_plot_organized(OUTPUT_DIR, experiment_id, genotype,
                              'coassoc_matrices', f'coassoc_k{k}', fig)
            plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not generate coassoc plot for k={k}: {e}")

    # ========== SECTION B: Temporal trends with membership ==========
    if verbose:
        print(f"\n  Temporal trends:")

    for k in K_VALUES:
        try:
            labels = all_results[k]['labels']
            membership = all_results[k]['membership']
            membership_classification = membership['classification']

            # Group trajectories by cluster
            trajectories = precomp['trajectories']
            common_grid = precomp['common_grid']
            df_long = precomp['df_long']
            embryo_ids = precomp['embryo_ids']

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

            # Generate plot
            fig = plot_temporal_trends_with_membership(
                trajectories_by_cluster,
                common_grid,
                k=k,
                membership_classification=membership_classification,
                cluster_indices_map=cluster_indices_map,
                title=f"Temporal Trends by Cluster (k={k})"
            )

            save_plot_organized(OUTPUT_DIR, experiment_id, genotype,
                              'temporal_trends', f'temporal_trends_k{k}', fig)
            plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not generate temporal trends for k={k}: {e}")

    # ========== SECTION C: Cluster trajectory overlays ==========
    if verbose:
        print(f"\n  Cluster overlays:")

    for k in K_VALUES:
        try:
            labels = all_results[k]['labels']

            # Group trajectories by cluster (reuse from above)
            trajectories = precomp['trajectories']
            common_grid = precomp['common_grid']
            df_long = precomp['df_long']
            embryo_ids = precomp['embryo_ids']

            trajectories_by_cluster = {}
            for cluster_id in np.unique(labels):
                cluster_indices = np.where(labels == cluster_id)[0]
                cluster_trajs = [trajectories[i] for i in cluster_indices]
                cluster_embryo_ids = [embryo_ids[i] for i in cluster_indices]

                cluster_trajs_padded = pad_trajectories_for_plotting(
                    cluster_trajs, common_grid, df_long, cluster_embryo_ids, verbose=False
                )
                trajectories_by_cluster[cluster_id] = cluster_trajs_padded

            # Generate overlay plot
            fig = plot_cluster_trajectory_overlay(
                trajectories_by_cluster,
                common_grid,
                k=k,
                title=f"Cluster Trajectory Overlay (k={k})"
            )

            save_plot_organized(OUTPUT_DIR, experiment_id, genotype,
                              'cluster_overlays', f'cluster_overlay_k{k}', fig)
            plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not generate overlay for k={k}: {e}")

    # ========== SECTION D: Membership vs K plot ==========
    if verbose:
        print(f"\n  Membership vs K:")

    try:
        # Collect membership data across all k values
        all_k_membership = {}
        for k in K_VALUES:
            all_k_membership[k] = all_results[k]['membership']

        # Generate plot
        fig = plot_membership_vs_k(
            all_k_membership,
            best_k=None,
            title=f"Membership Distribution Across K Values"
        )

        # Save to genotype root (not in subfolder)
        plot_dir = OUTPUT_DIR / experiment_id / genotype
        plot_dir.mkdir(parents=True, exist_ok=True)
        filepath = plot_dir / "membership_vs_k.png"
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)

        if verbose:
            print(f"      ✓ Saved {filepath.relative_to(OUTPUT_DIR)}")
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not generate membership vs k plot: {e}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for multi-experiment clustering."""
    if VERBOSE_OUTPUT:
        print("\n" + "="*80)
        print("MULTI-EXPERIMENT HIERARCHICAL CONSENSUS CLUSTERING")
        print("="*80)
        print(f"\nExperiments: {EXPERIMENTS}")
        print(f"K values: {K_VALUES}")
        print(f"Bootstrap: {N_BOOTSTRAP} iterations, {BOOTSTRAP_FRAC*100:.0f}% sampling")

    # ========== Load full dataset ==========
    if VERBOSE_OUTPUT:
        print(f"\nLoading experiments...")

    df = load_all_experiments(EXPERIMENTS, verbose=VERBOSE_OUTPUT)

    if VERBOSE_OUTPUT:
        print(f"  ✓ Loaded {len(df)} total samples")
        print(f"  ✓ Unique source experiments: {df['source_experiment'].unique().tolist()}")

    # ========== Process each experiment/genotype combination ==========
    for experiment_id in EXPERIMENTS:
        # Filter to experiment
        df_exp = df[df['source_experiment'] == experiment_id].copy()

        if len(df_exp) == 0:
            if VERBOSE_OUTPUT:
                print(f"\nWARNING: No data for experiment {experiment_id}")
            continue

        # Get unique genotypes for this experiment
        genotypes = sorted(df_exp['genotype'].unique())

        if VERBOSE_OUTPUT:
            print(f"\n{'='*80}")
            print(f"Experiment {experiment_id}: Found {len(genotypes)} genotypes")
            print(f"{'='*80}")
            for g in genotypes:
                count = (df_exp['genotype'] == g).sum()
                print(f"  {g}: {count} samples")

        for genotype in genotypes:
            try:
                # Run clustering
                results = run_clustering_for_genotype(
                    df, experiment_id, genotype, verbose=VERBOSE_OUTPUT
                )

                if results is None:
                    continue

                # Generate plots
                generate_plots_for_genotype(
                    results, experiment_id, genotype, verbose=VERBOSE_OUTPUT
                )

                # Save results
                save_data_organized(OUTPUT_DIR, experiment_id, genotype, 'results', results)

            except Exception as e:
                if VERBOSE_OUTPUT:
                    print(f"\nERROR processing {experiment_id} / {genotype}: {e}")
                    import traceback
                    traceback.print_exc()

    # ========== Generate genotype overlay plots ==========
    if VERBOSE_OUTPUT:
        print(f"\n{'='*80}")
        print("GENERATING GENOTYPE OVERLAY PLOTS")
        print(f"{'='*80}")

    for experiment_id in EXPERIMENTS:
        # Filter to experiment
        df_exp = df[df['source_experiment'] == experiment_id].copy()

        if len(df_exp) == 0:
            continue

        if VERBOSE_OUTPUT:
            print(f"\n  {experiment_id}:")

        try:
            # Create genotype overlay plot
            save_path = OUTPUT_DIR / experiment_id / 'genotype_overlay.png'
            fig = plot_embryos_metric_over_time(
                df_exp,
                metric=METRIC_NAME,
                time_col='predicted_stage_hpf',
                embryo_col='embryo_id',
                color_by='genotype',
                show_individual=True,
                show_mean=True,
                show_sd_band=False,
                alpha_individual=0.2,
                alpha_mean=0.9,
                title=f"Genotype Overlay - {experiment_id}",
                save_path=save_path
            )
            plt.close(fig)

            if VERBOSE_OUTPUT:
                print(f"    ✓ Saved {save_path.relative_to(OUTPUT_DIR)}")
        except Exception as e:
            if VERBOSE_OUTPUT:
                print(f"    ERROR: {e}")

    if VERBOSE_OUTPUT:
        print(f"\n{'='*80}")
        print("COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
