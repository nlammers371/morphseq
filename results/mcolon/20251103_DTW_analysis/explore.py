#!/usr/bin/env python3
"""
Main DTW Clustering Pipeline Script (explore.py)

Orchestrates complete analysis from data preprocessing through model fitting.
Runs Steps 0-6 of the trajectory clustering pipeline:

- Step 0: Load data and compute DTW distance matrix
- Step 1: Baseline clustering with k-medoids
- Step 2: Bootstrap stability analysis
- Step 3: Select optimal k with multiple metrics
- Step 4: Classify membership (core/uncertain/outlier)
- Step 5: Fit mixed-effects models with DBA centroids
- Step 6: Save results and generate core plots

Configuration
=============
All parameters are centralized in config.py

Usage
=====
python explore.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import (
    OUTPUT_DIR, K_VALUES, N_BOOTSTRAP, BOOTSTRAP_FRAC,
    CORE_THRESHOLD, OUTLIER_THRESHOLD, USE_DBA, DBA_MAX_ITER,
    RANDOM_SEED, VERBOSE_OUTPUT, PRIOR_K
)

# Import pipeline modules (handling hyphenated filenames)
import importlib.util
import sys

def load_module(name, filepath):
    """Load a module from a file path (handles hyphens in filenames)."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules with hyphens in names
cluster_module = load_module("cluster_module", "cluster-module.py")
select_k_module = load_module("select_k_module", "select-k-simple.py")
membership_module = load_module("membership_module", "membership-module.py")
fit_models_module = load_module("fit_models_module", "fit-models-module.py")
io_module = load_module("io_module", "io-module.py")

# Load our preprocessing script (normal underscore name)
dtw_precompute_module = load_module("dtw_precompute", "0_dtw_precompute.py")

# Extract functions
run_baseline = cluster_module.run_baseline
run_bootstrap = cluster_module.run_bootstrap
plot_coassoc_matrix = cluster_module.plot_coassoc_matrix
evaluate_all_k = select_k_module.evaluate_all_k
suggest_k = select_k_module.suggest_k
plot_metric_comparison = select_k_module.plot_metric_comparison
analyze_membership = membership_module.analyze_membership
plot_membership_distribution = membership_module.plot_membership_distribution
plot_membership_vs_k = membership_module.plot_membership_vs_k
fit_cluster_model = fit_models_module.fit_cluster_model
plot_cluster_trajectories = fit_models_module.plot_cluster_trajectories
plot_cluster_comparison = fit_models_module.plot_cluster_comparison
plot_spline_vs_dba = fit_models_module.plot_spline_vs_dba
plot_dba_trajectory = fit_models_module.plot_dba_trajectory
save_data = io_module.save_data
save_plot = io_module.save_plot
precompute_dtw = dtw_precompute_module.precompute_dtw

# Import plot utilities
from plot_utils import (plot_temporal_trends_by_cluster, plot_cluster_trajectory_overlay,
                        plot_temporal_trends_with_membership)

# Import padding helper from src
from src.analyze.dtw_time_trend_analysis.trajectory_utils import pad_trajectories_for_plotting

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: BASELINE CLUSTERING
# ============================================================================

def step_1_baseline_clustering(D, K_VALUES, verbose=True):
    """
    Run k-medoids clustering for each k value.

    Parameters
    ----------
    D : np.ndarray
        DTW distance matrix
    K_VALUES : list
        List of k values to try
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Baseline results for each k
    """
    if verbose:
        print("\n" + "="*80)
        print("STEP 1: BASELINE CLUSTERING")
        print("="*80)

    baseline_results = {}
    for k in K_VALUES:
        if verbose:
            print(f"\n  Clustering with k={k}...")
        labels, medoids = run_baseline(D, k)
        baseline_results[k] = {
            'labels': labels,
            'medoids': medoids,
            'n_clusters': k
        }
        if verbose:
            unique_labels = len(np.unique(labels))
            print(f"    Found {unique_labels} clusters")
            for c in np.unique(labels):
                n_in_cluster = (labels == c).sum()
                print(f"      Cluster {c}: {n_in_cluster} members")

    # Don't save at step 1, will save at step 3 with metrics
    return baseline_results


# ============================================================================
# STEP 2: BOOTSTRAP STABILITY
# ============================================================================

def step_2_bootstrap_stability(D, K_VALUES, N_BOOTSTRAP, BOOTSTRAP_FRAC, baseline_results, verbose=True):
    """
    Run bootstrap resampling to assess clustering stability.

    Parameters
    ----------
    D : np.ndarray
        DTW distance matrix
    K_VALUES : list
        List of k values to evaluate
    N_BOOTSTRAP : int
        Number of bootstrap iterations
    BOOTSTRAP_FRAC : float
        Fraction of data to sample in each iteration
    baseline_results : dict
        Results from Step 1 baseline clustering
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Bootstrap results with co-association matrices for each k
    """
    if verbose:
        print("\n" + "="*80)
        print("STEP 2: BOOTSTRAP STABILITY ANALYSIS")
        print("="*80)

    bootstrap_results = {}

    for k in K_VALUES:
        if verbose:
            print(f"\n  Bootstrap for k={k} ({N_BOOTSTRAP} iterations)...")

        boot_res = run_bootstrap(D, k, n_bootstrap=N_BOOTSTRAP, frac=BOOTSTRAP_FRAC, verbose=verbose)
        bootstrap_results[k] = boot_res

        if verbose:
            print(f"    Mean silhouette: {boot_res['mean_silhouette']:.3f}")
            print(f"    Mean ARI: {boot_res['mean_ari']:.3f}")

        # Save individual k results
        save_data(2, f'bootstrap_k{k}', boot_res, OUTPUT_DIR)

        # Plot co-association matrix for this k
        if verbose:
            print(f"    Plotting co-association matrix...")
        try:
            baseline_labels = baseline_results[k]['labels']
            fig = plot_coassoc_matrix(boot_res['coassoc'], labels=baseline_labels, k=k)
            save_plot(2, f'bootstrap_k{k}_coassoc', fig, OUTPUT_DIR)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"      Warning: Could not plot co-association: {e}")

    return bootstrap_results


# ============================================================================
# STEP 3: K SELECTION
# ============================================================================

def step_3_select_k(D, baseline_results, bootstrap_results, PRIOR_K=3, verbose=True):
    """
    Evaluate k-selection metrics and recommend best k.

    Parameters
    ----------
    D : np.ndarray
        DTW distance matrix
    baseline_results : dict
        Results from Step 1
    bootstrap_results : dict
        Results from Step 2
    PRIOR_K : int
        Prior expected k (used by some metrics)
    verbose : bool
        Print progress

    Returns
    -------
    int
        Recommended k value
    dict
        All metrics for each k
    """
    if verbose:
        print("\n" + "="*80)
        print("STEP 3: K SELECTION")
        print("="*80)

    # Extract k values
    K_VALUES = list(baseline_results.keys())

    # Evaluate all k values
    metrics = evaluate_all_k(D, baseline_results, bootstrap_results, verbose=verbose)

    # Suggest best k
    best_k = suggest_k(metrics, prior_k=PRIOR_K, verbose=verbose)

    if verbose:
        print(f"\n  RECOMMENDED K: {best_k}")
        print(f"\n  Metrics summary:")
        print(f"    K | Silhouette | Gap Stat | Eigengap | Bootstrap ARI")
        print(f"    {'-'*60}")
        for k in K_VALUES:
            gap_stat = metrics[k].get('gap_statistic', (np.nan, np.nan))[0]
            print(f"    {k} | {metrics[k].get('silhouette', np.nan):10.3f} | "
                  f"{gap_stat:8.3f} | "
                  f"{metrics[k].get('eigengap', np.nan):8.3f} | "
                  f"{metrics[k].get('mean_ari', np.nan):13.3f}")

    save_data(3, 'metrics', metrics, OUTPUT_DIR)
    save_data(3, 'best_k', best_k, OUTPUT_DIR)
    save_data(3, 'baseline_results', baseline_results, OUTPUT_DIR)

    # Plot k-selection metrics comparison
    if verbose:
        print(f"\n  Plotting k-selection metrics comparison...")
    try:
        fig = plot_metric_comparison(metrics, best_k=best_k)
        save_plot(3, 'k_selection_metrics', fig, OUTPUT_DIR)
        import matplotlib.pyplot as plt
        plt.close(fig)
        if verbose:
            print(f"    Saved k-selection metrics plot")
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not plot metrics: {e}")

    return best_k, metrics


# ============================================================================
# STEP 4: MEMBERSHIP ANALYSIS
# ============================================================================

def step_4_membership_analysis(D, baseline_results, bootstrap_results, best_k,
                               CORE_THRESHOLD, OUTLIER_THRESHOLD, verbose=True):
    """
    Classify embryos as core, uncertain, or outlier based on bootstrap stability.
    Analyzes membership for all k values, but returns detailed results for best_k.

    Parameters
    ----------
    D : np.ndarray
        DTW distance matrix
    baseline_results : dict
        Results from Step 1
    bootstrap_results : dict
        Results from Step 2
    best_k : int
        Selected k value from Step 3 (used for detailed membership analysis)
    CORE_THRESHOLD : float
        Threshold for core membership
    OUTLIER_THRESHOLD : float
        Threshold for outlier classification
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Membership classification and statistics for best_k
    dict
        Membership statistics for all k values
    """
    if verbose:
        print("\n" + "="*80)
        print("STEP 4: MEMBERSHIP ANALYSIS")
        print("="*80)

    # Analyze membership for all k values
    all_k_membership = {}

    K_VALUES = list(baseline_results.keys())

    for k in K_VALUES:
        if verbose:
            print(f"\n  Analyzing membership for k={k}...")

        labels = baseline_results[k]['labels']
        C = bootstrap_results[k]['coassoc']

        membership_results = analyze_membership(D, labels, C, core_thresh=CORE_THRESHOLD)
        all_k_membership[k] = membership_results

        if verbose:
            print(f"    Core members: {membership_results['summary']['n_core']}")
            print(f"    Uncertain members: {membership_results['summary']['n_uncertain']}")
            print(f"    Outliers: {membership_results['summary']['n_outlier']}")
            print(f"    Core fraction: {membership_results['summary']['core_fraction']:.1%}")

        # Plot membership distribution for this k
        if verbose:
            print(f"    Plotting membership distribution...")
        try:
            fig = plot_membership_distribution(
                membership_results['classification'],
                cluster_stats=membership_results['cluster_stats'],
                title=f"Membership Distribution (k={k})"
            )
            save_plot(4, f'membership_distribution_k{k}', fig, OUTPUT_DIR)
            import matplotlib.pyplot as plt
            plt.close(fig)
            if verbose:
                print(f"      Saved membership plot for k={k}")
        except Exception as e:
            if verbose:
                print(f"      Warning: Could not plot membership for k={k}: {e}")

    # Save results for all k and detailed results for best_k
    best_k_membership = all_k_membership[best_k]

    if verbose:
        print(f"\n  Detailed membership summary for k={best_k}:")
        print(f"    Per-cluster breakdown:")
        for cluster_id, stats in best_k_membership['cluster_stats'].items():
            print(f"      Cluster {cluster_id}: {stats['core']} core, "
                  f"{stats['uncertain']} uncertain, {stats['outlier']} outlier")

    save_data(4, 'membership_results', best_k_membership, OUTPUT_DIR)
    save_data(4, 'membership_all_k', all_k_membership, OUTPUT_DIR)
    save_data(4, 'core_indices', best_k_membership['core_indices'], OUTPUT_DIR)

    # Plot membership vs k (percentages across all k values)
    if verbose:
        print(f"\n  Plotting membership distribution across k values...")
    try:
        fig = plot_membership_vs_k(all_k_membership, best_k=best_k,
                                   title="Membership Category Percentages Across K Values")
        save_plot(4, 'membership_vs_k', fig, OUTPUT_DIR)
        import matplotlib.pyplot as plt
        plt.close(fig)
        if verbose:
            print(f"    Saved membership vs k plot")
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not plot membership vs k: {e}")

    return best_k_membership, all_k_membership


# ============================================================================
# STEP 5: MODEL FITTING
# ============================================================================

def step_5_fit_models(precomp_results, baseline_results, membership_results,
                      best_k, USE_DBA, DBA_MAX_ITER, K_VALUES, all_k_membership=None, verbose=True):
    """
    Fit mixed-effects models to clusters and generate plots for all k values.

    Parameters
    ----------
    precomp_results : dict
        Results from Step 0 (preprocessing)
    baseline_results : dict
        Results from Step 1
    membership_results : dict
        Results from Step 4 (for best_k)
    best_k : int
        Selected k value
    USE_DBA : bool
        Whether to use DBA for centroids
    DBA_MAX_ITER : int
        DBA iterations
    K_VALUES : list
        All k values to generate plots for
    all_k_membership : dict, optional
        Membership results for all k values (from Step 4)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Fitted models for each cluster
    """
    if verbose:
        print("\n" + "="*80)
        print("STEP 5: MODEL FITTING")
        print("="*80)

    labels = baseline_results[best_k]['labels']
    trajectories = precomp_results['trajectories']
    common_grid = precomp_results['common_grid']
    df_long = precomp_results['df_long']
    embryo_ids = precomp_results['embryo_ids']

    # Basic validation
    if verbose:
        print(f"\n  Data dimensions:")
        print(f"    Embryos: {len(trajectories)}")
        print(f"    Labels: {labels.shape[0]}")
        print(f"    Common grid: {len(common_grid)} timepoints")

    if labels.shape[0] != len(trajectories):
        if verbose:
            print(f"  WARNING: Dimension mismatch - labels ({labels.shape[0]}) != trajectories ({len(trajectories)})")

    cluster_models = {}
    trajectories_by_cluster = {}

    for cluster_id in np.unique(labels):
        if verbose:
            print(f"\n  Fitting model for cluster {cluster_id}...")

        # Get members of this cluster (indices in original array)
        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) < 2:
            if verbose:
                print(f"    Skipping (too few members: {len(cluster_indices)})")
            cluster_models[cluster_id] = None
            continue

        # Get trajectories for this cluster
        cluster_trajectories = [trajectories[i] for i in cluster_indices]
        trajectories_by_cluster[cluster_id] = cluster_trajectories

        # Get embryo_ids for this cluster (needed for padding)
        cluster_embryo_ids = [embryo_ids[i] for i in cluster_indices]

        # Pad trajectories to uniform length for plotting
        cluster_trajectories_padded = pad_trajectories_for_plotting(
            cluster_trajectories,
            common_grid,
            df_long,
            cluster_embryo_ids,
            verbose=False
        )

        # Get core mask for members of this cluster only
        # core_indices contains global indices, cluster_indices also contains global indices
        core_indices = membership_results['core_indices']
        core_mask = np.array([idx in core_indices for idx in cluster_indices])

        if verbose:
            n_core = core_mask.sum()
            print(f"    Size: {len(cluster_trajectories)}, Core: {n_core}")

        # Fit model
        try:
            model = fit_cluster_model(
                cluster_trajectories,
                common_grid=common_grid,
                core_mask=core_mask if core_mask.any() else None,
                use_dba=USE_DBA
            )
            cluster_models[cluster_id] = model

            if verbose:
                print(f"    Mean R²: {model['mean_r2']:.3f}")
                if model['dba_spline'] is not None:
                    print(f"    DBA: computed")

            # Plot individual cluster trajectories with membership highlighting
            if verbose:
                print(f"    Plotting cluster {cluster_id} trajectories with membership...")
            try:
                fig = plot_cluster_trajectories(
                    cluster_trajectories_padded,
                    common_grid,
                    cluster_id=cluster_id,
                    cluster_indices=cluster_indices,
                    membership_classification=membership_results['classification'],
                    title=f"Cluster {cluster_id} Temporal Trends with Membership (k={best_k})"
                )
                save_plot(5, f'cluster_trajectories_with_membership_k{best_k}_c{cluster_id}', fig, OUTPUT_DIR)
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                if verbose:
                    print(f"      Warning: Could not plot trajectories with membership: {e}")

            # Plot DBA trajectory overlay
            if model.get('dba_spline') is not None:
                if verbose:
                    print(f"    Plotting DBA trajectory visualization...")
                try:
                    # Extract DBA curve values at common grid points
                    t_centered = common_grid - model['t_center']
                    dba_trajectory = model['dba_spline'](t_centered)

                    fig = plot_dba_trajectory(
                        cluster_trajectories_padded,
                        common_grid,
                        dba_trajectory=dba_trajectory,
                        cluster_id=cluster_id,
                        title=f"Cluster {cluster_id} DBA Trajectory (k={best_k})"
                    )
                    save_plot(5, f'dba_trajectory_k{best_k}_c{cluster_id}', fig, OUTPUT_DIR)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception as e:
                    if verbose:
                        print(f"      Warning: Could not plot DBA trajectory: {e}")

            # Plot DBA vs Spline comparison if DBA was computed
            if model.get('dba_spline') is not None:
                if verbose:
                    print(f"    Plotting DBA vs Spline comparison...")
                try:
                    fig = plot_spline_vs_dba(
                        common_grid,
                        model,
                        title=f"Cluster {cluster_id} Mean Curve Methods (k={best_k})"
                    )
                    save_plot(5, f'spline_vs_dba_k{best_k}_c{cluster_id}', fig, OUTPUT_DIR)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception as e:
                    if verbose:
                        print(f"      Warning: Could not plot DBA comparison: {e}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            cluster_models[cluster_id] = None

    save_data(5, 'cluster_models', cluster_models, OUTPUT_DIR)

    # Pad trajectories for plotting (convert variable length to uniform length)
    if verbose:
        print(f"\n  Preparing trajectories for plotting (padding to uniform length)...")

    # Create padded version grouped by cluster
    trajectories_by_cluster_padded = {}
    for cluster_id, cluster_trajs in trajectories_by_cluster.items():
        # Get embryo_ids for this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embryo_ids = [embryo_ids[i] for i in cluster_indices]

        # Pad to uniform length
        padded = pad_trajectories_for_plotting(
            cluster_trajs,
            common_grid,
            df_long,
            cluster_embryo_ids,
            verbose=False
        )
        trajectories_by_cluster_padded[cluster_id] = padded

    # Plot all clusters together for comparison
    if verbose:
        print(f"\n  Plotting cluster comparison (all k={best_k} clusters together)...")
    try:
        fig = plot_cluster_comparison(
            trajectories_by_cluster_padded,
            common_grid,
            title=f"Cluster Trajectory Comparison (k={best_k})"
        )
        save_plot(5, f'cluster_comparison_k{best_k}', fig, OUTPUT_DIR)
        import matplotlib.pyplot as plt
        plt.close(fig)
        if verbose:
            print(f"    Saved cluster comparison plot")
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not plot comparison: {e}")

    # ===== GENERATE PLOTS FOR ALL K VALUES =====
    if verbose:
        print(f"\n  Generating plots for all k values...")

    for k in K_VALUES:
        if verbose:
            print(f"\n    Generating plots for k={k}...")

        # Get labels for this k
        labels_k = baseline_results[k]['labels']

        # Build trajectories grouped by cluster for this k
        trajectories_by_cluster_k = {}
        cluster_indices_map_k = {}

        for cluster_id in np.unique(labels_k):
            cluster_indices = np.where(labels_k == cluster_id)[0]

            # Skip clusters with too few members
            if len(cluster_indices) < 1:
                continue

            cluster_trajs = [trajectories[i] for i in cluster_indices]
            cluster_embryo_ids = [embryo_ids[i] for i in cluster_indices]

            # Pad trajectories to uniform length for plotting
            cluster_trajs_padded = pad_trajectories_for_plotting(
                cluster_trajs,
                common_grid,
                df_long,
                cluster_embryo_ids,
                verbose=False
            )

            trajectories_by_cluster_k[cluster_id] = cluster_trajs_padded
            cluster_indices_map_k[cluster_id] = cluster_indices

        # 1. Plot temporal trends by cluster (standard)
        if verbose:
            print(f"      Plotting temporal trends...")
        try:
            fig = plot_temporal_trends_by_cluster(
                trajectories_by_cluster_k,
                common_grid,
                k=k,
                title=f"Temporal Trends by Cluster (k={k})"
            )
            save_plot(5, f'temporal_trends_by_cluster_k{k}', fig, OUTPUT_DIR)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"        Warning: Could not plot temporal trends: {e}")

        # 2. Plot cluster trajectory overlay (standard)
        if verbose:
            print(f"      Plotting cluster overlay...")
        try:
            fig = plot_cluster_trajectory_overlay(
                trajectories_by_cluster_k,
                common_grid,
                k=k,
                title=f"Cluster Trajectory Overlay (k={k})"
            )
            save_plot(5, f'cluster_trajectory_overlay_k{k}', fig, OUTPUT_DIR)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            if verbose:
                print(f"        Warning: Could not plot cluster overlay: {e}")

        # 3. Plot temporal trends with membership (NEW - if membership data available)
        if all_k_membership is not None and k in all_k_membership:
            if verbose:
                print(f"      Plotting temporal trends with membership...")
            try:
                membership_k = all_k_membership[k]
                fig = plot_temporal_trends_with_membership(
                    trajectories_by_cluster_k,
                    common_grid,
                    k=k,
                    membership_classification=membership_k['classification'],
                    cluster_indices_map=cluster_indices_map_k,
                    title=f"Temporal Trends by Cluster with Membership (k={k})"
                )
                save_plot(5, f'temporal_trends_with_membership_k{k}', fig, OUTPUT_DIR)
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                if verbose:
                    print(f"        Warning: Could not plot membership trends: {e}")

    return cluster_models


# ============================================================================
# STEP 6: GENERATE OUTPUTS AND PLOTS
# ============================================================================

def step_6_generate_outputs(precomp_results, baseline_results, bootstrap_results,
                            best_k, membership_results, cluster_models,
                            embryo_ids, all_k_membership=None, verbose=True):
    """
    Generate summary statistics and core visualizations.

    Parameters
    ----------
    precomp_results : dict
        Results from Step 0
    baseline_results : dict
        Results from Step 1
    bootstrap_results : dict
        Results from Step 2
    best_k : int
        Selected k value
    membership_results : dict
        Results from Step 4 (for best_k)
    cluster_models : dict
        Results from Step 5
    embryo_ids : list
        Ordered embryo IDs
    all_k_membership : dict, optional
        Membership results for all k values
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Summary statistics
    """
    if verbose:
        print("\n" + "="*80)
        print("STEP 6: SUMMARY AND OUTPUTS")
        print("="*80)

    labels = baseline_results[best_k]['labels']

    # Create summary table
    summary_data = []
    for i, embryo_id in enumerate(embryo_ids):
        cluster = labels[i]
        member_info = membership_results['classification'].get(i, {})

        summary_data.append({
            'embryo_id': embryo_id,
            'cluster': cluster,
            'membership': member_info.get('category', 'unknown'),
            'intra_coassoc': member_info.get('intra_coassoc', np.nan),
            'silhouette': member_info.get('silhouette', np.nan)
        })

    summary_df = pd.DataFrame(summary_data)

    if verbose:
        print(f"\n  Final assignments (k={best_k}):")
        print(f"\n{summary_df[['embryo_id', 'cluster', 'membership']].head(10)}")
        print(f"\n  ... ({len(summary_df)} total)")

    # Save summary
    save_data(6, 'summary_table', summary_df, OUTPUT_DIR)

    # Save membership summary across all k values if available
    if all_k_membership:
        k_summary = {}
        for k, mem_res in all_k_membership.items():
            k_summary[k] = mem_res['summary']
        save_data(6, 'membership_summary_all_k', k_summary, OUTPUT_DIR)
        if verbose:
            print(f"\n  Saved membership summary for all k values")

    if verbose:
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print(f"\nKey output files:")
        print(f"  Data: {OUTPUT_DIR}/*_*/data/")
        print(f"  Plots: {OUTPUT_DIR}/*_*/plots/")
        print(f"\nKey plots to review:")
        print(f"  - {OUTPUT_DIR}/3_select_k/plots/k_selection_metrics.png")
        print(f"  - {OUTPUT_DIR}/2_select_k/plots/bootstrap_k*_coassoc.png")
        print(f"  - {OUTPUT_DIR}/5_fit_models/plots/cluster_trajectories_k{best_k}_*.png")
        print(f"  - {OUTPUT_DIR}/5_fit_models/plots/cluster_comparison_k{best_k}.png")
        print(f"  - {OUTPUT_DIR}/4_membership/plots/membership_distribution_k{best_k}.png")
        print(f"\n{'='*80}\n")

    return {
        'summary_df': summary_df,
        'best_k': best_k,
        'n_embryos': len(embryo_ids)
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config_overrides=None, verbose=True):
    """
    Run complete DTW clustering pipeline.

    Parameters
    ----------
    config_overrides : dict, optional
        Override default config values
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Complete results from all steps
    """
    if verbose:
        print("\n" + "="*80)
        print("DTW CLUSTERING PIPELINE - FULL RUN")
        print("="*80)

    # Use config defaults unless overridden
    k_values = config_overrides.get('K_VALUES', K_VALUES) if config_overrides else K_VALUES
    n_bootstrap = config_overrides.get('N_BOOTSTRAP', N_BOOTSTRAP) if config_overrides else N_BOOTSTRAP
    bootstrap_frac = config_overrides.get('BOOTSTRAP_FRAC', BOOTSTRAP_FRAC) if config_overrides else BOOTSTRAP_FRAC

    # ========== STEP 0: Precomputation ==========
    if verbose:
        print("\nRunning Step 0 (preprocessing)...")
    precomp_results = precompute_dtw(verbose=verbose)
    D = precomp_results['distance_matrix']
    embryo_ids = precomp_results['embryo_ids']

    # ========== STEP 1: Baseline ==========
    baseline_results = step_1_baseline_clustering(D, k_values, verbose=verbose)

    # ========== STEP 2: Bootstrap ==========
    bootstrap_results = step_2_bootstrap_stability(D, k_values, n_bootstrap, bootstrap_frac, baseline_results, verbose=verbose)

    # ========== STEP 3: Select K ==========
    best_k, metrics = step_3_select_k(D, baseline_results, bootstrap_results,
                                       PRIOR_K=PRIOR_K, verbose=verbose)

    # ========== STEP 4: Membership ==========
    membership_results, all_k_membership = step_4_membership_analysis(
        D, baseline_results, bootstrap_results, best_k,
        CORE_THRESHOLD, OUTLIER_THRESHOLD, verbose=verbose
    )

    # ========== STEP 5: Fit Models ==========
    cluster_models = step_5_fit_models(
        precomp_results, baseline_results, membership_results,
        best_k, USE_DBA, DBA_MAX_ITER, K_VALUES, all_k_membership=all_k_membership,
        verbose=verbose
    )

    # ========== STEP 6: Generate Outputs ==========
    outputs = step_6_generate_outputs(
        precomp_results, baseline_results, bootstrap_results,
        best_k, membership_results, cluster_models, embryo_ids,
        all_k_membership=all_k_membership, verbose=verbose
    )

    return {
        'precomp': precomp_results,
        'baseline': baseline_results,
        'bootstrap': bootstrap_results,
        'metrics': metrics,
        'best_k': best_k,
        'membership': membership_results,
        'membership_all_k': all_k_membership,
        'cluster_models': cluster_models,
        'outputs': outputs,
        'embryo_ids': embryo_ids
    }


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Run full pipeline
    np.random.seed(RANDOM_SEED)
    results = run_pipeline(verbose=VERBOSE_OUTPUT)

    print("\n✓ Pipeline execution complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"  Best k: {results['best_k']}")
    print(f"  Total embryos: {results['outputs']['n_embryos']}")
