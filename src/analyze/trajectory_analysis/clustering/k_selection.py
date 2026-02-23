"""
K Selection Pipeline for Consensus Clustering

Evaluates multiple k values BEFORE filtering to help decide optimal k.
This addresses the workflow issue where filtering was done before knowing
the right k, potentially removing embryos that would form good clusters.

New Workflow:
1. Compute distance matrix (no filtering)
2. Run bootstrap clustering for k in [2, 3, 4, 5, ...]
3. For each k, compute quality metrics:
   - % Core (high confidence assignments)
   - % Outlier (low confidence)
   - Mean max_p
   - Mean entropy
   - Silhouette score
4. Plot comparison → Pick best k
5. Re-run consensus pipeline with chosen k + filtering

Created: 2025-12-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path


# ============================================================================
# HELPER FUNCTIONS: Create convenient mappings
# ============================================================================

def _create_cluster_mappings(
    embryo_ids: List[str],
    cluster_labels: np.ndarray,
    k: int
) -> Dict[str, Any]:
    """
    Create convenient cluster assignment mappings.

    Parameters
    ----------
    embryo_ids : List[str]
        Ordered list of embryo identifiers
    cluster_labels : np.ndarray
        Cluster assignment for each embryo (length = len(embryo_ids))
    k : int
        Number of clusters

    Returns
    -------
    mappings : dict
        - 'cluster_labels': np.array of assignments
        - 'embryo_to_cluster': {embryo_id: cluster_id}
        - 'cluster_to_embryos': {cluster_id: [embryo_ids]}
    """
    # embryo_id → cluster_id
    embryo_to_cluster = dict(zip(embryo_ids, cluster_labels))

    # cluster_id → [embryo_ids] (reverse mapping)
    cluster_to_embryos = {}
    for cluster_id in range(k):
        mask = cluster_labels == cluster_id
        cluster_to_embryos[cluster_id] = [
            embryo_ids[i] for i in range(len(embryo_ids)) if mask[i]
        ]

    return {
        'cluster_labels': cluster_labels,
        'embryo_to_cluster': embryo_to_cluster,
        'cluster_to_embryos': cluster_to_embryos,
    }


def _create_membership_mappings(
    embryo_ids: List[str],
    categories: np.ndarray
) -> Dict[str, Any]:
    """
    Create convenient membership quality mappings.

    Parameters
    ----------
    embryo_ids : List[str]
        Ordered list of embryo identifiers
    categories : np.ndarray
        Membership quality for each embryo ('core', 'uncertain', 'outlier')

    Returns
    -------
    mappings : dict
        - 'membership_quality': np.array of quality categories
        - 'embryo_to_membership_quality': {embryo_id: quality}
    """
    embryo_to_membership_quality = dict(zip(embryo_ids, categories))

    return {
        'membership_quality': categories,
        'embryo_to_membership_quality': embryo_to_membership_quality,
    }


def add_membership_column(
    df: pd.DataFrame,
    classification: Dict[str, Any],
    column_name: str = 'membership'
) -> pd.DataFrame:
    """
    Add membership category column to DataFrame based on classification results.
    
    Maps each embryo_id to its membership category (core/uncertain/outlier).
    This enables using plot_feature_over_time with color_by='membership'.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame with 'embryo_id' column
    classification : Dict
        Output from classify_membership_2d() with 'embryo_ids' and 'category' keys
    column_name : str
        Name for the new column (default: 'membership')
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with new membership column added
        
    Example
    -------
    >>> classification = classify_membership_2d(posteriors['max_p'], ...)
    >>> df = add_membership_column(df, classification)
    >>> fig = plot_feature_over_time(df, features=[...], color_by='membership', facet_col='cluster')
    """
    # Create mapping from embryo_id to category
    embryo_to_cat = dict(zip(classification['embryo_ids'], classification['category']))
    
    # Map to DataFrame
    df = df.copy()
    df[column_name] = df['embryo_id'].map(embryo_to_cat)
    
    return df


def evaluate_k_range(
    D: np.ndarray,
    embryo_ids: List[str],
    k_range: List[int] = [2, 3, 4, 5, 6],
    n_bootstrap: int = 100,
    method: str = 'hierarchical',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate multiple k values for consensus clustering WITHOUT filtering.

    This helps decide the optimal k before applying any outlier filtering.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n)
    embryo_ids : List[str]
        Embryo identifiers
    k_range : List[int]
        K values to evaluate (default: [2, 3, 4, 5, 6])
    n_bootstrap : int
        Bootstrap iterations per k (default: 100)
    method : str, default='hierarchical'
        Clustering method: 'hierarchical' or 'kmedoids'
    verbose : bool
        Print progress

    Returns
    -------
    results : Dict
        Complete k-selection results with structure:

        {
            'best_k': int,              # Recommended k (highest % core)
            'k_values': [2, 3, 4, ...], # K values tested
            'summary_df': DataFrame,    # Quality metrics comparison table
            'embryo_ids': [embryo_ids], # Ordered list of embryo IDs
            'clustering_by_k': {
                k: {
                    'quality': {
                        'n_embryos': int,
                        'n_core': int,
                        'n_uncertain': int,
                        'n_outlier': int,
                        'pct_core': float,        # % core assignments
                        'pct_uncertain': float,   # % uncertain assignments
                        'pct_outlier': float,     # % outlier assignments
                        'mean_max_p': float,      # Mean max posterior probability
                        'mean_entropy': float,    # Mean assignment entropy
                        'silhouette': float,      # Silhouette score
                    },
                    'assignments': {
                        'cluster_labels': np.array,           # [embryo0_cluster, embryo1_cluster, ...]
                        'embryo_to_cluster': dict,            # {embryo_id: cluster_id}
                        'cluster_to_embryos': dict,           # {cluster_id: [embryo_ids]}
                    },
                    'membership': {
                        'membership_quality': np.array,               # ['core', 'uncertain', 'outlier', ...]
                        'embryo_to_membership_quality': dict,         # {embryo_id: 'core'/'uncertain'/'outlier'}
                    },
                    'posteriors': dict,         # Full posterior probabilities
                    'bootstrap_results': dict,  # Bootstrap clustering results
                    'classification': dict,     # Membership quality classification
                }
            }
        }

    Examples
    --------
    >>> results = evaluate_k_range(D, embryo_ids, k_range=[2, 3, 4, 5, 6])
    >>> print(f"Best k: {results['best_k']}")
    >>> # Get cluster assignment for specific embryo at k=3
    >>> cluster_id = results['clustering_by_k'][3]['assignments']['embryo_to_cluster']['embryo_001']
    >>> # Get all embryos in cluster 0 at k=4
    >>> embryos_in_c0 = results['clustering_by_k'][4]['assignments']['cluster_to_embryos'][0]
    >>> # Check membership quality
    >>> membership_quality = results['clustering_by_k'][3]['membership']['embryo_to_membership_quality']['embryo_001']
    """
    from .bootstrap_clustering import (
        run_bootstrap_hierarchical,
        run_bootstrap_kmedoids,
    )
    from .cluster_posteriors import analyze_bootstrap_results
    from .cluster_classification import classify_membership_2d
    from sklearn.metrics import silhouette_score

    # Validate method parameter
    if method not in ['hierarchical', 'kmedoids']:
        raise ValueError(
            f"method must be 'hierarchical' or 'kmedoids', got '{method}'"
        )

    results_by_k = {}

    for k in k_range:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating k={k} (method={method})")
            print('='*60)

        # Run bootstrap clustering with selected method
        if method == 'hierarchical':
            bootstrap_results = run_bootstrap_hierarchical(
                D=D,
                k=k,
                embryo_ids=embryo_ids,
                n_bootstrap=n_bootstrap,
                verbose=verbose
            )
        else:  # method == 'kmedoids'
            bootstrap_results = run_bootstrap_kmedoids(
                D=D,
                k=k,
                embryo_ids=embryo_ids,
                n_bootstrap=n_bootstrap,
                verbose=verbose
            )
        
        # Compute posteriors
        posteriors = analyze_bootstrap_results(bootstrap_results)
        
        # Classify membership quality
        classification = classify_membership_2d(
            max_p=posteriors['max_p'],
            log_odds_gap=posteriors['log_odds_gap'],
            modal_cluster=posteriors['modal_cluster'],
            embryo_ids=posteriors['embryo_ids']
        )
        
        # Compute silhouette score
        try:
            sil_score = silhouette_score(D, posteriors['modal_cluster'], metric='precomputed')
        except:
            sil_score = np.nan
        
        # Compute summary metrics
        categories = classification['category']
        n_total = len(categories)
        n_core = np.sum(categories == 'core')
        n_uncertain = np.sum(categories == 'uncertain')
        n_outlier = np.sum(categories == 'outlier')

        # Create convenient mappings
        assignments = _create_cluster_mappings(
            embryo_ids,
            posteriors['modal_cluster'],
            k
        )

        membership = _create_membership_mappings(
            embryo_ids,
            categories
        )

        # Organize results with clear naming
        results_by_k[k] = {
            'quality': {
                'n_embryos': n_total,
                'n_core': n_core,
                'n_uncertain': n_uncertain,
                'n_outlier': n_outlier,
                'pct_core': 100.0 * n_core / n_total,
                'pct_uncertain': 100.0 * n_uncertain / n_total,
                'pct_outlier': 100.0 * n_outlier / n_total,
                'mean_max_p': posteriors['max_p'].mean(),
                'mean_entropy': posteriors['entropy'].mean(),
                'silhouette': sil_score,
            },
            'assignments': assignments,
            'membership': membership,
            'posteriors': posteriors,
            'bootstrap_results': bootstrap_results,
            'classification': classification,
        }

        if verbose:
            quality = results_by_k[k]['quality']
            print(f"\nk={k} Summary:")
            print(f"  Core: {n_core} ({quality['pct_core']:.1f}%)")
            print(f"  Uncertain: {n_uncertain} ({quality['pct_uncertain']:.1f}%)")
            print(f"  Outlier: {n_outlier} ({quality['pct_outlier']:.1f}%)")
            print(f"  Mean max_p: {quality['mean_max_p']:.3f}")
            print(f"  Mean entropy: {quality['mean_entropy']:.3f}")
            print(f"  Silhouette: {quality['silhouette']:.3f}")
    
    # Create summary DataFrame
    summary_data = []
    for k in k_range:
        quality = results_by_k[k]['quality']
        summary_data.append({
            'k': k,
            'pct_core': quality['pct_core'],
            'pct_uncertain': quality['pct_uncertain'],
            'pct_outlier': quality['pct_outlier'],
            'mean_max_p': quality['mean_max_p'],
            'mean_entropy': quality['mean_entropy'],
            'silhouette': quality['silhouette'],
        })

    summary_df = pd.DataFrame(summary_data)

    # Find best k (highest core %)
    best_k = summary_df.loc[summary_df['pct_core'].idxmax(), 'k']

    if verbose:
        print(f"\n{'='*60}")
        print("K SELECTION SUMMARY")
        print('='*60)
        print(summary_df.to_string(index=False))
        print(f"\nRecommended k: {best_k} (highest % core assignments)")

    return {
        'best_k': int(best_k),
        'k_values': k_range,
        'summary_df': summary_df,
        'clustering_by_k': results_by_k,
        'embryo_ids': embryo_ids,
    }


def plot_k_selection(
    k_results: Dict[str, Any],
    figsize: tuple = (16, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot quality metrics across k values to help select optimal k.
    
    Creates a 2x2 grid:
    - Top-left: Membership % (core/uncertain/outlier) vs k
    - Top-right: Mean max_p vs k  
    - Bottom-left: Mean entropy vs k
    - Bottom-right: Silhouette score vs k
    
    Parameters
    ----------
    k_results : Dict
        Output from evaluate_k_range()
    figsize : tuple
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    summary_df = k_results['summary_df']
    k_values = summary_df['k'].values
    best_k = k_results['best_k']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Membership % vs k
    ax = axes[0, 0]
    ax.plot(k_values, summary_df['pct_core'], 'o-', color='green', 
            linewidth=2.5, markersize=10, label='Core')
    ax.plot(k_values, summary_df['pct_uncertain'], 's-', color='orange', 
            linewidth=2.5, markersize=10, label='Uncertain')
    ax.plot(k_values, summary_df['pct_outlier'], '^-', color='red', 
            linewidth=2.5, markersize=10, label='Outlier')
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5, 
               label=f'Best k={best_k}')
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Membership Quality vs K', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 2. Mean max_p vs k
    ax = axes[0, 1]
    ax.plot(k_values, summary_df['mean_max_p'], 'o-', color='steelblue', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Threshold (0.5)')
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Max Posterior', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Confidence vs K', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Mean entropy vs k
    ax = axes[1, 0]
    ax.plot(k_values, summary_df['mean_entropy'], 'o-', color='coral', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Assignment Ambiguity vs K (lower = better)', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    
    # 4. Silhouette score vs k
    ax = axes[1, 1]
    ax.plot(k_values, summary_df['silhouette'], 'o-', color='purple', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Separation vs K (higher = better)', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'K Selection Analysis\nRecommended k = {best_k}', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def run_k_selection_pipeline(
    D: np.ndarray,
    embryo_ids: List[str],
    df: pd.DataFrame,
    k_range: List[int] = [2, 3, 4, 5, 6],
    n_bootstrap: int = 100,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete K selection workflow.
    
    1. Evaluate all k values (NO filtering)
    2. Plot comparison metrics
    3. Recommend best k
    4. Return results for chosen k
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    embryo_ids : List[str]
        Embryo IDs
    df : pd.DataFrame
        Trajectory data (for plotting)
    k_range : List[int]
        K values to test
    n_bootstrap : int
        Bootstrap iterations
    output_dir : Path, optional
        Directory for output files
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict
        - 'k_results': full evaluation results
        - 'best_k': recommended k
        - 'best_results': bootstrap results for best k
        - 'summary_df': comparison DataFrame
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("K SELECTION PIPELINE (No Filtering)")
    print("="*70)
    print(f"\nTesting k values: {k_range}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Embryos: {len(embryo_ids)}")
    print("="*70)
    
    # Step 1: Evaluate all k values
    k_results = evaluate_k_range(
        D=D,
        embryo_ids=embryo_ids,
        k_range=k_range,
        n_bootstrap=n_bootstrap,
        verbose=verbose
    )
    
    # Step 2: Plot comparison
    fig = plot_k_selection(
        k_results,
        save_path=output_dir / 'k_selection_metrics.png' if output_dir else None
    )
    plt.show()
    
    # Step 3: Save summary
    if output_dir:
        k_results['summary_df'].to_csv(output_dir / 'k_selection_summary.csv', index=False)
        print(f"\n✓ Saved summary: {output_dir / 'k_selection_summary.csv'}")
    
    # Step 4: Return best k results
    best_k = k_results['best_k']
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION: k = {best_k}")
    print(f"{'='*70}")
    print(f"\nNext step: Run consensus pipeline with k={best_k} and filtering enabled")
    print(f"  results = run_consensus_pipeline(D, embryo_ids, k={best_k}, ...)")
    
    return {
        'k_results': k_results,
        'best_k': best_k,
        'best_bootstrap_results': k_results['metrics'][best_k]['bootstrap_results'],
        'best_posteriors': k_results['metrics'][best_k]['posteriors'],
        'best_classification': k_results['metrics'][best_k]['classification'],
        'summary_df': k_results['summary_df'],
    }


# ============================================================================
# ALTERNATIVE: TWO-PHASE PIPELINE
# ============================================================================

def run_two_phase_pipeline(
    D: np.ndarray,
    embryo_ids: List[str],
    k_range: List[int] = [2, 3, 4, 5, 6],
    n_bootstrap: int = 100,
    iqr_multiplier: float = 1.5,
    posterior_threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Two-phase consensus clustering pipeline.
    
    Phase 1: K Selection (no filtering)
    - Evaluate k_range with bootstrap clustering
    - Select best k based on quality metrics
    
    Phase 2: Final Clustering (with filtering)
    - Run consensus pipeline with best k
    - Apply Stage 1 (IQR distance) + Stage 2 (posterior) filtering
    
    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    embryo_ids : List[str]
        Embryo IDs
    k_range : List[int]
        K values to test in Phase 1
    n_bootstrap : int
        Bootstrap iterations
    iqr_multiplier : float
        IQR multiplier for Stage 1 filtering (Phase 2)
    posterior_threshold : float
        Posterior threshold for Stage 2 filtering (Phase 2)
    verbose : bool
        Print progress
        
    Returns
    -------
    results : Dict
        - 'phase1_results': K selection results
        - 'phase2_results': Final consensus pipeline results
        - 'best_k': Selected k value
    """
    from .consensus_pipeline import run_consensus_pipeline
    
    print("="*70)
    print("TWO-PHASE CONSENSUS CLUSTERING PIPELINE")
    print("="*70)
    
    # =========================================================================
    # PHASE 1: K Selection (no filtering)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: K SELECTION (No Filtering)")
    print("="*70)
    
    phase1_results = evaluate_k_range(
        D=D,
        embryo_ids=embryo_ids,
        k_range=k_range,
        n_bootstrap=n_bootstrap,
        verbose=verbose
    )
    
    best_k = phase1_results['best_k']
    
    # Plot k selection
    fig = plot_k_selection(phase1_results)
    plt.show()
    
    print(f"\n✓ Phase 1 complete. Best k = {best_k}")
    
    # =========================================================================
    # PHASE 2: Final Clustering (with filtering)
    # =========================================================================
    print("\n" + "="*70)
    print(f"PHASE 2: FINAL CLUSTERING (k={best_k}, with filtering)")
    print("="*70)
    
    phase2_results = run_consensus_pipeline(
        D=D,
        embryo_ids=embryo_ids,
        k=best_k,
        n_bootstrap=n_bootstrap,
        enable_stage1_filtering=True,
        enable_stage2_filtering=True,
        stage1_method='iqr',  # Use IQR distance filtering
        iqr_multiplier=iqr_multiplier,
        posterior_threshold=posterior_threshold,
        k_highlight=[best_k - 1, best_k, best_k + 1],
        verbose=verbose
    )
    
    print(f"\n✓ Phase 2 complete.")
    print(f"  Initial embryos: {len(embryo_ids)}")
    print(f"  After Stage 1: {len(phase2_results['embryo_ids_after_stage1'])}")
    print(f"  Final embryos: {len(phase2_results['final_embryo_ids'])}")
    
    return {
        'phase1_results': phase1_results,
        'phase2_results': phase2_results,
        'best_k': best_k,
    }


# ============================================================================
# FILE-BASED K SELECTION WITH PLOTS & CLUSTERING RESULTS
# ============================================================================

def run_k_selection_with_plots(
    df: pd.DataFrame,
    D: np.ndarray,
    embryo_ids: List[str],
    output_dir: Path,
    plotting_metrics: List[str] = None,
    k_range: List[int] = [2, 3, 4, 5, 6],
    n_bootstrap: int = 100,
    method: str = 'hierarchical',
    x_col: str = 'predicted_stage_hpf',
    metric_labels: Optional[Dict[str, str]] = None,
    enable_stage1_filtering: bool = True,
    stage1_method: str = 'iqr',
    iqr_multiplier: float = 2,
    k_neighbors: int = 5,
    filtering_hist_bins: int = 30,
    generate_cluster_flow: bool = True,
    cluster_flow_k_range: Optional[List[int]] = None,
    cluster_flow_title: str = "Cluster Flow Across k Values",
    cluster_flow_filename: str = "cluster_flow_sankey.html",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete k-selection pipeline with trajectory plots and cluster assignments.

    Evaluates multiple k values and generates:
    1. Individual membership trajectory plots for each k
    2. Summary comparison plot across all k values
    3. Summary metrics CSV
    4. Cluster assignments CSV (embryo_id + clustering_k_N columns)
    5. Full results pickle
    6. Optional Stage 1 IQR filtering histogram and outlier list

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame with embryo_id, predicted_stage_hpf, and metric columns
    D : np.ndarray
        Distance matrix (n × n)
    embryo_ids : List[str]
        Embryo identifiers
    output_dir : Path
        Directory to save all output files
    plotting_metrics : List[str], optional
        Which metrics to plot (default: curvature + body length)
    k_range : List[int]
        K values to evaluate (default: [2, 3, 4, 5, 6])
    n_bootstrap : int
        Bootstrap iterations per k (default: 100)
    method : str, default='hierarchical'
        Clustering method: 'hierarchical' or 'kmedoids'
    x_col : str
        Column name for x-axis (default: 'predicted_stage_hpf')
    metric_labels : Dict[str, str], optional
        Pretty labels for metrics in plots
    enable_stage1_filtering : bool, default=True
        If True, apply Stage 1 IQR filtering before k selection
    stage1_method : str, default='iqr'
        Stage 1 filtering method: 'iqr' (median distance) or 'knn' (k-NN IQR)
    iqr_multiplier : float, default=2
        IQR multiplier used for filtering threshold
    k_neighbors : int, default=5
        Number of neighbors for k-NN IQR filtering (stage1_method='knn')
    filtering_hist_bins : int, default=30
        Number of bins for the Stage 1 filtering histogram
    generate_cluster_flow : bool, default=True
        If True, generate a Sankey diagram showing cluster flow across k
    cluster_flow_k_range : List[int], optional
        K values to include in the flow diagram (default: all)
    cluster_flow_title : str, default="Cluster Flow Across k Values"
        Title for the flow diagram
    cluster_flow_filename : str, default="cluster_flow_sankey.html"
        Output filename for the flow diagram (saved in output_dir)
    verbose : bool
        Print progress

    Returns
    -------
    results : Dict
        K-selection results with keys:
        - 'k_values': list of k tested
        - 'metrics': Dict[k] → quality metrics for each k
        - 'summary_df': DataFrame with comparison
        - 'best_k': recommended k

    Output Files
    -----------
    In output_dir/:
    - k{N}_membership_trajectories.png : Trajectory plot for each k
    - k_selection_comparison.png : 2x2 summary metrics plot
    - k_selection_summary.csv : Metrics table
    - cluster_assignments.csv : embryo_id + clustering_k_N columns
    - k_results.pkl : Full results object (pickle)
    - stage1_{method}_iqr_histogram.png : Stage 1 filtering histogram (if enabled)
    - stage1_{method}_outliers.csv : Stage 1 outlier list (if enabled)
    - cluster_flow_sankey.html : Cluster flow Sankey diagram (if enabled)
    """
    from ..viz.plotting import plot_cluster_flow
    from analyze.viz.plotting import plot_feature_over_time
    from ..qc import identify_outliers, identify_embryo_outliers_iqr, filter_data_and_ids
    import pickle

    # Default plotting metrics
    if plotting_metrics is None:
        plotting_metrics = ['baseline_deviation_normalized', 'total_length_um']

    # Default metric labels
    if metric_labels is None:
        metric_labels = {
            'baseline_deviation_normalized': 'Curvature (normalized)',
            'total_length_um': 'Body Length (μm)',
            'aspect_ratio': 'Aspect Ratio',
            'centroid_velocity_um_per_hpf': 'Centroid Velocity (μm/hpf)',
        }

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("="*70)
        print("K SELECTION WITH TRAJECTORY PLOTS")
        print("="*70)
        print(f"Output directory: {output_dir}")
        print(f"Clustering method: {method}")
        print(f"Plotting metrics: {plotting_metrics}")
        print(f"K range: {k_range}")
        if enable_stage1_filtering:
            extra = f"{stage1_method}, iqr_multiplier={iqr_multiplier}"
            if stage1_method == 'knn':
                extra += f", k_neighbors={k_neighbors}"
            print(f"Stage 1 filtering: Enabled ({extra})")
        else:
            print("Stage 1 filtering: Disabled")
        print("="*70)

    # =========================================================================
    # STEP 0: Optional Stage 1 IQR filtering (before k selection)
    # =========================================================================
    stage1_results = None
    D_filtered = D
    embryo_ids_filtered = embryo_ids

    if enable_stage1_filtering:
        if verbose:
            print("\n" + "="*70)
            method_name = "IQR Distance Filtering" if stage1_method == 'iqr' else "k-NN IQR Filtering"
            print(f"STAGE 1: {method_name}")
            print("="*70)

        if stage1_method == 'iqr':
            outlier_ids, inlier_ids, stage1_info = identify_outliers(
                D,
                embryo_ids,
                method='iqr',
                threshold=iqr_multiplier,
                verbose=verbose
            )
            q1, q3 = np.percentile(stage1_info['median_distances'], [25, 75])
            iqr = q3 - q1
            stage1_results = {
                'method': 'iqr',
                'outlier_indices': stage1_info['outlier_indices'],
                'outlier_ids': outlier_ids,
                'kept_indices': stage1_info['inlier_indices'],
                'kept_ids': inlier_ids,
                'median_distances': stage1_info['median_distances'],
                'threshold': stage1_info['threshold'],
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
            }
            distances = stage1_info['median_distances']
            threshold = stage1_info['threshold']
            x_label_hist = "Median distance to all embryos"
        elif stage1_method == 'knn':
            stage1_results = identify_embryo_outliers_iqr(
                D,
                embryo_ids,
                iqr_multiplier=iqr_multiplier,
                k_neighbors=k_neighbors,
                verbose=verbose
            )
            stage1_results['method'] = 'knn'
            distances = stage1_results['knn_distances']
            threshold = stage1_results['threshold']
            effective_k = min(k_neighbors, len(embryo_ids) - 1)
            x_label_hist = f"Mean k-NN distance (k={effective_k})"
        else:
            raise ValueError("stage1_method must be 'iqr' or 'knn'")

        D_filtered, embryo_ids_filtered = filter_data_and_ids(
            D, embryo_ids, stage1_results['kept_indices']
        )

        # Histogram plot of filtering threshold
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(distances, bins=filtering_hist_bins, color='#4C78A8', alpha=0.8, edgecolor='white')
        ax.axvline(threshold, color='#D62728', linestyle='--', linewidth=2,
                   label=f"Threshold = {threshold:.2f}")
        ax.set_xlabel(x_label_hist)
        ax.set_ylabel("Embryo count")
        if stage1_method == 'iqr':
            title_text = f"Stage 1 IQR Filtering (multiplier={iqr_multiplier})"
        else:
            title_text = f"Stage 1 k-NN IQR Filtering (k={effective_k}, multiplier={iqr_multiplier})"
        ax.set_title(title_text)
        ax.legend(frameon=False)
        fig.tight_layout()

        hist_path = output_dir / f"stage1_{stage1_method}_iqr_histogram.png"
        fig.savefig(hist_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        outliers_csv = output_dir / f"stage1_{stage1_method}_outliers.csv"
        outlier_distances = distances[stage1_results['outlier_indices']] if len(stage1_results['outlier_indices']) > 0 else []
        outliers_df = pd.DataFrame({
            'embryo_id': stage1_results['outlier_ids'],
            'distance': outlier_distances,
        })
        outliers_df.to_csv(outliers_csv, index=False)

        stage1_results['histogram_path'] = str(hist_path)
        stage1_results['outliers_csv'] = str(outliers_csv)

        if verbose:
            print(f"\n✓ Stage 1 filtering complete.")
            print(f"  Kept embryos: {len(embryo_ids_filtered)} / {len(embryo_ids)}")
            print(f"  Histogram: {hist_path}")
            print(f"  Outliers CSV: {outliers_csv}")

    # =========================================================================
    # STEP 1: Evaluate all k values
    # =========================================================================
    k_results = evaluate_k_range(
        D=D_filtered,
        embryo_ids=embryo_ids_filtered,
        k_range=k_range,
        n_bootstrap=n_bootstrap,
        method=method,
        verbose=verbose
    )
    if stage1_results is not None:
        k_results['stage1_filtering'] = stage1_results

    # =========================================================================
    # STEP 2: Generate trajectory plots for each k
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("GENERATING TRAJECTORY PLOTS")
        print("="*70)

    for k in k_range:
        if verbose:
            print(f"\nGenerating plots for k={k}...")

        # Get results for this k
        clustering_info = k_results['clustering_by_k'][k]
        classification = clustering_info['classification']

        # Use Phase 1 cluster assignments (method-agnostic, no dendrogram needed)
        cluster_labels = clustering_info['assignments']['cluster_labels']
        cluster_map = dict(zip(embryo_ids_filtered, cluster_labels))

        # Prepare DataFrame for this k
        df_k = df[df['embryo_id'].isin(embryo_ids_filtered)].copy()
        df_k['cluster'] = df_k['embryo_id'].map(cluster_map)
        df_k = add_membership_column(df_k, classification, column_name='membership')

        # Generate trajectory plot
        try:
            fig = plot_feature_over_time(
                df_k,
                features=plotting_metrics,
                time_col=x_col,
                id_col='embryo_id',
                color_by='membership',
                facet_col='cluster',
                title=f'k={k}: Membership Quality by Cluster',
                backend='matplotlib',
                bin_width=2.0,
            )

            # Save plot
            fig_path = output_dir / f'k{k}_membership_trajectories.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            if verbose:
                print(f"  ✓ Saved: {fig_path}")
            plt.close(fig)

        except Exception as e:
            print(f"  ⚠ Plot generation failed for k={k}: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # STEP 3: Generate summary comparison plot
    # =========================================================================
    if verbose:
        print(f"\nGenerating summary comparison plot...")

    fig = plot_k_selection(k_results)
    summary_fig_path = output_dir / 'k_selection_comparison.png'
    plt.savefig(summary_fig_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"  ✓ Saved: {summary_fig_path}")
    plt.close(fig)

    # =========================================================================
    # STEP 4: Save summary metrics CSV
    # =========================================================================
    if verbose:
        print(f"Saving summary metrics...")

    summary_csv_path = output_dir / 'k_selection_summary.csv'
    k_results['summary_df'].to_csv(summary_csv_path, index=False)
    if verbose:
        print(f"  ✓ Saved: {summary_csv_path}")

    # =========================================================================
    # STEP 5: Create and save cluster assignments CSV
    # =========================================================================
    if verbose:
        print(f"Saving cluster assignments...")

    assignments_data = {'embryo_id': k_results['embryo_ids']}
    for k in k_range:
        cluster_labels = k_results['clustering_by_k'][k]['assignments']['cluster_labels']
        assignments_data[f'clustering_k_{k}'] = cluster_labels

    assignments_df = pd.DataFrame(assignments_data)
    assignments_csv_path = output_dir / 'cluster_assignments.csv'
    assignments_df.to_csv(assignments_csv_path, index=False)
    if verbose:
        print(f"  ✓ Saved: {assignments_csv_path}")

    # =========================================================================
    # STEP 6: Save full results pickle
    # =========================================================================
    if verbose:
        print(f"Saving full results pickle...")

    pkl_path = output_dir / 'k_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(k_results, f)
    if verbose:
        print(f"  ✓ Saved: {pkl_path}")

    # =========================================================================
    # STEP 7: Optional cluster flow diagram (Sankey)
    # =========================================================================
    flow_path = None
    if generate_cluster_flow:
        try:
            flow_path = output_dir / cluster_flow_filename
            plot_cluster_flow(
                k_results,
                k_range=cluster_flow_k_range,
                title=cluster_flow_title,
                output_path=flow_path,
            )
            k_results['cluster_flow_path'] = str(flow_path)
        except Exception as e:
            flow_path = None
            if verbose:
                print(f"  ⚠ Cluster flow plot failed: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("K SELECTION COMPLETE")
        print("="*70)
        print(f"\nBest k: {k_results['best_k']}")
        print(f"\nOutput files:")
        print(f"  - Trajectory plots: k{k_range[0]}_membership_trajectories.png → k{k_range[-1]}_membership_trajectories.png")
        print(f"  - Summary metrics: k_selection_comparison.png")
        print(f"  - Metrics table: k_selection_summary.csv")
        print(f"  - Cluster assignments: cluster_assignments.csv")
        print(f"  - Full results: k_results.pkl")
        if stage1_results is not None:
            hist_name = Path(stage1_results['histogram_path']).name
            outliers_name = Path(stage1_results['outliers_csv']).name
            print(f"  - Stage 1 histogram: {hist_name}")
            print(f"  - Stage 1 outliers: {outliers_name}")
        if flow_path is not None:
            print(f"  - Cluster flow: {Path(flow_path).name}")
        print(f"\nTo load cluster assignments:")
        print(f"  >>> df_clusters = pd.read_csv('{assignments_csv_path}')")
        print("="*70)

    return k_results
