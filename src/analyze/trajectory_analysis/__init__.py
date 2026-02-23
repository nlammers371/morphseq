"""
Trajectory Analysis Package

Comprehensive framework for DTW-based trajectory analysis, bootstrap consensus
clustering, and probabilistic quality assessment.

This package provides a DataFrame-centric API where time information (hpf - hours
post fertilization) travels with trajectory data throughout the entire pipeline,
eliminating time-axis misalignment issues common with array-based approaches.

Core Workflow
=============
1. Extract: Load data and filter by genotype → df_filtered [embryo_id, hpf, metric_value]
2. Align: Interpolate all trajectories to common hpf grid → df_interpolated
3. Convert: Extract arrays for DTW computation via df_to_trajectories() helper
4. Cluster: Compute DTW distances and bootstrap cluster
5. Analyze: Compute posterior probabilities and quality metrics
6. Plot: Visualize trajectories using df_interpolated with hpf column

This package unifies:
- DTW distance computation
- Trajectory processing and alignment (DataFrame-first API)
- Bootstrap consensus clustering
- Posterior probability analysis
- Membership quality classification
- Trajectory visualization

Examples
--------
>>> from src.analyze.trajectory_analysis import (
...     extract_trajectories_df,
...     interpolate_to_common_grid_df,
...     df_to_trajectories,
...     compute_dtw_distance_matrix,
...     run_bootstrap_hierarchical,
...     analyze_bootstrap_results,
...     classify_membership_2d,
...     plot_cluster_trajectories_df
... )
>>>
>>> # Extract and filter trajectories (DataFrame-centric)
>>> df_filtered = extract_trajectories_df(df, genotype_filter='wildtype')
>>> df_interpolated = interpolate_to_common_grid_df(df_filtered)
>>>
>>> # Convert to arrays for DTW (one-liner helper)
>>> trajectories, ids, grid = df_to_trajectories(df_interpolated)
>>>
>>> # Compute DTW distances
>>> D = compute_dtw_distance_matrix(trajectories, window=5)
>>>
>>> # Bootstrap clustering
>>> bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=ids, n_bootstrap=100)
>>>
>>> # Analyze posteriors
>>> posteriors = analyze_bootstrap_results(bootstrap_results)
>>>
>>> # Classify membership quality
>>> classification = classify_membership_2d(
...     posteriors['max_p'],
...     posteriors['log_odds_gap'],
...     posteriors['modal_cluster'],
...     embryo_ids=posteriors['embryo_ids']
... )
>>>
>>> # Visualize using DataFrame (preserves time alignment)
>>> fig = plot_cluster_trajectories_df(
...     df_interpolated,
...     posteriors['modal_cluster'],
...     embryo_ids=posteriors['embryo_ids'],
...     show_mean=True
... )

Migration from v0.1.x
=====================
The API changed significantly in v0.2.0 to fix time-axis alignment bugs.
Old array-based functions are deprecated but still functional with deprecation
warnings. See README.md for detailed migration guide.
"""

__version__ = '0.2.0'

# DTW & Distance Computation
from .distance import (
    compute_dtw_distance,
    compute_dtw_distance_matrix,
    prepare_multivariate_array,
    compute_md_dtw_distance_matrix,
    compute_trajectory_distances,
    dba,
)
from .io import compute_dtw_distance_from_df

# Trajectory Processing - NEW API (DataFrame-centric, v0.2.0+)
from .utilities import (
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    df_to_trajectories,
    compute_trend_line,
    # DEPRECATED (v0.1.x, kept for backward compatibility)
    extract_trajectories,
    interpolate_trajectories,
    interpolate_to_common_grid,
    pad_trajectories_for_plotting,
    extract_early_late_means,
    # Correlation Analysis
    test_anticorrelation,
)

# Projection
from .projection import (
    project_onto_reference_clusters,
    prepare_projection_arrays,
    project_onto_reference_clusters_from_distance,
    compute_cross_dtw_distance_matrix,
    assign_clusters_nearest_neighbor,
    assign_clusters_knn_posterior,
)

# Bootstrap Clustering (now in clustering subpackage)
from .clustering import (
    run_bootstrap_hierarchical,
    run_bootstrap_kmedoids,
    run_bootstrap_projection,
    run_bootstrap_projection_with_plots,
    compute_consensus_labels,
    get_cluster_assignments,
    compute_coassociation_matrix,
    coassociation_to_distance,
)

# Posterior Analysis (now in clustering subpackage)
from .clustering import (
    analyze_bootstrap_results,
    compute_assignment_posteriors,
    compute_quality_metrics,
    align_bootstrap_labels,
)

# Classification (now in clustering subpackage)
from .clustering import (
    classify_membership_2d,
    classify_membership_adaptive,
    get_classification_summary,
)

# Plotting - NEW API (DataFrame-centric, v0.2.0+)
from .viz.plotting import (
    plot_cluster_trajectories_df,
    plot_membership_trajectories_df,
    plot_membership_vs_k
)

# Plotting - DEPRECATED (v0.1.x, kept for backward compatibility)
from .viz.plotting import (
    plot_posterior_heatmap,
    plot_2d_scatter,
    plot_cluster_trajectories,
    plot_membership_trajectories
)

# Genotype Styling (Level 0 - Styling)
from .viz.styling import (
    extract_genotype_suffix,
    extract_genotype_prefix,
    get_color_for_genotype,
    sort_genotypes_by_suffix,
    build_genotype_style_config,
    format_genotype_label
)

# Faceted Plotting (Level 1 - Generic)
from .viz.plotting import (
    plot_cluster_flow,
)

# Pair Analysis functions removed (extracted to generic utils)
# Use: from src.analyze.utils.data_processing import get_trajectories_for_group

# Quality Control (Outlier Detection + Distance Filtering)
from .qc import (
    # General outlier detection
    identify_outliers,
    remove_outliers_from_distance_matrix,
    # Two-stage filtering
    identify_embryo_outliers_iqr,
    filter_data_and_ids,
    identify_cluster_outliers_combined,
)

# Consensus Pipeline (now in clustering subpackage)
from .clustering import (
    run_consensus_pipeline,
    create_filtering_log,
)

# K Selection Pipeline (now in clustering subpackage)
from .clustering import (
    evaluate_k_range,
    plot_k_selection,
    run_k_selection_pipeline,
    run_two_phase_pipeline,
    run_k_selection_with_plots,
)

# Dendrogram Visualization
from .viz import (
    generate_dendrograms,
    plot_dendrogram,  # Deprecated, kept for backward compatibility
    plot_dendrogram_with_categories,
    add_cluster_column,
    PASTEL_COLORS,
)

# 3D Plotting
from .viz.plotting import plot_3d_scatter

# PCA Embedding Utilities (now in utilities subpackage)
from .utilities import (
    fit_pca_on_embeddings,
    transform_embeddings_to_pca,
    compute_wt_reference_by_time,
    subtract_wt_reference,
    fit_transform_pca,
)

__all__ = [
    # DTW & Distance
    'compute_dtw_distance',
    'compute_dtw_distance_matrix',
    'prepare_multivariate_array',
    'compute_md_dtw_distance_matrix',
    'compute_trajectory_distances',
    'compute_dtw_distance_from_df',
    'dba',
    'project_onto_reference_clusters',
    'prepare_projection_arrays',
    'project_onto_reference_clusters_from_distance',
    'compute_cross_dtw_distance_matrix',
    'assign_clusters_nearest_neighbor',
    'assign_clusters_knn_posterior',

    # Trajectory Processing - NEW API (v0.2.0+)
    'extract_trajectories_df',
    'interpolate_to_common_grid_df',
    'df_to_trajectories',
    'compute_trend_line',

    # Trajectory Processing - DEPRECATED (v0.1.x)
    'extract_trajectories',
    'interpolate_trajectories',
    'interpolate_to_common_grid',
    'pad_trajectories_for_plotting',
    'extract_early_late_means',

    # Correlation analysis
    'test_anticorrelation',

    # Bootstrap clustering
    'run_bootstrap_hierarchical',
    'run_bootstrap_kmedoids',
    'run_bootstrap_projection',
    'run_bootstrap_projection_with_plots',
    'compute_consensus_labels',
    'get_cluster_assignments',
    'compute_coassociation_matrix',
    'coassociation_to_distance',

    # Posterior analysis
    'analyze_bootstrap_results',
    'compute_assignment_posteriors',
    'compute_quality_metrics',
    'align_bootstrap_labels',

    # Classification
    'classify_membership_2d',
    'classify_membership_adaptive',
    'get_classification_summary',

    # Plotting - NEW API (v0.2.0+)
    'plot_cluster_trajectories_df',
    'plot_membership_trajectories_df',
    'plot_membership_vs_k',
    'plot_cluster_flow',

    # Plotting - DEPRECATED (v0.1.x)
    'plot_posterior_heatmap',
    'plot_2d_scatter',
    'plot_cluster_trajectories',
    'plot_membership_trajectories',

    # Genotype Styling (Level 0)
    'extract_genotype_suffix',
    'extract_genotype_prefix',
    'get_color_for_genotype',
    'sort_genotypes_by_suffix',
    'build_genotype_style_config',
    'format_genotype_label',

    # Pair Analysis functions removed (see src.analyze.utils.data_processing)

    # Outlier Detection
    'identify_outliers',
    'remove_outliers_from_distance_matrix',

    # Distance Filtering (Two-Stage)
    'identify_embryo_outliers_iqr',
    'filter_data_and_ids',
    'identify_cluster_outliers_combined',

    # Consensus Pipeline
    'run_consensus_pipeline',
    'create_filtering_log',

    # K Selection Pipeline
    'evaluate_k_range',
    'plot_k_selection',
    'run_k_selection_pipeline',
    'run_two_phase_pipeline',
    'run_k_selection_with_plots',

    # Dendrogram Visualization
    'generate_dendrograms',
    'plot_dendrogram',  # Deprecated
    'plot_dendrogram_with_categories',
    'add_cluster_column',
    'PASTEL_COLORS',

    # 3D Plotting
    'plot_3d_scatter',

    # PCA Embedding Utilities
    'fit_pca_on_embeddings',
    'transform_embeddings_to_pca',
    'compute_wt_reference_by_time',
    'subtract_wt_reference',
    'fit_transform_pca',
]
