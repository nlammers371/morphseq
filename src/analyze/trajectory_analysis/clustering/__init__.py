"""
Clustering Subpackage

Bootstrap consensus clustering, posterior probability analysis, and k-selection pipelines.

Modules
-------
- bootstrap_clustering: Bootstrap hierarchical/k-medoids clustering
- cluster_posteriors: Posterior probability computation from bootstrap results
- cluster_classification: Membership quality classification (core/uncertain/outlier)
- consensus_pipeline: Two-stage filtering + consensus clustering pipeline
- k_selection: K-value selection and evaluation pipelines
- cluster_extraction: Cluster extraction utilities for k-selection results
"""

# Bootstrap Clustering
from .bootstrap_clustering import (
    run_bootstrap_hierarchical,
    run_bootstrap_kmedoids,
    run_bootstrap_projection,
    run_bootstrap_projection_with_plots,
    bootstrap_projection_assignments_from_distance,
    compute_consensus_labels,
    get_cluster_assignments,
    compute_coassociation_matrix,
    coassociation_to_distance,
)

# Posterior Analysis
from .cluster_posteriors import (
    analyze_bootstrap_results,
    compute_assignment_posteriors,
    compute_quality_metrics,
    align_bootstrap_labels,
)

# Classification
from .cluster_classification import (
    classify_membership_2d,
    classify_membership_adaptive,
    get_classification_summary,
)

# Consensus Pipeline
from .consensus_pipeline import (
    run_consensus_pipeline,
    create_filtering_log,
)

# K Selection Pipeline
from .k_selection import (
    evaluate_k_range,
    plot_k_selection,
    run_k_selection_pipeline,
    run_two_phase_pipeline,
    run_k_selection_with_plots,
    add_membership_column,
)

# Cluster Extraction
from .cluster_extraction import (
    extract_cluster_embryos,
    get_cluster_summary,
    map_clusters_to_phenotypes,
)

__all__ = [
    # Bootstrap Clustering
    'run_bootstrap_hierarchical',
    'run_bootstrap_kmedoids',
    'run_bootstrap_projection',
    'run_bootstrap_projection_with_plots',
    'bootstrap_projection_assignments_from_distance',
    'compute_consensus_labels',
    'get_cluster_assignments',
    'compute_coassociation_matrix',
    'coassociation_to_distance',
    # Posterior Analysis
    'analyze_bootstrap_results',
    'compute_assignment_posteriors',
    'compute_quality_metrics',
    'align_bootstrap_labels',
    # Classification
    'classify_membership_2d',
    'classify_membership_adaptive',
    'get_classification_summary',
    # Consensus Pipeline
    'run_consensus_pipeline',
    'create_filtering_log',
    # K Selection Pipeline
    'evaluate_k_range',
    'plot_k_selection',
    'run_k_selection_pipeline',
    'run_two_phase_pipeline',
    'run_k_selection_with_plots',
    'add_membership_column',
    # Cluster Extraction
    'extract_cluster_embryos',
    'get_cluster_summary',
    'map_clusters_to_phenotypes',
]
