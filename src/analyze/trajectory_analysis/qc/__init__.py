"""
Quality Control Subpackage for Trajectory Analysis

Provides outlier detection and filtering functions for distance matrices.

Functions
=========
General Outlier Detection:
- identify_outliers : Detect outlier embryos based on median distance
- remove_outliers_from_distance_matrix : Convenience wrapper for removal

Two-Stage Filtering (for clustering pipelines):
- identify_embryo_outliers_iqr : Stage 1 k-NN IQR filtering (before clustering)
- filter_data_and_ids : Safe filtering maintaining index alignment
- identify_cluster_outliers_combined : Stage 2 cluster + posterior filtering

Example
-------
>>> from trajectory_analysis.qc import (
...     identify_outliers,
...     filter_data_and_ids,
...     identify_embryo_outliers_iqr
... )
>>> outliers, inliers, info = identify_outliers(D, embryo_ids, method='iqr')
"""

from .quality_control import (
    # General outlier detection
    identify_outliers,
    remove_outliers_from_distance_matrix,
    # Two-stage filtering
    identify_embryo_outliers_iqr,
    filter_data_and_ids,
    identify_cluster_outliers_combined,
)

__all__ = [
    # General outlier detection
    'identify_outliers',
    'remove_outliers_from_distance_matrix',
    # Two-stage filtering
    'identify_embryo_outliers_iqr',
    'filter_data_and_ids',
    'identify_cluster_outliers_combined',
]
