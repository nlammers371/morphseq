"""Utility Functions for Trajectory Analysis

This subpackage contains utility functions for trajectory data manipulation,
PCA embedding, correlation analysis, and DTW utilities.

Modules
=======
trajectory_utils : Data extraction, interpolation, and preprocessing
pca : PCA embedding and transformation functions
correlation : Correlation analysis utilities
dtw_utils : DTW utility functions for preparing trajectory data
"""

# Trajectory utilities
from .trajectory_utils import (
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    interpolate_to_common_grid_multi_df,
    df_to_trajectories,
    extract_early_late_means,
    compute_trend_line,
    # Legacy API (deprecated)
    extract_trajectories,
    interpolate_trajectories,
    interpolate_to_common_grid,
    pad_trajectories_for_plotting,
)

# PCA utilities (canonical in analyze.utils.pca)
from analyze.utils.pca import (
    fit_pca_on_embeddings,
    transform_embeddings_to_pca,
    compute_wt_reference_by_time,
    subtract_wt_reference,
    fit_transform_pca,
)

# Correlation analysis
from .correlation import (
    test_anticorrelation,
)

# DTW utilities
from .dtw_utils import (
    prepare_multivariate_array,
    compute_trajectory_distances,
)


__all__ = [
    # Trajectory utilities (new API)
    'extract_trajectories_df',
    'interpolate_to_common_grid_df',
    'interpolate_to_common_grid_multi_df',
    'df_to_trajectories',
    'extract_early_late_means',
    'compute_trend_line',
    # Trajectory utilities (legacy)
    'extract_trajectories',
    'interpolate_trajectories',
    'interpolate_to_common_grid',
    'pad_trajectories_for_plotting',
    # PCA
    'fit_pca_on_embeddings',
    'transform_embeddings_to_pca',
    'compute_wt_reference_by_time',
    'subtract_wt_reference',
    'fit_transform_pca',
    # Correlation
    'test_anticorrelation',
    # DTW utilities
    'prepare_multivariate_array',
    'compute_trajectory_distances',
]
