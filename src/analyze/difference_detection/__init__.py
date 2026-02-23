"""
Difference detection package for analyzing genotype effects and temporal patterns.

This package provides tools for:
1. Classification-based tests (logistic regression for discrimination)
2. Distribution-based tests (energy distance, MMD)
3. Horizon plots (time series comparisons)
4. Time matrix analysis (temporal correlation/prediction matrices)

Submodules
==========
- permutation_utils : Core permutation testing utilities (p-value, shuffles)
- distance_metrics : Distance functions (energy, MMD, mean distance, etc.)
- distribution_test : Distribution-based permutation tests
- classification_test : Binary classification tests
- classification_test_multiclass : Multiclass classification tests
- horizon_plots : Heatmap visualization utilities
- time_matrix : Temporal data reshaping and analysis
- pipelines : High-level orchestration workflows
"""

# Core utilities (conditionally import if available)
try:
    from . import horizon_plots
except ImportError:
    horizon_plots = None

try:
    from . import time_matrix
except ImportError:
    time_matrix = None

try:
    from . import pipelines
except ImportError:
    pipelines = None

# Permutation testing framework (NEW)
from . import permutation_utils
from . import distance_metrics
from . import distribution_test

# Classification tests
from .classification_test import (
    assign_group_labels,
    run_binary_classification_test,
    compute_timeseries_divergence,
)
from .classification_test_multiclass import (
    run_multiclass_classification_test,
    run_classification_test,
    extract_temporal_confusion_profile,
)

# Result containers
from .results import (
    MulticlassOVRResults,
    ComparisonSpec,
)

# Classification test visualization
from .classification_test_viz import (
    plot_auroc_with_null,
    plot_multiple_aurocs,
    plot_multiclass_ovr_aurocs,
    plot_feature_comparison_grid,
)

# Expose key functions at package level for convenience (if modules exist)
try:
    from .horizon_plots import (
        plot_horizon_grid,
        plot_single_horizon,
        plot_best_condition_map,
        compute_shared_colorscale,
    )
except ImportError:
    pass

try:
    from .time_matrix import (
        load_time_matrix_results,
        build_metric_matrices,
        align_matrix_times,
        compute_matrix_statistics,
        filter_matrices_by_time_range,
        interpolate_missing_times,
    )
except ImportError:
    pass

try:
    from .pipelines import (
        HorizonPlotContext,
        load_and_prepare_time_matrices,
        render_horizon_grid,
        summarise_bundles,
    )
except ImportError:
    pass

# Permutation testing utilities (NEW)
from .permutation_utils import (
    compute_pvalue,
    pool_shuffle,
    label_shuffle,
    PermutationResult,
)

# Distance metrics (NEW)
from .distance_metrics import (
    compute_energy_distance,
    compute_mmd,
    compute_mean_distance,
    compute_rbf_kernel,
    estimate_bandwidth_median,
)

# Distribution-based testing (NEW)
from .distribution_test import (
    permutation_test_distribution,
    permutation_test_energy,
    permutation_test_mmd,
)

# Legacy distribution imports (for backwards compatibility)
# TODO: Phase these out as code migrates to new API
try:
    from .distribution import (
        hotellings_t2_test,
        mmd_kernel_width_test,
        compute_mahalanobis_distance,
        compute_euclidean_distance,
    )
except ImportError:
    # If old distribution module is deleted, these won't be available
    pass

__all__ = [
    # Submodules
    'horizon_plots',
    'time_matrix',
    'pipelines',
    'permutation_utils',
    'distance_metrics',
    'distribution_test',
    # Horizon plots
    'plot_horizon_grid',
    'plot_single_horizon',
    'plot_best_condition_map',
    'compute_shared_colorscale',
    # Time matrix utilities
    'load_time_matrix_results',
    'build_metric_matrices',
    'align_matrix_times',
    'compute_matrix_statistics',
    'filter_matrices_by_time_range',
    'interpolate_missing_times',
    # Pipeline orchestration
    'HorizonPlotContext',
    'load_and_prepare_time_matrices',
    'render_horizon_grid',
    'summarise_bundles',
    # Permutation testing utilities (NEW)
    'compute_pvalue',
    'pool_shuffle',
    'label_shuffle',
    'PermutationResult',
    # Distance metrics (NEW)
    'compute_energy_distance',
    'compute_mmd',
    'compute_mean_distance',
    'compute_rbf_kernel',
    'estimate_bandwidth_median',
    # Distribution-based testing (NEW)
    'permutation_test_distribution',
    'permutation_test_energy',
    'permutation_test_mmd',
    # Classification tests
    'assign_group_labels',
    'run_binary_classification_test',
    'compute_timeseries_divergence',
    'run_multiclass_classification_test',
    'run_comparison_test',  # New API
    'extract_temporal_confusion_profile',
    # Result containers
    'MulticlassOVRResults',
    'ComparisonSpec',
    # Classification test visualization
    'plot_auroc_with_null',
    'plot_multiple_aurocs',
    'plot_multiclass_ovr_aurocs',
    'plot_feature_comparison_grid',
]
