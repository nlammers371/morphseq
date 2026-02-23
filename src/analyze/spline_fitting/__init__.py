"""Spline fitting and trajectory analysis tools.

This module provides comprehensive tools for fitting smooth curves through
trajectory data using Local Principal Curves (LPC), with support for:
- Bootstrap uncertainty estimation
- Curve segmentation and projection
- Alignment and comparison
- Trajectory dynamics analysis
- Visualization

Quick Start
-----------
Fit a spline through trajectory data:

    >>> from src.analyze.spline_fitting import LocalPrincipalCurve
    >>> lpc = LocalPrincipalCurve(bandwidth=0.5)
    >>> lpc.fit(points, start_points=start_point)
    >>> curve = lpc.cubic_splines[0]

Fit with bootstrap uncertainty:

    >>> from src.analyze.spline_fitting import spline_fit_wrapper
    >>> wt_spline = spline_fit_wrapper(
    ...     wt_df, pca_cols=['PC1', 'PC2', 'PC3'], n_bootstrap=100
    ... )

Fit multiple splines by group:

    >>> all_splines = spline_fit_wrapper(
    ...     df, group_by='phenotype',
    ...     pca_cols=['PC1', 'PC2', 'PC3']
    ... )

Module Organization
-------------------
lpc_model
    Core LocalPrincipalCurve algorithm (standalone, no dependencies)

bootstrap
    Bootstrap spline fitting with group_by support

fitter
    Placeholder for future SplineFitter class (not yet implemented)

curve_ops
    Curve geometry operations (segmentation, projection, point mapping)

alignment
    Curve alignment functions (quaternion_alignment, legacy procrustes)

utils.spline_metrics
    Metrics for comparing splines (RMSE, direction consistency, dispersion)

dynamics
    Trajectory dynamics (journey simulation, developmental shifts)

viz
    Visualization (augmentors and convenience functions)

_compat
    Removed (no shims)
"""

# Core algorithm
from .lpc_model import LocalPrincipalCurve

# Bootstrap fitting
from .bootstrap import spline_fit_wrapper

# Curve operations
from .curve_ops import (
    split_spline,
    assign_points_to_segments,
    create_spline_segments_for_df,
    project_onto_plane,
    project_points_onto_reference_spline,
)

# Alignment
from .alignment import (
    quaternion_alignment,
    procrustes_alignment,  # LEGACY - not validated
)

# Metrics
from .utils.spline_metrics import (
    rmse,
    rmsd,
    segment_direction_consistency,
    calculate_dispersion_metrics,
    compute_dispersion,
)

# Dynamics
from .dynamics import (
    run_bootstrap_journeys,
    compute_developmental_shifts,
    summarize_journeys,
)

# Visualization
from .viz import (
    add_spline_to_fig,
    add_uncertainty_tube,
    plot_3d_with_spline,
)

__all__ = [
    # Core
    'LocalPrincipalCurve',

    # Bootstrap
    'spline_fit_wrapper',

    # Curve operations
    'split_spline',
    'assign_points_to_segments',
    'create_spline_segments_for_df',
    'project_onto_plane',
    'project_points_onto_reference_spline',

    # Alignment
    'quaternion_alignment',
    'procrustes_alignment',

    # Metrics
    'rmse',
    'rmsd',
    'segment_direction_consistency',
    'calculate_dispersion_metrics',
    'compute_dispersion',

    # Dynamics
    'run_bootstrap_journeys',
    'compute_developmental_shifts',
    'summarize_journeys',

    # Visualization
    'add_spline_to_fig',
    'add_uncertainty_tube',
    'plot_3d_with_spline',
]

__version__ = '0.1.0'
