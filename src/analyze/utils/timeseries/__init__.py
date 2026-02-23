"""
Time Series Utilities

Generic algorithms for time series analysis, including Dynamic Time Warping (DTW),
DTW Barycenter Averaging (DBA), and interpolation utilities.

This module provides domain-agnostic time series algorithms that can be used
across different analysis contexts (trajectory analysis, signal processing, etc.).

Submodules
==========
- dtw : Dynamic Time Warping distance computation
- dba : DTW Barycenter Averaging for consensus sequences
- interpolation : Time series interpolation and alignment

Functions
=========
DTW Distance:
- compute_dtw_distance : Compute DTW distance between two 1D sequences
- compute_dtw_distance_matrix : Compute pairwise DTW distances for multiple 1D sequences
- compute_md_dtw_distance_matrix : Compute pairwise multivariate DTW distances

DBA:
- dba : DTW Barycenter Averaging for computing consensus sequences

Interpolation:
- interpolate_to_common_grid : Interpolate trajectories to a common time grid
- pad_trajectories_for_plotting : Pad trajectories to uniform length with NaN
"""

from .dtw import (
    compute_dtw_distance,
    compute_dtw_distance_matrix,
    compute_md_dtw_distance_matrix,
    _dtw_multivariate_pair,
)

from .dba import dba

from .interpolation import (
    interpolate_to_common_grid,
    pad_trajectories_for_plotting,
)

__all__ = [
    # DTW
    'compute_dtw_distance',
    'compute_dtw_distance_matrix',
    'compute_md_dtw_distance_matrix',
    '_dtw_multivariate_pair',
    # DBA
    'dba',
    # Interpolation
    'interpolate_to_common_grid',
    'pad_trajectories_for_plotting',
]
