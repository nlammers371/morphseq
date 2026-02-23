"""Distance Computation Algorithms

This subpackage contains distance computation algorithms for trajectory analysis,
including Dynamic Time Warping (DTW) and DTW Barycenter Averaging (DBA).

Generic DTW/DBA algorithms are imported from analyze.utils.timeseries.
Domain-specific functions are in the utilities subpackage.

Functions
=========
DTW Distance (generic from utils.timeseries):
- compute_dtw_distance : Compute DTW distance between two 1D sequences
- compute_dtw_distance_matrix : Compute pairwise DTW distances for multiple 1D sequences
- compute_md_dtw_distance_matrix : Compute pairwise MD-DTW distances

DTW Distance (domain-specific from utilities):
- prepare_multivariate_array : Convert DataFrame to 3D array for MD-DTW
- compute_trajectory_distances : High-level function to compute trajectory distances

DBA (from utils.timeseries):
- dba : DTW Barycenter Averaging for computing consensus sequences
"""

# Generic DTW functions from canonical location
try:
    from analyze.utils.timeseries.dtw import (
        compute_dtw_distance,
        compute_dtw_distance_matrix,
        compute_md_dtw_distance_matrix,
    )
    from analyze.utils.timeseries.dba import dba
except ModuleNotFoundError as exc:
    # Fallback for legacy path setups where `src/analyze` is on sys.path
    if exc.name not in {
        "analyze",
        "analyze.utils",
        "analyze.utils.timeseries",
        "analyze.utils.timeseries.dtw",
        "analyze.utils.timeseries.dba",
    }:
        raise
    from utils.timeseries.dtw import (
        compute_dtw_distance,
        compute_dtw_distance_matrix,
        compute_md_dtw_distance_matrix,
    )
    from utils.timeseries.dba import dba

# Domain-specific functions from utilities
from ..utilities.dtw_utils import (
    prepare_multivariate_array,
    compute_trajectory_distances,
)

__all__ = [
    'compute_dtw_distance',
    'compute_dtw_distance_matrix',
    'prepare_multivariate_array',
    'compute_md_dtw_distance_matrix',
    'compute_trajectory_distances',
    'dba',
]
