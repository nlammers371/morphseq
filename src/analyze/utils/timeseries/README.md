# Generic Time Series Utilities

**Location:** `src/analyze/utils/timeseries/`

## Overview

Domain-agnostic utilities for time series analysis. These functions have **no dependencies** on trajectory analysis or morphology-specific code, making them reusable across different analysis contexts.

## Modules

### `dtw.py` - Dynamic Time Warping
- `compute_dtw_distance()`: Compute DTW distance between two 1D time series
- `compute_dtw_distance_matrix()`: Pairwise DTW distances for multiple 1D series
- `compute_md_dtw_distance_matrix()`: Multivariate DTW (MD-DTW) for multi-feature time series
- Uses Sakoe-Chiba band constraint for efficiency
- **NaN-aware**: Multivariate DTW handles missing features via scaled distance computation (see below)

### `dba.py` - DTW Barycentric Averaging
- `dba()`: Compute DTW Barycentric Average of multiple time series
- Iterative refinement to find optimal "average" sequence
- Useful for finding representative trajectories

### `interpolation.py` - Time Series Interpolation
- `interpolate_trajectory()`: Interpolate time series to common time grid
- Handles irregular sampling and missing data
- Supports multiple interpolation methods (linear, cubic, etc.)

## Usage Examples

### DTW Distance
```python
from src.analyze.utils.timeseries import compute_dtw_distance
import numpy as np

# Compare two time series
ts1 = np.array([1.0, 2.0, 3.0, 4.0])
ts2 = np.array([1.1, 2.2, 3.1, 4.2])

distance = compute_dtw_distance(ts1, ts2)
print(f"DTW distance: {distance:.4f}")
```

### DTW Barycentric Average
```python
from src.analyze.utils.timeseries import dba

# Average multiple time series
trajectories = [
    np.array([1.0, 2.0, 3.0, 4.0]),
    np.array([1.1, 2.1, 2.9, 4.1]),
    np.array([0.9, 2.2, 3.1, 3.9]),
]

average_trajectory = dba(trajectories, max_iter=10)
```

### Interpolation
```python
from src.analyze.utils.timeseries import interpolate_trajectory

# Interpolate to common time grid
times = np.array([0.0, 1.0, 2.5, 4.0])
values = np.array([10.0, 15.0, 18.0, 20.0])
new_times = np.linspace(0, 4, 100)

interpolated = interpolate_trajectory(times, values, new_times, method='cubic')
```

## NaN Handling in Multivariate DTW

The multivariate DTW functions (`_nan_aware_cost_matrix`, `_dtw_multivariate_pair`, `compute_md_dtw_distance_matrix`) handle missing data gracefully:

- **Partial feature NaNs**: When some features are NaN at a timepoint pair, distance is computed using valid features and scaled by `sqrt(n_features / valid_counts)`. This assumes equal variance across features.
- **Full timepoint NaNs**: When all features are NaN at a timepoint, that cell has `inf` cost. If this occurs at the start or end of a series, the DTW path is blocked and the distance is `inf`.

**Best practice**: Trim time series to remove leading/trailing all-NaN timepoints before computing DTW distances. The NaN-aware scaling is designed for *partial* missing data (e.g., one feature missing at some timepoints), not for entirely missing observations.

Unit tests for NaN handling are in `src/analyze/_tests/test_dtw.py`.

## Design Principles

1. **Domain Agnostic**: No assumptions about embryo trajectories, morphology, or biology
2. **Reusable**: Can be imported by any analysis module
3. **Well-Tested**: Comprehensive unit tests in `tests/`
4. **Documented**: Clear docstrings with parameter descriptions and examples

## Migration Notes

These utilities were extracted from `trajectory_analysis/` during the deep restructuring (Phase 1-2, commit d1f411ee). Code that previously imported from trajectory-specific locations should now import from `utils.timeseries`.

### Before
```python
from src.analyze.trajectory_analysis.utilities.dtw import compute_dtw_distance
```

### After
```python
from src.analyze.utils.timeseries import compute_dtw_distance
```

## Related Modules

- **`viz/plotting/`**: Generic time series plotting (domain-agnostic visualization)
- **`trajectory_analysis/`**: Domain-specific trajectory analysis (uses these utilities)
