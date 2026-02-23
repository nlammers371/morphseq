# DBA Real Path Implementation Plan
**Date**: 2026-01-03
**Author**: Claude Code
**Status**: Ready for Implementation

---

## TL;DR - What To Do

1. Add `return_path=False` param to `compute_dtw_distance()` in `dtw_distance.py`
2. Fix 2 files using fake diagonal paths (`plotting.py`, `plotting_faceted.py`)
3. Add `trend_statistic='dba'` option to `facetted_plotting.py`
4. Run tests in `results/mcolon/20260103_dba_refactor/`

---

## Problem Statement

DBA (DTW Barycenter Averaging) is already implemented in the codebase, but it uses **fake diagonal paths** instead of real DTW alignment paths. This defeats the purpose of DTW-based averaging.

### Current Broken Code (plotting.py:274-279)
```python
def dtw_func(seq1, seq2):
    dist = compute_dtw_distance(seq1, seq2, window=dtw_window)
    # Create approximate path (diagonal alignment) ← THIS IS WRONG!
    min_len = min(len(seq1), len(seq2))
    path = [(i, i) for i in range(min_len)]  # ← FAKE DIAGONAL PATH!
    return path, dist
```

### What Real DTW Path Looks Like
```python
# Real DTW path (with warping to handle temporal shifts):
[(0,0), (1,0), (2,1), (3,2), (4,2), (5,3), ...]  # Can repeat indices!

# Fake diagonal path (current broken implementation):
[(0,0), (1,1), (2,2), (3,3), ...]  # Just diagonal = linear interpolation
```

### Why This Matters
- Real DTW path handles temporal shifts, speed variations, different trajectory lengths
- Fake diagonal path = just linear interpolation = loses all DTW benefits
- Current DBA is basically broken

---

## Implementation Steps

### STEP 1: Add `return_path` to compute_dtw_distance()

**File**: `src/analyze/trajectory_analysis/dtw_distance.py`

**What to change**: Add optional `return_path=False` parameter (backward compatible)

#### 1a. Update function signature (line 29)

**BEFORE:**
```python
def compute_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: int = 3,
    normalize: bool = False
) -> float:
```

**AFTER:**
```python
def compute_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: int = 3,
    normalize: bool = False,
    return_path: bool = False,  # NEW
) -> Union[float, Tuple[float, List[Tuple[int, int]]]]:
```

#### 1b. Add import at top of file
```python
from typing import List, Tuple, Optional, Union  # Add Union
```

#### 1c. Update docstring (add after existing params)
```python
    return_path : bool, default=False
        If True, also return the DTW alignment path.

    Returns
    -------
    float or Tuple[float, List[Tuple[int, int]]]
        If return_path=False: DTW distance (float)
        If return_path=True: (distance, path) where path is list of (i, j) tuples
```

#### 1d. Add path return logic (after line 109, before the final return)

**FIND THIS CODE (around line 109):**
```python
    distance = dtw_matrix[n, m]

    # Normalize by path length if requested (length-independent metric)
    if normalize and not np.isinf(distance):
        distance = distance / (n + m)

    return float(distance)
```

**REPLACE WITH:**
```python
    distance = dtw_matrix[n, m]

    # NEW: Optional path backtracking
    if return_path:
        path = _backtrack_dtw_path(dtw_matrix, n, m)
        if normalize and not np.isinf(distance):
            distance = distance / (n + m)
        return float(distance), path

    # Existing behavior (unchanged)
    if normalize and not np.isinf(distance):
        distance = distance / (n + m)

    return float(distance)
```

#### 1e. Add helper function (add BEFORE compute_dtw_distance, around line 28)

```python
def _backtrack_dtw_path(
    dtw_matrix: np.ndarray,
    n: int,
    m: int
) -> List[Tuple[int, int]]:
    """
    Backtrack through DTW cost matrix to recover optimal alignment path.

    Parameters
    ----------
    dtw_matrix : np.ndarray
        The accumulated cost matrix from DTW computation
    n : int
        Length of first sequence
    m : int
        Length of second sequence

    Returns
    -------
    List[Tuple[int, int]]
        Alignment path as list of (i, j) tuples from start to end.
        i indexes into seq1, j indexes into seq2.
    """
    path = []
    i, j = n, m

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))  # 0-indexed

        # Find which direction we came from (min cost)
        candidates = [
            (dtw_matrix[i - 1, j], i - 1, j),      # insertion (move in seq1 only)
            (dtw_matrix[i, j - 1], i, j - 1),      # deletion (move in seq2 only)
            (dtw_matrix[i - 1, j - 1], i - 1, j - 1)  # match (move in both)
        ]
        _, i, j = min(candidates, key=lambda x: x[0])

    # Handle edge cases: walk remaining steps along edges
    while i > 0:
        path.append((i - 1, 0))
        i -= 1
    while j > 0:
        path.append((0, j - 1))
        j -= 1

    path.reverse()
    return path
```

---

### STEP 2: Fix Existing DBA Callers

**Two files need 1-line changes each.**

#### 2a. Fix src/analyze/utils/plotting.py (lines 274-279)

**FIND:**
```python
                            def dtw_func(seq1, seq2):
                                dist = compute_dtw_distance(seq1, seq2, window=dtw_window)
                                # Create approximate path (diagonal alignment)
                                min_len = min(len(seq1), len(seq2))
                                path = [(i, i) for i in range(min_len)]
                                return path, dist
```

**REPLACE WITH:**
```python
                            def dtw_func(seq1, seq2):
                                dist, path = compute_dtw_distance(seq1, seq2, window=dtw_window, return_path=True)
                                return path, dist
```

#### 2b. Fix src/analyze/utils/plotting_faceted.py (lines 128-133)

**FIND:**
```python
                            def dtw_func(seq1, seq2):
                                dist = compute_dtw_distance(seq1, seq2, window=dtw_window)
                                # Create approximate path (diagonal alignment)
                                min_len = min(len(seq1), len(seq2))
                                path = [(i, i) for i in range(min_len)]
                                return path, dist
```

**REPLACE WITH:**
```python
                            def dtw_func(seq1, seq2):
                                dist, path = compute_dtw_distance(seq1, seq2, window=dtw_window, return_path=True)
                                return path, dist
```

---

### STEP 3: Add DBA Trend Line Option to facetted_plotting.py

**File**: `src/analyze/trajectory_analysis/facetted_plotting.py`

**Goal**: Allow `trend_statistic='dba'` in plotting functions (simple API extension).

#### 3a. Add helper function (after imports, around line 40)

```python
def _compute_dba_trend_line(
    trajectories: List[Dict],
    bin_width: float,
    smooth_sigma: float = 0.0,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Compute trend line using DTW Barycenter Averaging.

    Parameters
    ----------
    trajectories : List[Dict]
        List of trajectory dicts, each with 'times' and 'metrics' keys
    bin_width : float
        Width of time bins for output grid
    smooth_sigma : float
        Gaussian smoothing sigma for DBA

    Returns
    -------
    Tuple[Optional[List[float]], Optional[List[float]]]
        (bin_times, bin_values) matching compute_trend_line() API
    """
    from .dba import dba
    from .dtw_distance import compute_dtw_distance

    if not trajectories:
        return None, None

    # Extract metric values as series list
    series_list = [np.asarray(t['metrics'], dtype=float) for t in trajectories]

    # Skip if only one trajectory (DBA needs multiple)
    if len(series_list) < 2:
        # Fall back to just returning the single trajectory binned
        all_times = trajectories[0]['times']
        all_metrics = trajectories[0]['metrics']
        t_min, t_max = np.min(all_times), np.max(all_times)
        bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_values = np.interp(bin_centers, all_times, all_metrics)
        return bin_centers.tolist(), bin_values.tolist()

    # DTW function with real path
    def dtw_func(seq1, seq2):
        dist, path = compute_dtw_distance(seq1, seq2, window=3, return_path=True)
        return path, dist

    # Compute DBA consensus
    consensus = dba(
        series_list,
        dtw_func=dtw_func,
        max_iter=10,
        smooth_sigma=smooth_sigma,
        verbose=False
    )

    # Build time grid from original trajectories
    all_times = np.concatenate([t['times'] for t in trajectories])
    t_min, t_max = all_times.min(), all_times.max()

    # Interpolate consensus onto regular time grid
    consensus_times = np.linspace(t_min, t_max, len(consensus))
    bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Interpolate consensus values at bin centers
    bin_values = np.interp(bin_centers, consensus_times, consensus)

    return bin_centers.tolist(), bin_values.tolist()
```

#### 3b. Update trend line computation in _build_traces_for_cell() (around line 444-449)

**FIND:**
```python
        # 3. Trend Line
        bin_times, bin_stats = compute_trend_line(
            all_times, all_metrics, bin_width,
            statistic=trend_statistic,
            smooth_sigma=trend_smooth_sigma
        )
```

**REPLACE WITH:**
```python
        # 3. Trend Line
        if trend_statistic == 'dba':
            bin_times, bin_stats = _compute_dba_trend_line(
                trajectories,
                bin_width=bin_width,
                smooth_sigma=trend_smooth_sigma or 0.0,
            )
        else:
            bin_times, bin_stats = compute_trend_line(
                all_times, all_metrics, bin_width,
                statistic=trend_statistic,
                smooth_sigma=trend_smooth_sigma
            )
```

#### 3c. Update docstrings in plot_trajectories_faceted() and plot_multimetric_trajectories()

**FIND (in both functions):**
```python
    trend_statistic : str, default='median'
        Central tendency measure ('mean' or 'median')
```

**REPLACE WITH:**
```python
    trend_statistic : str, default='median'
        Central tendency measure: 'mean', 'median', or 'dba'.
        'dba' uses DTW Barycenter Averaging for temporal alignment -
        better for variable-length trajectories with temporal shifts.
```

---

### STEP 4: Create Test Files

#### 4a. Unit test for DTW path

**File**: `results/mcolon/20260103_dba_refactor/test_dtw_path.py`

```python
#!/usr/bin/env python
"""
Unit tests for DTW path extraction.
Run with: python test_dtw_path.py

Tests:
1. Identical sequences -> diagonal path
2. Different length sequences -> warped path
3. Backward compatibility -> default returns float only
"""
import numpy as np
import sys
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.dtw_distance import compute_dtw_distance


def test_dtw_path_basic():
    """Identical sequences should have diagonal path."""
    seq1 = np.array([1.0, 2.0, 3.0, 4.0])
    seq2 = np.array([1.0, 2.0, 3.0, 4.0])

    dist, path = compute_dtw_distance(seq1, seq2, return_path=True)

    expected_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert path == expected_path, f"Expected {expected_path}, got {path}"
    assert dist == 0.0, f"Expected 0 distance for identical sequences, got {dist}"
    print("✓ test_dtw_path_basic passed")


def test_dtw_path_with_shift():
    """Different length sequences should have warped path."""
    seq1 = np.array([0.0, 1.0, 2.0, 3.0])
    seq2 = np.array([1.0, 2.0, 3.0])

    dist, path = compute_dtw_distance(seq1, seq2, return_path=True)

    # Path should connect (0,0) to (3,2)
    assert len(path) >= 3, f"Path too short: {path}"
    assert path[0] == (0, 0), f"Path should start at (0,0), got {path[0]}"
    assert path[-1] == (3, 2), f"Path should end at (3,2), got {path[-1]}"

    # Path should NOT be purely diagonal (that would be wrong)
    is_diagonal = all(p[0] == p[1] for p in path)
    assert not is_diagonal, f"Path should NOT be diagonal for different length sequences: {path}"

    print(f"✓ test_dtw_path_with_shift passed")
    print(f"  Path: {path}")


def test_dtw_path_with_stretch():
    """Stretched sequence should have repeated indices in path."""
    # seq2 is seq1 but with middle value repeated (temporal stretch)
    seq1 = np.array([1.0, 2.0, 3.0])
    seq2 = np.array([1.0, 2.0, 2.0, 3.0])  # stretched in middle

    dist, path = compute_dtw_distance(seq1, seq2, return_path=True)

    assert path[0] == (0, 0), f"Path should start at (0,0)"
    assert path[-1] == (2, 3), f"Path should end at (2,3)"

    # Check that some index is repeated (warping happened)
    i_indices = [p[0] for p in path]
    has_repeat = len(i_indices) != len(set(i_indices))
    # Note: might also have j repeats, either is fine

    print(f"✓ test_dtw_path_with_stretch passed")
    print(f"  Path: {path}")


def test_backward_compatible():
    """Default behavior should be unchanged (returns float only)."""
    seq1 = np.array([1.0, 2.0, 3.0])
    seq2 = np.array([1.5, 2.5, 3.5])

    # Without return_path (default) - should return float
    result = compute_dtw_distance(seq1, seq2)
    assert isinstance(result, float), f"Expected float, got {type(result)}"

    # With return_path=True - should return tuple
    result = compute_dtw_distance(seq1, seq2, return_path=True)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
    assert isinstance(result[0], float), f"First element should be float (distance)"
    assert isinstance(result[1], list), f"Second element should be list (path)"

    print("✓ test_backward_compatible passed")


def test_distance_unchanged():
    """Distance should be same whether or not path is returned."""
    seq1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seq2 = np.array([1.5, 2.5, 3.5, 4.5])

    dist_only = compute_dtw_distance(seq1, seq2)
    dist_with_path, _ = compute_dtw_distance(seq1, seq2, return_path=True)

    assert abs(dist_only - dist_with_path) < 1e-10, \
        f"Distance mismatch: {dist_only} vs {dist_with_path}"

    print("✓ test_distance_unchanged passed")


if __name__ == '__main__':
    print("=" * 60)
    print("DTW Path Extraction Tests")
    print("=" * 60)
    print()

    test_dtw_path_basic()
    test_dtw_path_with_shift()
    test_dtw_path_with_stretch()
    test_backward_compatible()
    test_distance_unchanged()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
```

#### 4b. Visual comparison script

**File**: `results/mcolon/20260103_dba_refactor/compare_dba_vs_median.py`

```python
#!/usr/bin/env python
"""
Visual comparison of DBA vs binned median trend lines.

This script generates side-by-side plots to validate that DBA
produces sensible consensus trajectories.

Output:
- trend_binned_median.html / .png
- trend_dba.html / .png
"""
from pathlib import Path
import sys
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

import pandas as pd
from src.analyze.trajectory_analysis.facetted_plotting import plot_multimetric_trajectories

# Output directory
output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data...")
# TODO: Update this path to your actual data file
# df = pd.read_parquet('/path/to/your/data.parquet')

# For testing, you can use a subset:
# df = df[df['cluster'].isin([0, 1, 2])]

print(f"Generating plots in {output_dir}")

# Binned median (current default)
print("  Generating binned median plot...")
fig_median = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized'],
    col_by='cluster',
    trend_statistic='median',
    title='Trend: Binned Median',
    output_path=output_dir / 'trend_binned_median.html',
    backend='both',
)

# DBA trend line
print("  Generating DBA plot...")
fig_dba = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized'],
    col_by='cluster',
    trend_statistic='dba',  # NEW!
    title='Trend: DBA (DTW Barycenter Averaging)',
    output_path=output_dir / 'trend_dba.html',
    backend='both',
)

print()
print(f"Plots saved to {output_dir}")
print("  - trend_binned_median.html / .png")
print("  - trend_dba.html / .png")
print()
print("Compare the trend lines visually to verify DBA is working correctly.")
```

---

## Files Summary

### Files to Modify

| File | Change | Est. Lines |
|------|--------|------------|
| `src/analyze/trajectory_analysis/dtw_distance.py` | Add `return_path` param + `_backtrack_dtw_path()` | ~45 |
| `src/analyze/trajectory_analysis/facetted_plotting.py` | Add `_compute_dba_trend_line()` + handle `'dba'` | ~40 |
| `src/analyze/utils/plotting.py` | Use real path instead of fake | 1 |
| `src/analyze/utils/plotting_faceted.py` | Use real path instead of fake | 1 |

### Files to Create

| File | Purpose |
|------|---------|
| `results/mcolon/20260103_dba_refactor/test_dtw_path.py` | Unit tests |
| `results/mcolon/20260103_dba_refactor/compare_dba_vs_median.py` | Visual validation |

### Files NOT Changed (verify still work)

- `dba.py` - algorithm unchanged, just gets real paths now
- `compute_dtw_distance_matrix()` - unchanged (doesn't use path)
- All clustering code - unchanged

---

## Verification Checklist

After implementation, verify:

- [ ] `python results/mcolon/20260103_dba_refactor/test_dtw_path.py` passes
- [ ] `compute_dtw_distance(seq1, seq2)` returns float (backward compatible)
- [ ] `compute_dtw_distance(seq1, seq2, return_path=True)` returns (float, list)
- [ ] Path starts at (0, 0) and ends at (n-1, m-1)
- [ ] Path is NOT purely diagonal for different-length sequences
- [ ] `trend_statistic='dba'` works in plotting functions
- [ ] Existing clustering workflows still work

---

## Usage After Implementation

```python
from src.analyze.trajectory_analysis.facetted_plotting import plot_multimetric_trajectories

# Binned median (default, unchanged)
fig = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized'],
    col_by='cluster',
    trend_statistic='median',
)

# DBA trend line (NEW!)
fig = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized'],
    col_by='cluster',
    trend_statistic='dba',  # Just change this!
)
```

---

## Estimated Time

- Step 1 (DTW path): 2-3 hours
- Step 2 (Fix callers): 15 minutes
- Step 3 (facetted_plotting): 1-2 hours
- Step 4 (Tests): 30 minutes
- Verification: 30 minutes

**Total: ~5-6 hours**
