#!/usr/bin/env python3
"""
Unit Tests for NaN-aware Multivariate DTW

Tests for the NaN-aware cost matrix computation introduced in commit 22d3e501.
These tests verify correct handling of missing data (NaNs) at time series edges,
which is the main use case for trajectory data where embryos may have different
observation windows.
"""/

import sys
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

# Determine repo root and add src to path
REPO_ROOT = os.environ.get('MORPHSEQ_REPO_ROOT')
if not REPO_ROOT:
    REPO_ROOT = str(Path(__file__).resolve().parents[3])

SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from analyze.utils.timeseries.dtw import (
    _nan_aware_cost_matrix,
    _dtw_multivariate_pair,
    compute_md_dtw_distance_matrix,
)


class TestNanAwareCostMatrix:
    """Tests for _nan_aware_cost_matrix function."""

    def test_nan_aware_cost_matrix_no_nans(self):
        """Verify that _nan_aware_cost_matrix matches scipy.cdist when there are no NaNs."""
        np.random.seed(42)
        ts_a = np.random.randn(10, 3)  # 10 timepoints, 3 features
        ts_b = np.random.randn(8, 3)   # 8 timepoints, 3 features

        # Our NaN-aware implementation
        result = _nan_aware_cost_matrix(ts_a, ts_b)

        # scipy reference
        expected = cdist(ts_a, ts_b, metric='euclidean')

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_nan_aware_cost_matrix_edge_nans(self):
        """Test with NaNs at the edges of time series (main use case)."""
        # Simulate embryos with different observation windows
        # Embryo A: observed from t=2 to t=7 (NaN at edges)
        # Embryo B: observed from t=0 to t=5 (NaN at end)
        n_features = 2

        ts_a = np.array([
            [np.nan, np.nan],  # t=0: not observed
            [np.nan, np.nan],  # t=1: not observed
            [1.0, 2.0],        # t=2: observed
            [1.5, 2.5],        # t=3: observed
            [2.0, 3.0],        # t=4: observed
            [2.5, 3.5],        # t=5: observed
            [3.0, 4.0],        # t=6: observed
            [3.5, 4.5],        # t=7: observed
        ])

        ts_b = np.array([
            [0.5, 1.5],        # t=0: observed
            [1.0, 2.0],        # t=1: observed
            [1.5, 2.5],        # t=2: observed
            [2.0, 3.0],        # t=3: observed
            [2.5, 3.5],        # t=4: observed
            [3.0, 4.0],        # t=5: observed
            [np.nan, np.nan],  # t=6: not observed
            [np.nan, np.nan],  # t=7: not observed
        ])

        result = _nan_aware_cost_matrix(ts_a, ts_b)

        # Check shape
        assert result.shape == (8, 8)

        # Rows 0-1 of ts_a are all NaN, so cost should be inf for all cols
        assert np.all(np.isinf(result[0, :]))
        assert np.all(np.isinf(result[1, :]))

        # Cols 6-7 of ts_b are all NaN, so cost should be inf for all rows
        assert np.all(np.isinf(result[:, 6]))
        assert np.all(np.isinf(result[:, 7]))

        # Valid region (rows 2-7, cols 0-5) should have finite costs
        valid_region = result[2:8, 0:6]
        assert np.all(np.isfinite(valid_region))

        # Verify specific values in valid region match standard Euclidean
        for i in range(2, 8):
            for j in range(6):
                expected = np.sqrt(np.sum((ts_a[i] - ts_b[j]) ** 2))
                np.testing.assert_allclose(result[i, j], expected, rtol=1e-10)

    def test_nan_aware_cost_matrix_one_feature_all_nan(self):
        """When one entire feature is NaN, verify remaining features are used with scaling."""
        # ts_a: feature 0 is always NaN, feature 1 is valid
        ts_a = np.array([
            [np.nan, 1.0],
            [np.nan, 2.0],
            [np.nan, 3.0],
        ])

        # ts_b: both features valid
        ts_b = np.array([
            [0.0, 1.0],
            [0.0, 2.0],
        ])

        result = _nan_aware_cost_matrix(ts_a, ts_b)

        # All comparisons should use only feature 1 (the valid one)
        # With scaling: sqrt(SSE * (n_features / valid_counts))
        # n_features = 2, valid_counts = 1
        # So distance = sqrt(diff^2 * 2) = diff * sqrt(2)

        n_features = 2
        valid_counts = 1
        scaling = n_features / valid_counts

        for i in range(3):
            for j in range(2):
                diff = ts_a[i, 1] - ts_b[j, 1]
                expected = np.sqrt(diff ** 2 * scaling)
                np.testing.assert_allclose(result[i, j], expected, rtol=1e-10)

    def test_nan_aware_cost_matrix_no_overlap(self):
        """When no features have valid data at a (i,j) pair, cost should be inf."""
        # ts_a: only feature 0 valid
        ts_a = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
        ])

        # ts_b: only feature 1 valid
        ts_b = np.array([
            [np.nan, 1.0],
            [np.nan, 2.0],
        ])

        result = _nan_aware_cost_matrix(ts_a, ts_b)

        # No feature has overlap, all costs should be inf
        assert np.all(np.isinf(result))

    def test_nan_aware_cost_matrix_scaling_math(self):
        """
        Key test: Verify the scaling formula is mathematically correct.

        The formula sqrt(SSE * (n_features / valid_counts)) scales variance
        proportionally to missing features:
        - If 2 features total, 1 valid with squared error e^2 -> result is e * sqrt(2)
        - This correctly estimates what the full-feature distance would be
          assuming equal variance across features.
        """
        n_features = 4

        # Construct test case: only 2 of 4 features valid at overlap
        # Features 0,1 valid in ts_a, features 0,1 valid in ts_b -> 2 overlap
        ts_a = np.array([
            [1.0, 2.0, np.nan, np.nan],
        ])
        ts_b = np.array([
            [2.0, 4.0, np.nan, np.nan],
        ])

        result = _nan_aware_cost_matrix(ts_a, ts_b)

        # Valid features: 0 and 1
        # SSE = (1-2)^2 + (2-4)^2 = 1 + 4 = 5
        # valid_counts = 2
        # scaling = n_features / valid_counts = 4 / 2 = 2
        # result = sqrt(SSE * scaling) = sqrt(5 * 2) = sqrt(10)

        sse = (1.0 - 2.0) ** 2 + (2.0 - 4.0) ** 2
        valid_counts = 2
        scaling = n_features / valid_counts
        expected = np.sqrt(sse * scaling)

        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-10)

        # Additional check: if all 4 features were present with same per-feature variance,
        # the expected distance would be sqrt(SSE_per_feature * n_features)
        # Our scaling estimates this from partial data
        # With 2 features having SSE=5, per-feature SSE = 5/2 = 2.5
        # Full 4-feature distance would be sqrt(2.5 * 4) = sqrt(10)
        # This matches our result!

    def test_nan_aware_cost_matrix_single_timepoint(self):
        """Edge case: single timepoint per series."""
        ts_a = np.array([[1.0, 2.0, 3.0]])
        ts_b = np.array([[1.5, 2.5, 3.5]])

        result = _nan_aware_cost_matrix(ts_a, ts_b)

        expected = cdist(ts_a, ts_b, metric='euclidean')
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_nan_aware_cost_matrix_dimension_mismatch(self):
        """Verify error on feature dimension mismatch."""
        ts_a = np.array([[1.0, 2.0]])
        ts_b = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="Feature mismatch"):
            _nan_aware_cost_matrix(ts_a, ts_b)

    def test_nan_aware_cost_matrix_1d_input_error(self):
        """Verify error on 1D input (must be 2D)."""
        ts_a = np.array([1.0, 2.0, 3.0])
        ts_b = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="must be 2D"):
            _nan_aware_cost_matrix(ts_a, ts_b)


class TestDtwMultivariatePair:
    """Tests for _dtw_multivariate_pair function."""

    def test_dtw_multivariate_pair_with_nans(self):
        """Integration test: verify finite distances when inputs have partial feature NaNs.

        Note: DTW requires valid data at both start (0,0) and end (n,m) positions
        since the warping path must traverse from corner to corner. However, NaNs
        in *some* features are handled by the NaN-aware cost matrix, which scales
        the distance based on available features.
        """
        # Simulate two embryos where one feature has edge NaNs but other feature is valid
        # Feature 0: has NaNs at edges
        # Feature 1: fully valid
        ts_a = np.array([
            [np.nan, 1.0],  # Feature 0 missing, feature 1 valid
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
            [2.5, 3.5],
        ])

        ts_b = np.array([
            [0.5, 1.5],
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
            [np.nan, 4.0],  # Feature 0 missing, feature 1 valid
        ])

        distance = _dtw_multivariate_pair(ts_a, ts_b, window=3)

        # Should produce a finite distance (at least one feature valid at each position)
        assert np.isfinite(distance)
        assert distance >= 0

    def test_dtw_multivariate_pair_edge_nans_inf(self):
        """When all features are NaN at corners, DTW path is blocked -> inf distance."""
        # All NaN at start of ts_a blocks the DTW path
        ts_a = np.array([
            [np.nan, np.nan],  # All features NaN at t=0
            [1.0, 2.0],
            [1.5, 2.5],
        ])

        ts_b = np.array([
            [0.5, 1.5],
            [1.0, 2.0],
            [1.5, 2.5],
        ])

        distance = _dtw_multivariate_pair(ts_a, ts_b, window=3)

        # Path cannot start from (0,0) since cost is inf there
        assert np.isinf(distance)

    def test_dtw_multivariate_pair_identical_series(self):
        """Identical series should have zero distance."""
        ts = np.array([
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
        ])

        distance = _dtw_multivariate_pair(ts, ts.copy(), window=3)

        np.testing.assert_allclose(distance, 0.0, atol=1e-10)

    def test_dtw_multivariate_pair_no_overlap_returns_inf(self):
        """When there's no valid overlap, distance should be inf."""
        # ts_a has only feature 0 valid
        ts_a = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
        ])

        # ts_b has only feature 1 valid
        ts_b = np.array([
            [np.nan, 1.0],
            [np.nan, 2.0],
        ])

        distance = _dtw_multivariate_pair(ts_a, ts_b, window=3)

        assert np.isinf(distance)

    def test_dtw_multivariate_pair_unconstrained(self):
        """Test unconstrained DTW (window=None)."""
        np.random.seed(42)
        ts_a = np.random.randn(5, 2)
        ts_b = np.random.randn(6, 2)

        dist_unconstrained = _dtw_multivariate_pair(ts_a, ts_b, window=None)
        dist_constrained = _dtw_multivariate_pair(ts_a, ts_b, window=3)

        # Unconstrained should be <= constrained (more paths available)
        assert dist_unconstrained <= dist_constrained + 1e-10


class TestMdDtwDistanceMatrix:
    """Tests for compute_md_dtw_distance_matrix function."""

    def test_md_dtw_distance_matrix_with_partial_nans(self):
        """End-to-end test: distance matrix with partial feature NaNs is valid and symmetric.

        Tests the main use case: some features may be missing at certain timepoints,
        but at least one feature is valid at each timepoint. The NaN-aware cost
        matrix scales distances appropriately based on available features.
        """
        np.random.seed(42)
        n_samples = 5
        n_timepoints = 10
        n_features = 3

        # Create data with partial feature NaNs (not entire timepoints)
        X = np.random.randn(n_samples, n_timepoints, n_features)

        # Add NaNs to individual features at edges, but keep at least one feature valid
        X[0, :2, 0] = np.nan   # Sample 0: feature 0 missing at first 2 timepoints
        X[1, -3:, 1] = np.nan  # Sample 1: feature 1 missing at last 3 timepoints
        X[2, :1, 2] = np.nan   # Sample 2: feature 2 missing at first timepoint
        X[2, -1:, 0] = np.nan  # Sample 2: feature 0 missing at last timepoint

        D = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3, n_jobs=1, verbose=False)

        # Check shape
        assert D.shape == (n_samples, n_samples)

        # Check symmetry
        np.testing.assert_allclose(D, D.T, rtol=1e-10)

        # Check diagonal is zero
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

        # Check all off-diagonal values are finite and non-negative
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    assert np.isfinite(D[i, j]), f"D[{i},{j}] is not finite"
                    assert D[i, j] >= 0, f"D[{i},{j}] is negative"

    def test_md_dtw_distance_matrix_full_nan_timepoints(self):
        """Test behavior when entire timepoints are NaN (path may be blocked)."""
        np.random.seed(42)
        n_samples = 3
        n_timepoints = 10
        n_features = 2

        X = np.random.randn(n_samples, n_timepoints, n_features)

        # Sample 0: first timepoint all NaN
        X[0, 0, :] = np.nan
        # Sample 1: last timepoint all NaN
        X[1, -1, :] = np.nan

        D = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3, n_jobs=1, verbose=False)

        # Check symmetry still holds
        np.testing.assert_allclose(D, D.T, rtol=1e-10)

        # D[0,1] should be inf (sample 0 starts with NaN, sample 1 ends with NaN)
        # The path (0,0) to (n-1, m-1) goes through inf costs at corners
        assert np.isinf(D[0, 1])
        assert np.isinf(D[1, 0])

        # D[0,2] should also be inf (sample 0 has NaN at start)
        assert np.isinf(D[0, 2])

        # D[1,2] should also be inf (sample 1 has NaN at end)
        assert np.isinf(D[1, 2])

    def test_md_dtw_distance_matrix_no_nans(self):
        """Verify distance matrix is correct when there are no NaNs."""
        np.random.seed(42)
        X = np.random.randn(4, 8, 2)

        D = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3, n_jobs=1, verbose=False)

        # Manually compute expected distances
        for i in range(4):
            for j in range(i + 1, 4):
                expected = _dtw_multivariate_pair(X[i], X[j], window=3)
                np.testing.assert_allclose(D[i, j], expected, rtol=1e-10)
                np.testing.assert_allclose(D[j, i], expected, rtol=1e-10)

    def test_md_dtw_distance_matrix_parallel(self):
        """Verify parallel computation gives same result as single-threaded."""
        np.random.seed(42)
        X = np.random.randn(4, 8, 2)

        D_serial = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3, n_jobs=1, verbose=False)
        D_parallel = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3, n_jobs=2, verbose=False)

        np.testing.assert_allclose(D_serial, D_parallel, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
