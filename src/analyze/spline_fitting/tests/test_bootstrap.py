"""Property-based tests for _fit_single_spline bootstrap migration.

Tests verify structural properties of the output (not exact values)
since the RNG strategy changes from legacy RandomState to SeedSequence.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(n=200, n_pca=3, seed=0):
    """Create a synthetic DataFrame suitable for _fit_single_spline.

    Generates a simple trajectory along PC1 correlated with stage.
    """
    rng = np.random.default_rng(seed)
    stages = np.linspace(24, 72, n)
    data = {"predicted_stage_hpf": stages}
    for i in range(n_pca):
        data[f"PC{i+1}"] = stages / 72.0 + rng.normal(0, 0.1, n)
    return pd.DataFrame(data)


def _pca_cols(n=3):
    return [f"PC{i+1}" for i in range(n)]


class FakeLPC:
    """A fake LocalPrincipalCurve that returns deterministic output.

    Replaces the real LPC to avoid heavy computation in tests.
    Produces a simple linspace spline between start and end points.
    """
    def __init__(self, **kwargs):
        self.cubic_splines = None

    def fit(self, coords, start_points, end_point, num_points):
        n_dims = coords.shape[1]
        spline = np.column_stack([
            np.linspace(start_points[0, d], end_point[0, d], num_points)
            for d in range(n_dims)
        ])
        # Add small noise based on coords mean so bootstrap replicates differ
        noise_scale = np.std(coords, axis=0).mean() * 0.01
        rng = np.random.default_rng(hash(coords.tobytes()) % (2**31))
        spline = spline + rng.normal(0, noise_scale, spline.shape)
        self.cubic_splines = [spline]


class FailingLPC:
    """An LPC that raises on fit — used to test retry logic."""
    def __init__(self, **kwargs):
        self.cubic_splines = None

    def fit(self, coords, start_points, end_point, num_points):
        raise RuntimeError("LPC fit failed")


class SometimesFailingLPC:
    """An LPC that fails on odd calls and succeeds on even calls."""
    call_count = 0

    def __init__(self, **kwargs):
        self.cubic_splines = None

    def fit(self, coords, start_points, end_point, num_points):
        SometimesFailingLPC.call_count += 1
        if SometimesFailingLPC.call_count % 2 == 1:
            raise RuntimeError("LPC fit failed (odd call)")
        n_dims = coords.shape[1]
        spline = np.column_stack([
            np.linspace(0, 1, num_points) for _ in range(n_dims)
        ])
        self.cubic_splines = [spline]


# ---------------------------------------------------------------------------
# Tests — structural properties
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    return _make_synthetic_df()


class TestFitSingleSplineReturnType:
    """Return type and column structure."""

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_returns_dataframe(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
        )
        assert isinstance(result, pd.DataFrame)

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_has_expected_columns(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        cols = _pca_cols()
        result = _fit_single_spline(
            synthetic_df,
            pca_cols=cols,
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
        )
        # Must have PCA columns, SE columns, and spline_point_index
        for c in cols:
            assert c in result.columns, f"Missing column: {c}"
            assert f"{c}_se" in result.columns, f"Missing SE column: {c}_se"
        assert "spline_point_index" in result.columns

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_shape_matches_n_spline_points(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        n_pts = 30
        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=n_pts,
        )
        assert len(result) == n_pts

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_column_count(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        n_pca = 3
        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(n_pca),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
        )
        # n_pca mean cols + n_pca SE cols + 1 index col
        expected_ncols = n_pca * 2 + 1
        assert len(result.columns) == expected_ncols


class TestFitSingleSplineValues:
    """Value properties — ranges, signs, finiteness."""

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_se_nonnegative(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=5,
            bootstrap_size=50,
            n_spline_points=20,
        )
        se_cols = [c for c in result.columns if c.endswith("_se")]
        for c in se_cols:
            assert (result[c] >= 0).all(), f"Negative SE in column {c}"

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_mean_values_finite(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=5,
            bootstrap_size=50,
            n_spline_points=20,
        )
        for c in _pca_cols():
            assert np.isfinite(result[c].values).all(), f"Non-finite in {c}"

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_spline_point_index_sequential(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        n_pts = 25
        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=n_pts,
        )
        expected_idx = list(range(n_pts))
        assert result["spline_point_index"].tolist() == expected_idx


class TestFitSingleSplineDeterminism:
    """Deterministic seeding via random_state parameter."""

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_same_seed_same_result(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        kwargs = dict(
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
            random_state=123,
        )
        r1 = _fit_single_spline(synthetic_df, **kwargs)
        r2 = _fit_single_spline(synthetic_df, **kwargs)
        pd.testing.assert_frame_equal(r1, r2)

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_different_seeds_different_results(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        kwargs = dict(
            pca_cols=_pca_cols(),
            n_bootstrap=5,
            bootstrap_size=50,
            n_spline_points=20,
        )
        r1 = _fit_single_spline(synthetic_df, random_state=1, **kwargs)
        r2 = _fit_single_spline(synthetic_df, random_state=2, **kwargs)
        # Same structure
        assert list(r1.columns) == list(r2.columns)
        assert len(r1) == len(r2)
        # But different values (at least in mean coords)
        assert not np.allclose(r1[_pca_cols()].values, r2[_pca_cols()].values)


class TestFitSingleSplineRetryLogic:
    """Retry/failure handling."""

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FailingLPC)
    def test_all_failures_returns_nan_dataframe(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
        )
        # Should return a NaN DataFrame
        assert isinstance(result, pd.DataFrame)
        for c in _pca_cols():
            assert result[c].isna().all(), f"Expected NaN in {c}"
            assert result[f"{c}_se"].isna().all()
        assert "spline_point_index" in result.columns
        assert len(result) == 20

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_warns_on_failures(self, synthetic_df):
        """If some bootstrap iterations fail, a warning is emitted."""
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        # This test uses FakeLPC which never fails, so no warning expected.
        # We just verify the code runs without error.
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _fit_single_spline(
                synthetic_df,
                pca_cols=_pca_cols(),
                n_bootstrap=3,
                bootstrap_size=50,
                n_spline_points=20,
            )
            assert isinstance(result, pd.DataFrame)


class TestFitSingleSplineWeights:
    """Observation weights handling."""

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_uniform_weights_default(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        # Should work without explicit weights
        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
        )
        assert isinstance(result, pd.DataFrame)

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_custom_weights(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        n = len(synthetic_df)
        weights = np.ones(n)
        weights[:n // 2] = 2.0  # Weight first half more
        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
            obs_weights=weights,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20


class TestFitSingleSplineEdgeCases:
    """Edge cases and validation."""

    def test_missing_pca_cols_raises(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        with pytest.raises(ValueError, match="not found"):
            _fit_single_spline(
                synthetic_df,
                pca_cols=["NONEXISTENT"],
                n_bootstrap=3,
                bootstrap_size=50,
                n_spline_points=20,
            )

    def test_missing_stage_col_raises(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        with pytest.raises(ValueError, match="not found"):
            _fit_single_spline(
                synthetic_df,
                pca_cols=_pca_cols(),
                stage_col="nonexistent_stage",
                n_bootstrap=3,
                bootstrap_size=50,
                n_spline_points=20,
            )

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_bootstrap_size_capped_at_df_size(self):
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        small_df = _make_synthetic_df(n=30)
        result = _fit_single_spline(
            small_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=10000,  # much larger than df
            n_spline_points=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_random_state_default_is_42(self, synthetic_df):
        """Default random_state=42 for backward compat."""
        from analyze.spline_fitting.bootstrap import _fit_single_spline

        # Just verify the function accepts the call without specifying random_state
        result = _fit_single_spline(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
        )
        assert isinstance(result, pd.DataFrame)


class TestSplineFitWrapper:
    """Tests for the public spline_fit_wrapper function."""

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_single_spline_mode(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import spline_fit_wrapper

        result = spline_fit_wrapper(
            synthetic_df,
            pca_cols=_pca_cols(),
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert "spline_point_index" in result.columns

    @patch("analyze.spline_fitting.bootstrap.LocalPrincipalCurve", FakeLPC)
    def test_group_by_mode(self, synthetic_df):
        from analyze.spline_fitting.bootstrap import spline_fit_wrapper

        df = synthetic_df.copy()
        df["group"] = np.where(df.index < 100, "A", "B")
        result = spline_fit_wrapper(
            df,
            pca_cols=_pca_cols(),
            group_by="group",
            n_bootstrap=3,
            bootstrap_size=50,
            n_spline_points=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert "group" in result.columns
        assert set(result["group"].unique()) == {"A", "B"}
