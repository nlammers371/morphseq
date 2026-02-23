"""Tests for reference embryo module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from analyze.optimal_transport_morphometrics.uot_masks.reference_embryo import (
    ReferenceField,
    ReferenceTimeseries,
    build_reference_field,
    compute_deviation_from_reference,
    compute_residual_field,
    deviation_timeseries,
)


@dataclass
class _FakeResult:
    """Minimal UOTResult mock."""
    mass_created_canon: np.ndarray
    mass_destroyed_canon: np.ndarray
    velocity_canon_px_per_step_yx: np.ndarray
    cost: float = 1.0
    diagnostics: Optional[dict] = None


def _make_result(h=10, w=10, vel_y=1.0, vel_x=0.5, created=0.1, destroyed=0.05, seed=0):
    rng = np.random.default_rng(seed)
    mc = np.zeros((h, w), dtype=np.float32)
    md = np.zeros((h, w), dtype=np.float32)
    vel = np.zeros((h, w, 2), dtype=np.float32)

    mc[3:7, 3:7] = created + rng.uniform(-0.01, 0.01, (4, 4))
    md[3:7, 3:7] = destroyed + rng.uniform(-0.01, 0.01, (4, 4))
    vel[3:7, 3:7, 0] = vel_y
    vel[3:7, 3:7, 1] = vel_x

    return _FakeResult(
        mass_created_canon=mc,
        mass_destroyed_canon=md,
        velocity_canon_px_per_step_yx=vel,
    )


class TestBuildReferenceField:
    def test_mean_of_identical(self):
        """Reference from 3 identical results → mean == input."""
        r = _make_result(vel_y=2.0, vel_x=1.0, created=0.5, destroyed=0.3, seed=99)
        # Use exact same data 3 times
        r1 = _FakeResult(r.mass_created_canon.copy(), r.mass_destroyed_canon.copy(), r.velocity_canon_px_per_step_yx.copy())
        r2 = _FakeResult(r.mass_created_canon.copy(), r.mass_destroyed_canon.copy(), r.velocity_canon_px_per_step_yx.copy())
        r3 = _FakeResult(r.mass_created_canon.copy(), r.mass_destroyed_canon.copy(), r.velocity_canon_px_per_step_yx.copy())

        ref = build_reference_field([r1, r2, r3])
        assert ref.n_embryos == 3
        np.testing.assert_allclose(ref.velocity_yx, r.velocity_canon_px_per_step_yx, atol=1e-5)
        np.testing.assert_allclose(ref.mass_created, r.mass_created_canon, atol=1e-5)
        np.testing.assert_allclose(ref.mass_destroyed, r.mass_destroyed_canon, atol=1e-5)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No results"):
            build_reference_field([])

    def test_unsupported_method(self):
        r = _make_result()
        with pytest.raises(ValueError, match="Unsupported method"):
            build_reference_field([r], method="median")


class TestComputeDeviation:
    def test_self_deviation_zero(self):
        """Deviation of self → RMSE == 0."""
        r = _make_result(vel_y=2.0, vel_x=1.0, seed=42)
        ref = build_reference_field([r])
        dev = compute_deviation_from_reference(r, ref)
        assert dev["rmse_velocity"] == pytest.approx(0.0, abs=1e-5)
        assert dev["rmse_mass_created"] == pytest.approx(0.0, abs=1e-5)
        assert dev["rmse_mass_destroyed"] == pytest.approx(0.0, abs=1e-5)
        assert dev["cosine_similarity"] == pytest.approx(1.0, abs=1e-3)

    def test_different_velocity_nonzero_rmse(self):
        r1 = _make_result(vel_y=2.0, vel_x=1.0, seed=42)
        r2 = _make_result(vel_y=0.0, vel_x=0.0, seed=42)
        ref = build_reference_field([r1])
        dev = compute_deviation_from_reference(r2, ref)
        assert dev["rmse_velocity"] > 0.5

    def test_residual_field(self):
        r = _make_result(vel_y=3.0, vel_x=2.0, seed=42)
        ref = build_reference_field([_make_result(vel_y=1.0, vel_x=1.0, seed=42)])
        residual = compute_residual_field(r, ref)
        assert residual.shape == r.velocity_canon_px_per_step_yx.shape
        # In the active region, residual should be ~(2.0, 1.0)
        assert np.abs(residual[5, 5, 0] - 2.0) < 0.1
        assert np.abs(residual[5, 5, 1] - 1.0) < 0.1


class TestDeviationTimeseries:
    def test_basic_timeseries(self):
        r1 = _make_result(vel_y=1.0, seed=10)
        r2 = _make_result(vel_y=1.5, seed=20)
        ref1 = build_reference_field([_make_result(vel_y=1.0, seed=10)])
        ref2 = build_reference_field([_make_result(vel_y=1.5, seed=20)])

        ref_ts = ReferenceTimeseries(fields={(0, 1): ref1, (1, 2): ref2})

        embryo_results = {(0, 1): r1, (1, 2): r2}
        df = deviation_timeseries(embryo_results, ref_ts, embryo_id="test_embryo")

        assert len(df) == 2
        assert "embryo_id" in df.columns
        assert "rmse_velocity" in df.columns
        assert all(df["rmse_velocity"] < 1e-3)  # Self-comparison should be ~0


class TestReferenceTimeseries:
    def test_contains_and_getitem(self):
        ref = build_reference_field([_make_result()])
        ts = ReferenceTimeseries(fields={(0, 1): ref})
        assert (0, 1) in ts
        assert (1, 2) not in ts
        assert ts[(0, 1)] is ref
        assert ts.frame_pairs == [(0, 1)]
