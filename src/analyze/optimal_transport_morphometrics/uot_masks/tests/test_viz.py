"""Tests for Phase 2 viz contract enforcement."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from analyze.optimal_transport_morphometrics.uot_masks.viz import (
    apply_nan_mask,
    _build_support_mask_from_result,
    plot_uot_summary,
    plot_velocity_histogram,
    write_diagnostics_json,
)


@dataclass
class _FakeResult:
    """Minimal mock of UOTResult for viz testing."""
    mass_created_canon: np.ndarray
    mass_destroyed_canon: np.ndarray
    velocity_canon_px_per_step_yx: np.ndarray
    cost: float = 1.0
    diagnostics: Optional[dict] = None


def _make_fake_result(h=20, w=30):
    """Create a fake result with some non-zero data in a central region."""
    created = np.zeros((h, w), dtype=np.float32)
    destroyed = np.zeros((h, w), dtype=np.float32)
    velocity = np.zeros((h, w, 2), dtype=np.float32)

    # Put data in central region
    created[5:15, 10:20] = np.random.default_rng(0).uniform(0, 0.1, (10, 10))
    destroyed[5:15, 10:20] = np.random.default_rng(1).uniform(0, 0.05, (10, 10))
    velocity[5:15, 10:20, 0] = 1.5  # vy
    velocity[5:15, 10:20, 1] = 0.5  # vx

    return _FakeResult(
        mass_created_canon=created,
        mass_destroyed_canon=destroyed,
        velocity_canon_px_per_step_yx=velocity,
        diagnostics={"metrics": {"total_transport_cost": 1.0, "transported_mass": 0.5}},
    )


class TestApplyNanMask:
    def test_2d_basic(self):
        field = np.ones((4, 4))
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        result = apply_nan_mask(field, mask)
        assert np.isnan(result[0, 0])
        assert result[1, 1] == 1.0

    def test_3d_basic(self):
        field = np.ones((4, 4, 2))
        mask = np.zeros((4, 4), dtype=bool)
        mask[2, 2] = True
        result = apply_nan_mask(field, mask)
        assert np.isnan(result[0, 0, 0])
        assert result[2, 2, 0] == 1.0
        assert result[2, 2, 1] == 1.0

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            apply_nan_mask(np.ones(5), np.ones(5, dtype=bool))


class TestBuildSupportMask:
    def test_detects_nonzero_regions(self):
        result = _make_fake_result()
        mask = _build_support_mask_from_result(result)
        assert mask.shape == (20, 30)
        assert mask[10, 15]  # inside data region
        assert not mask[0, 0]  # outside data region


class TestPlotUotSummary:
    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        result = _make_fake_result()
        fig = plot_uot_summary(result, title="Test")
        assert fig is not None
        assert len(fig.axes) == 6  # 4 panels + 2 colorbars

    def test_saves_to_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        result = _make_fake_result()
        out = str(tmp_path / "summary.png")
        fig = plot_uot_summary(result, output_path=out)
        assert (tmp_path / "summary.png").exists()


class TestPlotVelocityHistogram:
    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        result = _make_fake_result()
        fig = plot_velocity_histogram(result)
        assert fig is not None


class TestWriteDiagnosticsJson:
    def test_writes_valid_json(self, tmp_path):
        result = _make_fake_result()
        out = str(tmp_path / "diag.json")
        write_diagnostics_json(result, out)
        with open(out) as f:
            data = json.load(f)
        assert "cost" in data
        assert "n_support_pixels" in data
        assert data["n_support_pixels"] > 0
        assert "mean_velocity_px_per_step" in data
