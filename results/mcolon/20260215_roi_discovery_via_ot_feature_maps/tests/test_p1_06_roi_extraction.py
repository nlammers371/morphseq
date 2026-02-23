"""
Phase 1 / Task 1.6 — ROI extraction via quantile thresholding.

Checks:
- ROI is contained within mask_ref
- area_fraction is in (0, 1) and shrinks with higher quantile
- n_components is a positive integer
- For planted tail signal: ROI concentrates in the tail
- Empty weight maps produce empty ROI
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from conftest import GRID_H, GRID_W, N_CHANNELS, TAIL_START_ROW
from roi_trainer import extract_roi, train


@pytest.fixture
def trained_result(planted_data, class_weights, tiny_trainer_config):
    """A trained model on the planted data with moderate regularization.

    Uses moderate L1+TV so that the TV penalty suppresses the large but
    canceling weight patterns that appear with near-zero regularization.
    This ensures magnitude-based ROI extraction localizes correctly.
    """
    return train(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        class_weights=class_weights,
        lam=5e-3,
        mu=5e-3,
        config=tiny_trainer_config,
        channel_names=planted_data["channel_names"],
    )


def test_roi_inside_mask(trained_result, mask_ref):
    """ROI must be a subset of mask_ref."""
    roi, stats = extract_roi(trained_result.w_full, mask_ref, quantile=0.9)
    assert np.all(roi <= mask_ref), "ROI extends outside mask_ref"


def test_roi_area_fraction_range(trained_result, mask_ref):
    """area_fraction should be between 0 and 1."""
    roi, stats = extract_roi(trained_result.w_full, mask_ref, quantile=0.9)
    assert 0.0 < stats["area_fraction"] < 1.0


def test_roi_shrinks_with_higher_quantile(trained_result, mask_ref):
    """Higher quantile threshold should produce a smaller ROI."""
    _, stats_85 = extract_roi(trained_result.w_full, mask_ref, quantile=0.85)
    _, stats_95 = extract_roi(trained_result.w_full, mask_ref, quantile=0.95)

    assert stats_95["area_fraction"] <= stats_85["area_fraction"], (
        f"q=0.95 area={stats_95['area_fraction']} > q=0.85 area={stats_85['area_fraction']}"
    )


def test_roi_n_components_positive(trained_result, mask_ref):
    """ROI should have at least 1 connected component."""
    roi, stats = extract_roi(trained_result.w_full, mask_ref, quantile=0.9)
    assert stats["n_components"] >= 1


def test_roi_tail_concentration(trained_result, mask_ref):
    """
    BIOLOGICAL PRIOR (cep290): Most ROI pixels should be in the tail.
    We check that >50% of ROI area is in rows >= TAIL_START_ROW.
    """
    roi, stats = extract_roi(trained_result.w_full, mask_ref, quantile=0.9)
    total_roi = roi.sum()
    tail_roi = roi[TAIL_START_ROW:, :].sum()

    if total_roi > 0:
        tail_frac = tail_roi / total_roi
        assert tail_frac > 0.5, (
            f"Only {tail_frac:.1%} of ROI is in tail (expected >50%)"
        )


def test_empty_weight_map_gives_empty_roi(mask_ref):
    """A zero weight map should produce an empty ROI."""
    w = np.zeros((GRID_H, GRID_W, N_CHANNELS), dtype=np.float32)
    roi, stats = extract_roi(w, mask_ref, quantile=0.9)
    assert roi.sum() == 0
    assert stats["area_fraction"] == 0.0
    assert stats["n_components"] == 0


def test_roi_stats_keys(trained_result, mask_ref):
    """ROI stats dict must contain all specified keys."""
    roi, stats = extract_roi(trained_result.w_full, mask_ref, quantile=0.9)
    required = {"area_fraction", "n_components", "boundary_fraction", "threshold", "quantile"}
    assert required.issubset(stats.keys()), f"Missing keys: {required - set(stats.keys())}"
