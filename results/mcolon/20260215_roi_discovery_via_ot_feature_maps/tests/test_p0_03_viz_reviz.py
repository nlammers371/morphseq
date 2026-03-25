"""Tests for Phase 0 contour re-visualization helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap

from viz.phase0 import plot_cost_density_suite


def test_plot_cost_density_suite_accepts_custom_styling(tmp_path, mask_ref, tail_roi_mask):
    rng = np.random.default_rng(7)
    n_wt = 2
    n_mut = 2
    n_total = n_wt + n_mut

    X = rng.uniform(0.0, 0.2, size=(n_total, *mask_ref.shape, 1)).astype(np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    outlier_flag = np.zeros(n_total, dtype=bool)

    X[:, ~mask_ref, 0] = 0.0
    X[n_wt:, tail_roi_mask, 0] += 0.8

    figs = plot_cost_density_suite(
        X,
        y,
        mask_ref.astype(np.uint8),
        outlier_flag,
        sigma_grid=(2.0,),
        save_dir=tmp_path,
        contour_linewidth=1.8,
        outline_linewidth=3.0,
        contour_level_count=7,
        diff_level_count=9,
        save_suffix="presentation",
        save_dpi=120,
    )

    assert "cost_raw" in figs
    assert "cost_contour_sigma2.0" in figs
    assert (tmp_path / "fig_A1_A3_cost_density_raw_presentation.png").exists()
    assert (tmp_path / "fig_A_cost_contour_sigma2_presentation.png").exists()


def test_plot_cost_density_suite_supports_raw_background_contours(tmp_path, mask_ref, tail_roi_mask):
    rng = np.random.default_rng(9)
    X = rng.uniform(0.0, 0.2, size=(4, *mask_ref.shape, 1)).astype(np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    outlier_flag = np.zeros(4, dtype=bool)

    X[:, ~mask_ref, 0] = 0.0
    X[2:, tail_roi_mask, 0] += 0.6

    figs = plot_cost_density_suite(
        X,
        y,
        mask_ref.astype(np.uint8),
        outlier_flag,
        sigma_grid=(2.0,),
        contour_background="raw",
        save_dir=tmp_path,
        save_suffix="raw_overlay",
    )

    assert "cost_contour_sigma2.0" in figs
    assert (tmp_path / "fig_A_cost_contour_sigma2_raw_overlay.png").exists()


def test_plot_cost_density_suite_supports_custom_colormaps_and_colorbars(mask_ref, tail_roi_mask):
    X = np.zeros((4, *mask_ref.shape, 1), dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    outlier_flag = np.zeros(4, dtype=bool)
    X[2:, tail_roi_mask, 0] = 1.0

    mean_cmap = LinearSegmentedColormap.from_list("test_mean", ["#ffffff", "#2FB7B0"])
    diff_cmap = LinearSegmentedColormap.from_list("test_diff", ["#2FB7B0", "#ffffff", "#E76FA2"])

    figs = plot_cost_density_suite(
        X,
        y,
        mask_ref.astype(np.uint8),
        outlier_flag,
        sigma_grid=(2.0,),
        contour_background="raw",
        mean_cmap=mean_cmap,
        diff_cmap=diff_cmap,
        show_contour_colorbars=True,
    )

    contour_fig = figs["cost_contour_sigma2.0"]
    assert len(contour_fig.axes) > 3, "Expected extra axes from per-panel colorbars"


def test_reviz_phase0_smoke_writes_both_variants(tmp_path):
    run_dir = Path(
        "results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/output/phase0_run_004"
    )
    if not run_dir.exists():
        pytest.skip("phase0_run_004 cache is not available in this checkout")

    script_path = Path(
        "results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/reviz_phase0.py"
    )
    spec = importlib.util.spec_from_file_location("reviz_phase0", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    written = module.render_reviz(
        run_dir=run_dir.resolve(),
        output_dir=tmp_path,
        sigma=2.0,
        preset="both",
    )

    expected = {
        tmp_path / "fig_A_cost_contour_sigma2_warm_publication.png",
        tmp_path / "fig_A_cost_contour_sigma2_warm_presentation.png",
    }
    assert expected.issubset(set(written))
    for path in expected:
        assert path.exists(), f"Expected output not written: {path}"
