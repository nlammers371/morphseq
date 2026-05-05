from __future__ import annotations

import numpy as np
import pandas as pd

from dev.particle_prediction.eval.latent_order import (
    effective_rank,
    infer_latent_columns,
    latent_corr_order_summary,
    mean_abs_offdiag_corr,
    rms_offdiag_corr,
    summarize_latent_order_dataframe,
    top_k_eigen_fraction,
)


def test_latent_corr_order_metrics_smoke() -> None:
    R = np.array(
        [
            [1.0, 0.5, -0.25],
            [0.5, 1.0, 0.0],
            [-0.25, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert np.isclose(rms_offdiag_corr(R), np.sqrt((0.5**2 + 0.25**2 + 0.5**2 + 0.25**2) / 6.0))
    assert np.isclose(mean_abs_offdiag_corr(R), (0.5 + 0.25 + 0.5 + 0.25) / 6.0)
    assert 1.0 <= effective_rank(R) <= 3.0
    assert 0.0 < top_k_eigen_fraction(R, k=1) <= top_k_eigen_fraction(R, k=3) <= 1.0


def test_latent_order_summary_by_dataframe_smoke() -> None:
    df = pd.DataFrame(
        {
            "temperature": [28.5, 28.5, 28.5, 28.5, 30.0, 30.0, 30.0],
            "stage_bin": ["early", "early", "early", "early", "early", "early", "early"],
            "z_mu_b0": [0.0, 1.0, 2.0, np.nan, 0.0, 1.0, 2.0],
            "z_mu_b1": [0.0, 2.0, 4.0, 5.0, 2.0, 1.0, 0.0],
            "z_mu_b2": [1.0, 1.5, 2.0, 3.0, 0.0, 1.0, 3.0],
        }
    )

    summary = summarize_latent_order_dataframe(df, temp_col="temperature")

    assert list(summary.columns) == [
        "temp",
        "stage_bin",
        "n_samples",
        "n_latent_features",
        "rms_offdiag_corr",
        "mean_abs_offdiag_corr",
        "effective_rank",
        "top1_eigen_fraction",
        "top3_eigen_fraction",
        "top5_eigen_fraction",
    ]
    assert summary["temp"].tolist() == [28.5, 30.0]
    assert summary["n_samples"].tolist() == [3, 3]
    assert summary["n_latent_features"].tolist() == [3, 3]
    assert np.all(np.isfinite(summary["rms_offdiag_corr"]))
    assert np.all(np.isfinite(summary["top3_eigen_fraction"]))


def test_infer_latent_columns_smoke() -> None:
    df = pd.DataFrame(columns=["z_mu_b10", "stage_bin", "z_mu_b2", "z_mu_b0"])

    assert infer_latent_columns(df) == ["z_mu_b0", "z_mu_b2", "z_mu_b10"]


def test_latent_corr_order_summary_handles_small_groups() -> None:
    summary = latent_corr_order_summary(np.array([[1.0, 2.0]], dtype=np.float64))

    assert summary["n_samples"] == 1
    assert summary["n_latent_features"] == 2
    assert np.isnan(summary["effective_rank"])

    single_feature_summary = latent_corr_order_summary(np.array([[1.0], [2.0]], dtype=np.float64))
    assert single_feature_summary["n_latent_features"] == 1
    assert np.isnan(single_feature_summary["rms_offdiag_corr"])
    assert single_feature_summary["effective_rank"] == 1.0
