import numpy as np
import pandas as pd
import pytest

from analyze.classification.misclassification.trajectory import (
    STAGE_HARD,
    STAGE_RESIDUAL,
    STAGE_RESIDUAL_DTW,
    STAGE_SOFT,
    build_stage_feature_matrix,
    compute_rolling_window_destination_confusion_significance,
    compute_rolling_window_wrong_rate_significance,
    run_stage_geometry,
    validate_predictions_for_stage,
)


def _base_df() -> pd.DataFrame:
    rows = [
        {"embryo_id": "e1", "time_bin": 0, "true_class": "A", "pred_class": "A"},
        {"embryo_id": "e1", "time_bin": 2, "true_class": "A", "pred_class": "B"},
        {"embryo_id": "e2", "time_bin": 0, "true_class": "A", "pred_class": "A"},
        {"embryo_id": "e2", "time_bin": 2, "true_class": "A", "pred_class": "A"},
        {"embryo_id": "e3", "time_bin": 0, "true_class": "B", "pred_class": "B"},
        {"embryo_id": "e3", "time_bin": 2, "true_class": "B", "pred_class": "A"},
        {"embryo_id": "e4", "time_bin": 0, "true_class": "B", "pred_class": "B"},
        {"embryo_id": "e4", "time_bin": 2, "true_class": "B", "pred_class": "B"},
    ]
    return pd.DataFrame(rows)


def _soft_df() -> pd.DataFrame:
    df = _base_df().copy()
    df["time_bin_center"] = df["time_bin"].astype(float)
    pA = np.array([0.85, 0.20, 0.90, 0.88, 0.18, 0.72, 0.12, 0.08], dtype=float)
    df["pred_proba_A"] = pA
    df["pred_proba_B"] = 1.0 - pA
    df["p_pred"] = np.where(df["pred_class"] == "A", df["pred_proba_A"], df["pred_proba_B"])
    df["p_true"] = np.where(df["true_class"] == "A", df["pred_proba_A"], df["pred_proba_B"])
    return df


def test_stage0_validation_without_probabilities():
    df = _base_df()
    validate_predictions_for_stage(df, class_labels=["A", "B"], stage_mode=STAGE_HARD)


def test_stage1_validation_requires_probabilities():
    df = _base_df()
    with pytest.raises(ValueError, match="pred_proba"):
        validate_predictions_for_stage(df, class_labels=["A", "B"], stage_mode=STAGE_SOFT)


def test_probability_set_and_row_sum_validation():
    df = _soft_df()
    df["pred_proba_C"] = 0.0
    with pytest.raises(ValueError, match="mismatch"):
        validate_predictions_for_stage(df, class_labels=["A", "B"], stage_mode=STAGE_SOFT)

    df = _soft_df()
    df.loc[0, "pred_proba_B"] = 0.40
    with pytest.raises(ValueError, match="row sums"):
        validate_predictions_for_stage(df, class_labels=["A", "B"], stage_mode=STAGE_SOFT)


def test_build_and_run_stage_geometry_soft_outputs_expected_columns():
    df = _soft_df()
    x, tensor, cols, time_bins, meta, baseline = build_stage_feature_matrix(
        df,
        class_labels=["A", "B"],
        stage_mode=STAGE_SOFT,
        wrong_rate_n_permutations=50,
        random_state=42,
    )
    assert x.shape == (4, 4)
    assert tensor.shape == (4, 2, 2)
    assert len(cols) == 4
    assert time_bins == [0, 2]
    assert baseline is None
    assert "wrong_frac" in meta.columns
    assert "is_wrong_more_often" in meta.columns
    assert "is_wrong_top_quartile" in meta.columns
    assert "is_wrong_significant_in_window_perm" in meta.columns
    assert "qval_wrong_rate_window_perm" in meta.columns

    result = run_stage_geometry(
        df,
        class_labels=["A", "B"],
        stage_mode=STAGE_SOFT,
        k_values=(2, 3),
        pca_components=2,
        random_state=42,
        wrong_rate_n_permutations=50,
    )
    out = result.stage_table
    assert "PC1" in out.columns
    assert "cluster_k2" in out.columns
    assert "cluster_k3" in out.columns
    assert any(m["k"] == 2 for m in result.metrics_by_k)


def test_residual_stage_emits_baseline_and_optional_dtw_distance():
    df = _soft_df()
    residual = run_stage_geometry(
        df,
        class_labels=["A", "B"],
        stage_mode=STAGE_RESIDUAL,
        k_values=(2,),
        pca_components=2,
        random_state=42,
        wrong_rate_n_permutations=50,
    )
    assert residual.baseline_mu is not None
    assert set(residual.baseline_mu.columns) >= {
        "true_class",
        "time_bin",
        "pred_proba_A_mu",
        "pred_proba_B_mu",
    }

    dtw = run_stage_geometry(
        df,
        class_labels=["A", "B"],
        stage_mode=STAGE_RESIDUAL_DTW,
        k_values=(2,),
        pca_components=2,
        random_state=42,
        dtw_window=1,
        wrong_rate_n_permutations=50,
    )
    assert dtw.distance_matrix is not None
    assert dtw.distance_matrix.shape == (4, 4)
    assert np.allclose(np.diag(dtw.distance_matrix), 0.0)


def test_rolling_window_significance_returns_expected_columns():
    df = _soft_df()
    out = compute_rolling_window_wrong_rate_significance(
        df,
        class_labels=["A", "B"],
        window_hpf=2.0,
        n_permutations=30,
        random_state=42,
        q_threshold=0.10,
    )
    assert not out.empty
    assert {"embryo_id", "window_center_hpf", "qval_wrong_rate_window_global_perm"} <= set(out.columns)
    assert {"is_wrong_significant_in_window_perm", "is_wrong_significant_in_window_global_perm"} <= set(out.columns)


def test_rolling_destination_confusion_significance_returns_expected_columns():
    df = _soft_df()
    out = compute_rolling_window_destination_confusion_significance(
        df,
        class_labels=["A", "B"],
        source_class="A",
        target_class="B",
        window_hpf=2.0,
        n_permutations=30,
        random_state=42,
        q_threshold=0.10,
    )
    assert not out.empty
    assert {"embryo_id", "window_center_hpf", "qval_dest_confusion_global_perm"} <= set(out.columns)
    assert {"is_dest_confusion_significant_perm", "is_dest_confusion_significant_global_perm"} <= set(out.columns)
