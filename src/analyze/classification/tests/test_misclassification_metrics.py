import numpy as np
import pandas as pd
import pytest

from analyze.classification.misclassification.metrics import compute_per_embryo_metrics


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"embryo_id": "e1", "time_bin": 0, "time_bin_center": 1.0, "true_class": "A", "pred_class": "A", "p_true": 0.9, "p_pred": 0.9, "pred_proba_A": 0.9, "pred_proba_B": 0.1},
            {"embryo_id": "e1", "time_bin": 2, "time_bin_center": 3.0, "true_class": "A", "pred_class": "B", "p_true": 0.2, "p_pred": 0.8, "pred_proba_A": 0.2, "pred_proba_B": 0.8},
            {"embryo_id": "e1", "time_bin": 4, "time_bin_center": 5.0, "true_class": "A", "pred_class": "B", "p_true": 0.1, "p_pred": 0.9, "pred_proba_A": 0.1, "pred_proba_B": 0.9},
            {"embryo_id": "e2", "time_bin": 0, "time_bin_center": 1.0, "true_class": "B", "pred_class": "B", "p_true": 0.8, "p_pred": 0.8, "pred_proba_A": 0.2, "pred_proba_B": 0.8},
            {"embryo_id": "e2", "time_bin": 2, "time_bin_center": 3.0, "true_class": "B", "pred_class": "A", "p_true": 0.3, "p_pred": 0.7, "pred_proba_A": 0.7, "pred_proba_B": 0.3},
        ]
    )


def test_compute_metrics_shapes_and_signals():
    per, baseline_ct, baseline_c = compute_per_embryo_metrics(_df())
    assert len(per) == 2
    assert "longest_wrong_streak" in per.columns
    assert "flip_rate" in per.columns
    assert "expected_wrong_rate" in per.columns
    # e1 has confident wrong bins, margin should be negative
    e1 = per[per["embryo_id"] == "e1"].iloc[0]
    assert e1["mean_margin"] < 0
    assert np.isclose(e1["top_confused_frac"], 1.0)
    assert not baseline_ct.empty
    assert not baseline_c.empty


def test_multi_true_class_raises_by_default():
    df = _df().copy()
    df.loc[df["embryo_id"] == "e1", "true_class"] = ["A", "B", "A"]
    with pytest.raises(ValueError):
        compute_per_embryo_metrics(df)
