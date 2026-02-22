import pandas as pd

from analyze.classification.misclassification.metrics import compute_per_embryo_metrics
from analyze.classification.misclassification.null import (
    null_test_streak,
    null_test_top_confused_frac,
    null_test_wrong_rate,
)


def _df() -> pd.DataFrame:
    rows = []
    for embryo_id, true_class in [("e1", "A"), ("e2", "A"), ("e3", "B"), ("e4", "B")]:
        for tb in [0, 2, 4, 6]:
            pred = true_class
            if embryo_id in {"e1", "e3"} and tb in {2, 4}:
                pred = "B" if true_class == "A" else "A"
            rows.append(
                {
                    "embryo_id": embryo_id,
                    "time_bin": tb,
                    "time_bin_center": tb + 1.0,
                    "true_class": true_class,
                    "pred_class": pred,
                    "pred_proba_A": 0.8 if pred == "A" else 0.2,
                    "pred_proba_B": 0.8 if pred == "B" else 0.2,
                    "p_true": 0.8 if pred == true_class else 0.2,
                    "p_pred": 0.8,
                }
            )
    return pd.DataFrame(rows)


def test_null_tests_add_expected_columns():
    df = _df()
    per, baseline_ct, _ = compute_per_embryo_metrics(df)
    idx_map = per.set_index("embryo_id")["embryo_idx"].to_dict()
    df["embryo_idx"] = df["embryo_id"].map(idx_map)

    per2, run_wrong = null_test_wrong_rate(
        embryo_predictions=df,
        per_embryo_metrics=per,
        class_labels=["A", "B"],
        n_permutations=50,
        random_state=1,
    )
    assert "pval_wrong_rate" in per2.columns
    assert run_wrong.summary["stat_name"] == "wrong_rate"

    per3, run_streak = null_test_streak(
        per_embryo_metrics=per2,
        baseline_ct_df=baseline_ct,
        embryo_time_bins=df[["embryo_id", "time_bin"]].drop_duplicates(),
        n_sim=200,
        random_state=2,
    )
    assert "pval_streak" in per3.columns
    assert run_streak.summary["stat_name"] == "longest_wrong_streak"

    per4, run_top = null_test_top_confused_frac(
        per_embryo_metrics=per3,
        embryo_predictions=df,
        class_labels=["A", "B"],
        n_sim=200,
        random_state=3,
        require_n_wrong_min=1,
        loo_min_class_size=2,
    )
    assert "pval_top_confused_frac" in per4.columns
    assert run_top.summary["stat_name"] == "top_confused_frac"
