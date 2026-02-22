import numpy as np
import pandas as pd

from analyze.classification.classification_test import run_classification_test


def _make_df() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    classes = ["A", "B", "WT"]
    for cls_idx, cls in enumerate(classes):
        for emb_i in range(3):
            embryo_id = f"{cls}_e{emb_i}"
            for t in [10.0, 14.0]:
                rows.append(
                    {
                        "embryo_id": embryo_id,
                        "cluster": cls,
                        "predicted_stage_hpf": t,
                        "z_mu_b_0": float(rng.normal(loc=cls_idx * 1.0, scale=0.3)),
                        "z_mu_b_1": float(rng.normal(loc=cls_idx * 0.5, scale=0.3)),
                    }
                )
    return pd.DataFrame(rows)


def test_all_rest_outputs_all_probability_columns_non_nan():
    df = _make_df()
    res = run_classification_test(
        df=df,
        groupby="cluster",
        groups="all",
        reference="rest",
        features=["z_mu_b_0", "z_mu_b_1"],
        n_permutations=6,
        n_jobs=1,
        verbose=False,
    )

    emb = res.embryo_predictions
    assert emb is not None
    for col in ["pred_proba_A", "pred_proba_B", "pred_proba_WT"]:
        assert col in emb.columns
        assert not emb[col].isna().any()

    assert "p_true" in emb.columns
    assert "p_pred" in emb.columns
    assert "is_wrong" in emb.columns


def test_one_vs_one_reference_mode_still_works():
    df = _make_df()
    res = run_classification_test(
        df=df,
        groupby="cluster",
        groups=["A"],
        reference="WT",
        features=["z_mu_b_0", "z_mu_b_1"],
        n_permutations=4,
        n_jobs=1,
        verbose=False,
    )

    assert len(res.comparisons) > 0
    assert set(res.comparisons["positive"].unique()) == {"A"}
    assert set(res.comparisons["negative"].unique()) == {"WT"}
