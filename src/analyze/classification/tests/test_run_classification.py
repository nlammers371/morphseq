"""Integration tests for run_classification()."""

import numpy as np
import pandas as pd
import pytest

from analyze.classification.run_classification import run_classification


def _make_df(
    n_classes: int = 3,
    n_embryos: int = 3,
    n_times: int = 2,
    n_features: int = 2,
) -> pd.DataFrame:
    """Synthetic data: classes with separable means."""
    rng = np.random.default_rng(0)
    classes = [chr(ord("A") + i) for i in range(n_classes)]
    rows = []
    for cls_idx, cls in enumerate(classes):
        for emb_i in range(n_embryos):
            embryo_id = f"{cls}_e{emb_i}"
            for t_idx in range(n_times):
                t = 24.0 + t_idx * 4.0
                row = {
                    "embryo_id": embryo_id,
                    "genotype": cls,
                    "predicted_stage_hpf": t,
                }
                for f in range(n_features):
                    row[f"z_mu_b_{f}"] = float(
                        rng.normal(loc=cls_idx * 2.0, scale=0.3)
                    )
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Basic modes
# ---------------------------------------------------------------------------


def test_all_vs_rest():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
    )
    assert len(result.scores) > 0
    assert set(result.scores.columns) >= {
        "feature_set", "comparison_id", "positive_label",
        "negative_label", "time_bin_center", "auroc_obs", "pval",
    }
    assert result.feature_sets == ["emb"]
    assert len(result.comparison_ids) == 3  # A vs rest, B vs rest, C vs rest


def test_explicit_pair():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive="A",
        negative="B",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
    )
    assert len(result.comparison_ids) == 1
    assert result.scores["positive_label"].unique().tolist() == ["A"]
    assert result.scores["negative_label"].unique().tolist() == ["B"]


def test_all_pairs():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        comparisons="all_pairs",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
    )
    # C(3,2) = 3 pairs
    assert len(result.comparison_ids) == 3


def test_pooled():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive=("A", "B"),
        negative="C",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
    )
    assert len(result.comparison_ids) == 1
    assert "A+B" in result.scores["positive_label"].values


def test_multi_feature():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive="A",
        negative="B",
        features={
            "emb": "z_mu_b",
            "single": ["z_mu_b_0"],
        },
        n_permutations=4,
        n_jobs=1,
        verbose=False,
    )
    assert sorted(result.feature_sets) == ["emb", "single"]
    # Each feature set should have same number of time bins
    counts = result.scores.groupby("feature_set").size()
    assert len(counts.unique()) == 1


# ---------------------------------------------------------------------------
# Predictions & layers
# ---------------------------------------------------------------------------


def test_save_predictions():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive="A",
        negative="B",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
        save_predictions=True,
    )
    preds = result.layers["predictions"]
    assert isinstance(preds, pd.DataFrame)
    assert "y_true" in preds.columns
    assert "p_pos" in preds.columns


def test_save_multiclass_predictions():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
        save_multiclass_predictions=True,
    )
    mc = result.layers["multiclass_predictions"]
    assert isinstance(mc, pd.DataFrame)
    assert "pred_proba_A" in mc.columns


def test_confusion_always_stored():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive="A",
        negative="B",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
    )
    assert "confusion" in result.layers


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_mutual_exclusion():
    df = _make_df()
    with pytest.raises(ValueError, match="Cannot combine"):
        run_classification(
            df,
            class_col="genotype",
            id_col="embryo_id",
            time_col="predicted_stage_hpf",
            positive="A",
            comparisons=pd.DataFrame({"positive": ["A"], "negative": ["B"]}),
            features={"emb": "z_mu_b"},
            verbose=False,
        )


def test_misclassification_missing_layer():
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive="A",
        negative="B",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
        save_multiclass_predictions=False,
    )
    with pytest.raises(KeyError, match="not computed"):
        result.layers["multiclass_predictions"]


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    df = _make_df()
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive="A",
        negative="B",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
    )
    out_path = result.save(tmp_path / "test_run")
    from analyze.classification.engine.analysis import ClassificationAnalysis
    loaded = ClassificationAnalysis.load(out_path)
    pd.testing.assert_frame_equal(loaded.scores, result.scores)
    assert loaded.uns["schema_version"] == "classification_v1"


def test_save_dir_auto_save(tmp_path):
    """Test that save_dir parameter auto-saves results."""
    df = _make_df()
    save_path = tmp_path / "auto_saved_run"
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive="A",
        negative="B",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=1,
        verbose=False,
        save_predictions=True,
        save_dir=save_path,
    )
    # Check that files were written to save_dir
    assert (save_path / "scores.parquet").exists()
    assert (save_path / "metadata.json").exists()
    assert (save_path / "predictions.parquet").exists()

    # Check that loaded results match
    from analyze.classification.engine.analysis import ClassificationAnalysis
    loaded = ClassificationAnalysis.load(save_path)
    pd.testing.assert_frame_equal(loaded.scores, result.scores)
    preds_original = result.layers["predictions"]
    preds_loaded = loaded.layers["predictions"]
    pd.testing.assert_frame_equal(preds_loaded, preds_original)
