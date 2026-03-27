"""Integration tests for run_classification()."""

import numpy as np
import pandas as pd
import pytest

import analyze.classification.engine.loop as loop_module
from analyze.classification.engine.loop import _bin_and_aggregate, _build_binary_labels
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


def test_all_vs_rest_sparse_bin_retains_valid_classes():
    rows = []
    for cls_idx, cls in enumerate(["A", "B", "C"]):
        for emb_i in range(3):
            embryo_id = f"{cls}_e{emb_i}"
            rows.append({
                "embryo_id": embryo_id,
                "genotype": cls,
                "predicted_stage_hpf": 24.0,
                "z_mu_b_0": float(cls_idx * 3.0 + emb_i * 0.1),
            })
            if cls != "C" or emb_i == 0:
                rows.append({
                    "embryo_id": embryo_id,
                    "genotype": cls,
                    "predicted_stage_hpf": 28.0,
                    "z_mu_b_0": float(cls_idx * 3.0 + 1.5 + emb_i * 0.1),
                })

    df = pd.DataFrame(rows)
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        features={"emb": ["z_mu_b_0"]},
        bin_width=4.0,
        n_permutations=4,
        n_jobs=1,
        verbose=False,
        save_multiclass_predictions=True,
    )

    late_scores = result.scores[result.scores["time_bin"] == 28].copy()
    assert set(late_scores["positive_label"]) == {"A", "B"}
    assert "C" not in late_scores["positive_label"].values
    assert late_scores.groupby(["feature_set", "comparison_id", "time_bin"]).size().max() == 1

    mc = result.layers["multiclass_predictions"]
    assert set(mc["time_bin"].unique()) == {24}


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


def test_binary_parallel_outputs_match_serial():
    df = _make_df(n_classes=2, n_embryos=4, n_times=3, n_features=2)
    serial = run_classification(
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
    parallel = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive="A",
        negative="B",
        features={"emb": "z_mu_b"},
        n_permutations=4,
        n_jobs=4,
        verbose=False,
    )

    serial_scores = serial.scores.sort_values(
        ["feature_set", "comparison_id", "time_bin_center"]
    ).reset_index(drop=True)
    parallel_scores = parallel.scores.sort_values(
        ["feature_set", "comparison_id", "time_bin_center"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(serial_scores, parallel_scores)


def test_binary_loop_clamps_workers_and_forces_inner_serial(monkeypatch):
    df = _make_df(n_classes=2, n_embryos=4, n_times=3, n_features=1)
    labeled = _build_binary_labels(
        df,
        class_col="genotype",
        comparison=type(
            "RC",
            (),
            {
                "positive_members": ["A"],
                "negative_members": ["B"],
            },
        )(),
    )
    df_binned = _bin_and_aggregate(
        labeled,
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        feature_cols=["z_mu_b_0"],
        bin_width=4.0,
    )

    observed = {"parallel_n_jobs": None, "inner_n_jobs": [], "time_bins": []}

    class FakeParallel:
        def __init__(self, n_jobs):
            observed["parallel_n_jobs"] = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return wrapper

    def fake_score_binary_ovr_bin(**kwargs):
        observed["inner_n_jobs"].append(kwargs["n_jobs"])
        observed["time_bins"].append(kwargs["time_bin"])
        return {
            "time_bin": kwargs["time_bin"],
            "time_bin_center": float(kwargs["time_bin"]) + kwargs["bin_width"] / 2.0,
            "bin_width": kwargs["bin_width"],
            "auroc_obs": 0.75,
            "pval": 0.25,
            "n_positive": 4,
            "n_negative": 4,
            "auroc_null_mean": 0.5,
            "auroc_null_std": 0.1,
            "n_permutations": kwargs["n_permutations"],
            "_null_array": np.array([0.5]),
            "_confusion_matrix": np.array([[1, 0], [0, 1]]),
            "_predictions": [],
        }

    monkeypatch.setattr(loop_module, "Parallel", FakeParallel)
    monkeypatch.setattr(loop_module, "delayed", fake_delayed)
    monkeypatch.setattr(loop_module, "joblib_effective_n_jobs", lambda n_jobs: n_jobs)
    monkeypatch.setattr(loop_module, "_score_binary_ovr_bin", fake_score_binary_ovr_bin)

    results = loop_module._run_binary_classification_loop(
        df_binned=df_binned,
        feature_cols=["z_mu_b_0"],
        id_col="embryo_id",
        bin_width=4.0,
        n_splits=3,
        n_permutations=4,
        n_jobs=8,
        random_state=42,
        verbose=False,
    )

    assert observed["parallel_n_jobs"] == 3
    assert observed["inner_n_jobs"] == [1, 1, 1]
    assert observed["time_bins"] == [24, 28, 32]
    assert [entry["time_bin"] for entry in results] == [24, 28, 32]


def test_binary_loop_single_bin_falls_back_to_serial(monkeypatch):
    df = _make_df(n_classes=2, n_embryos=4, n_times=1, n_features=1)
    labeled = _build_binary_labels(
        df,
        class_col="genotype",
        comparison=type(
            "RC",
            (),
            {
                "positive_members": ["A"],
                "negative_members": ["B"],
            },
        )(),
    )
    df_binned = _bin_and_aggregate(
        labeled,
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        feature_cols=["z_mu_b_0"],
        bin_width=4.0,
    )

    calls = {"parallel_called": False, "inner_n_jobs": []}

    def fail_parallel(*args, **kwargs):
        calls["parallel_called"] = True
        raise AssertionError("Parallel should not be used for a single-bin job")

    def fake_score_binary_ovr_bin(**kwargs):
        calls["inner_n_jobs"].append(kwargs["n_jobs"])
        return {
            "time_bin": kwargs["time_bin"],
            "time_bin_center": float(kwargs["time_bin"]) + kwargs["bin_width"] / 2.0,
            "bin_width": kwargs["bin_width"],
            "auroc_obs": 0.75,
            "pval": 0.25,
            "n_positive": 4,
            "n_negative": 4,
            "auroc_null_mean": 0.5,
            "auroc_null_std": 0.1,
            "n_permutations": kwargs["n_permutations"],
            "_null_array": np.array([0.5]),
            "_confusion_matrix": np.array([[1, 0], [0, 1]]),
            "_predictions": [],
        }

    monkeypatch.setattr(loop_module, "Parallel", fail_parallel)
    monkeypatch.setattr(loop_module, "joblib_effective_n_jobs", lambda n_jobs: n_jobs)
    monkeypatch.setattr(loop_module, "_score_binary_ovr_bin", fake_score_binary_ovr_bin)

    results = loop_module._run_binary_classification_loop(
        df_binned=df_binned,
        feature_cols=["z_mu_b_0"],
        id_col="embryo_id",
        bin_width=4.0,
        n_splits=3,
        n_permutations=4,
        n_jobs=8,
        random_state=42,
        verbose=False,
    )

    assert calls["parallel_called"] is False
    assert calls["inner_n_jobs"] == [1]
    assert [entry["time_bin"] for entry in results] == [24]


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
