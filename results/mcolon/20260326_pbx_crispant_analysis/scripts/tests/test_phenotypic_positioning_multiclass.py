from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from phenotypic_positioning.multiclass import (
    build_multiclass_logit_vectors,
    build_multiclass_probability_vectors,
    prepare_multiclass_confusion_summary,
    prepare_multiclass_predictions,
    summarize_multiclass_centroids,
    summarize_probability_trajectories,
)


def _make_predictions() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    class_labels = ["A", "B", "C"]
    pred_df = pd.DataFrame(
        [
            {
                "embryo_id": "A_0",
                "time_bin": 20,
                "time_bin_center": 21.0,
                "true_class": "A",
                "pred_class": "A",
                "is_correct": True,
                "is_wrong": 0,
                "p_true": 0.8,
                "p_pred": 0.8,
                "feature_set": "z_mu_b",
                "pred_proba_A": 0.8,
                "pred_proba_B": 0.1,
                "pred_proba_C": 0.1,
            },
            {
                "embryo_id": "B_0",
                "time_bin": 20,
                "time_bin_center": 21.0,
                "true_class": "B",
                "pred_class": "B",
                "is_correct": True,
                "is_wrong": 0,
                "p_true": 0.7,
                "p_pred": 0.7,
                "feature_set": "z_mu_b",
                "pred_proba_A": 0.1,
                "pred_proba_B": 0.7,
                "pred_proba_C": 0.2,
            },
            {
                "embryo_id": "C_0",
                "time_bin": 22,
                "time_bin_center": 23.0,
                "true_class": "C",
                "pred_class": "B",
                "is_correct": False,
                "is_wrong": 1,
                "p_true": 0.3,
                "p_pred": 0.5,
                "feature_set": "z_mu_b",
                "pred_proba_A": 0.2,
                "pred_proba_B": 0.5,
                "pred_proba_C": 0.3,
            },
        ]
    )
    embryo_meta = pd.DataFrame(
        {
            "embryo_id": ["A_0", "B_0", "C_0"],
            "true_label": ["A", "B", "C"],
            "experiment_id": ["exp1", "exp1", "exp2"],
        }
    )
    return pred_df, embryo_meta, class_labels


def test_prepare_multiclass_predictions_and_vectors():
    pred_df, embryo_meta, class_labels = _make_predictions()
    prepared, prob_cols = prepare_multiclass_predictions(
        pred_df,
        embryo_meta=embryo_meta,
        class_labels=class_labels,
    )
    vectors, vector_cols = build_multiclass_probability_vectors(prepared, class_labels=class_labels)
    logits, logit_cols = build_multiclass_logit_vectors(prepared, class_labels=class_labels)

    assert prob_cols == ["pred_proba_A", "pred_proba_B", "pred_proba_C"]
    assert vector_cols == prob_cols
    assert logit_cols == ["logit_A", "logit_B", "logit_C"]
    assert len(vectors) == 3
    assert (vectors[prob_cols].sum(axis=1) - 1.0).abs().max() < 1e-9
    assert set(vectors["experiment_id"]) == {"exp1", "exp2"}
    assert np.isfinite(logits[logit_cols].to_numpy(dtype=float)).all()


def test_multiclass_confusion_and_centroids():
    pred_df, embryo_meta, class_labels = _make_predictions()
    prepared, _ = prepare_multiclass_predictions(
        pred_df,
        embryo_meta=embryo_meta,
        class_labels=class_labels,
    )
    vectors, _ = build_multiclass_probability_vectors(prepared, class_labels=class_labels)
    centroids, distances = summarize_multiclass_centroids(vectors, class_labels=class_labels)
    trajectories = summarize_probability_trajectories(vectors, class_labels=class_labels)

    confusion_df = pd.DataFrame(
        [
            {
                "feature_set": "z_mu_b",
                "time_bin": 20,
                "time_bin_center": 21.0,
                "true_class": "A",
                "predicted_class": "A",
                "proportion": 1.0,
                "count": 1,
                "is_correct": True,
            },
            {
                "feature_set": "z_mu_b",
                "time_bin": 20,
                "time_bin_center": 21.0,
                "true_class": "B",
                "predicted_class": "B",
                "proportion": 1.0,
                "count": 1,
                "is_correct": True,
            },
        ]
    )
    prepared_confusion = prepare_multiclass_confusion_summary(
        confusion_df,
        class_labels=class_labels,
        bin_width=2.0,
    )

    assert set(centroids["genotype"]) == {"A", "B", "C"}
    assert {"distance_l1", "distance_l2"} <= set(distances.columns)
    assert (distances["distance_l2"] >= 0.0).all()
    assert set(trajectories["predicted_class"]) == {"A", "B", "C"}
    assert len(prepared_confusion) == 2


def test_multiclass_confusion_reconstructs_time_bin_from_center():
    confusion_df = pd.DataFrame(
        [
            {
                "feature_set": "z_mu_b",
                "time_bin_center": 21.0,
                "true_class": "A",
                "predicted_class": "A",
                "proportion": 1.0,
                "count": 2,
                "is_correct": True,
            }
        ]
    )
    prepared_confusion = prepare_multiclass_confusion_summary(
        confusion_df,
        class_labels=["A", "B"],
        bin_width=2.0,
    )
    assert int(prepared_confusion.loc[0, "time_bin"]) == 20
