from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from phenotypic_positioning.pairwise import run_pairwise_support_analysis
from phenotypic_positioning.support import build_support_reference, score_support_metrics


def _make_binned_df() -> pd.DataFrame:
    rows = []
    genotypes = {
        "A": [(-2.0, 0.0), (-1.6, 0.1), (-1.8, -0.1)],
        "B": [(2.0, 0.0), (1.7, 0.2), (1.9, -0.1)],
        "C": [(0.0, 2.0), (0.3, 2.3), (-0.2, 1.9)],
    }
    for genotype, coords in genotypes.items():
        for idx, (x0, x1) in enumerate(coords):
            rows.append(
                {
                    "embryo_id": f"{genotype}_{idx}",
                    "genotype": genotype,
                    "experiment_id": "exp",
                    "_time_bin": 20,
                    "time_bin_center": 21.0,
                    "z_mu_b_0": x0,
                    "z_mu_b_1": x1,
                }
            )
    return pd.DataFrame(rows)


def test_pairwise_support_outputs_are_dense(tmp_path):
    df_binned = _make_binned_df()
    axis_df, score_df, model_index_df, feature_support_df, coefficient_df = run_pairwise_support_analysis(
        df_binned,
        feature_cols=["z_mu_b_0", "z_mu_b_1"],
        genotypes=["A", "B", "C"],
        n_splits=3,
        n_bootstraps=3,
        random_state=42,
        k_neighbors=2,
        models_dir=tmp_path / "models",
    )

    assert len(axis_df) == len(df_binned) * 3
    assert set(axis_df["pair_id"].unique()) == {"A_vs_B", "A_vs_C", "B_vs_C"}
    ab = axis_df[axis_df["pair_id"] == "A_vs_B"].copy()
    assert set(ab.loc[ab["genotype"].isin(["A", "B"]), "score_role"]) == {"in_pair_oof"}
    assert set(ab.loc[ab["genotype"] == "C", "score_role"]) == {"out_pair_probe"}
    assert set(model_index_df["pair_id"]) == {"A_vs_B", "A_vs_C", "B_vs_C"}
    assert score_df["oof_auroc"].between(0.0, 1.0).all()
    assert set(feature_support_df["feature"]) == {"z_mu_b_0", "z_mu_b_1"}
    assert set(coefficient_df["feature"]) == {"z_mu_b_0", "z_mu_b_1"}
    assert "__all__" in set(feature_support_df["genotype"])


def test_support_metrics_raise_for_off_axis_points():
    X_train = np.array(
        [
            [-2.0, 0.0],
            [-1.8, 0.1],
            [-1.6, -0.1],
            [1.6, 0.0],
            [1.8, 0.1],
            [2.0, -0.1],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    reference = build_support_reference(
        X_train,
        y,
        feature_cols=["z0", "z1"],
        group1="A",
        group2="B",
        k_neighbors=2,
    )

    queries = pd.DataFrame(
        {
            "embryo_id": ["on_axis", "off_axis"],
            "genotype": ["A", "C"],
            "experiment_id": ["exp", "exp"],
            "_time_bin": [20, 20],
            "time_bin_center": [21.0, 21.0],
            "z0": [0.0, 0.0],
            "z1": [0.0, 3.0],
        }
    )
    scored = score_support_metrics(reference, queries, feature_cols=["z0", "z1"]).set_index("embryo_id")
    assert scored.loc["off_axis", "axis_residual"] > scored.loc["on_axis", "axis_residual"]
    assert scored.loc["off_axis", "knn_novelty"] > scored.loc["on_axis", "knn_novelty"]
