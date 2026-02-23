#!/usr/bin/env python3
"""Difference detection, classification, and clustering on raw-field PCA embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold

def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.difference_detection.classification_test import run_binary_classification_test
from analyze.difference_detection.distribution_test import permutation_test_energy


DEFAULT_OUTPUT_ROOT = ANALYSIS_ROOT / "output"


def _jsonify(obj):
    """Recursively convert numpy/pandas scalars to plain JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _grouped_logreg_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: str,
    n_splits: int = 3,
) -> Dict:
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = (df[label_col].astype(str).to_numpy() == "mutant").astype(int)
    groups = df[group_col].astype(str).to_numpy()

    unique_groups = np.unique(groups)
    n_splits = int(min(max(2, n_splits), len(unique_groups)))
    gkf = GroupKFold(n_splits=n_splits)
    model = LogisticRegression(max_iter=3000, class_weight="balanced")

    all_idx = []
    all_prob = []
    all_true = []
    for tr, te in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        all_idx.append(te)
        all_prob.append(p)
        all_true.append(y[te])

    if not all_idx:
        raise ValueError("No valid GroupKFold splits for binary classification.")
    idx = np.concatenate(all_idx)
    prob = np.concatenate(all_prob)
    true = np.concatenate(all_true)
    pred = (prob >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(true, prob)),
        "accuracy": float(accuracy_score(true, pred)),
        "n_samples_eval": int(len(true)),
        "index": idx,
        "prob": prob,
        "true": true,
    }


def _embryo_label_shuffle_pvalue(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_perm: int,
    random_state: int,
) -> Dict:
    rng = np.random.default_rng(random_state)
    embryo_map = (
        df.groupby("embryo_id")["set_type"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )
    embryo_ids = np.array(sorted(embryo_map.keys()))
    labels = np.array([embryo_map[e] for e in embryo_ids], dtype=object)

    observed = _grouped_logreg_cv(df, feature_cols, label_col="set_type", group_col="embryo_id", n_splits=3)["auroc"]
    null = []
    for _ in range(n_perm):
        perm = labels.copy()
        rng.shuffle(perm)
        perm_map = {e: p for e, p in zip(embryo_ids, perm)}
        dperm = df.copy()
        dperm["set_type"] = dperm["embryo_id"].map(perm_map)
        if len(set(dperm["set_type"])) < 2:
            continue
        try:
            auc = _grouped_logreg_cv(dperm, feature_cols, label_col="set_type", group_col="embryo_id", n_splits=3)[
                "auroc"
            ]
            null.append(float(auc))
        except Exception:
            continue
    null_arr = np.asarray(null, dtype=np.float64)
    pval = float((1.0 + np.sum(null_arr >= observed)) / (len(null_arr) + 1.0))
    return {"observed_auroc": float(observed), "pvalue": pval, "null_distribution": null_arr}


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    pca_dir = output_root / "pca"
    out_dir = output_root / "difference_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = pca_dir / f"raw_velocity_pca_embeddings_{args.run_id}.csv"
    emb = pd.read_csv(emb_path)
    pc_cols = [c for c in emb.columns if c.startswith("PC")]
    if not pc_cols:
        raise ValueError(f"No PC columns found in {emb_path}")
    feat_cols = pc_cols[: min(args.n_pcs_for_tests, len(pc_cols))]

    # Primary binary cohort: mutants vs heldout WT (exclude reference WT from this binary test).
    binary_df = emb[emb["set_type"].isin(["mutant", "heldout_wt"])].copy()
    if binary_df.empty:
        raise ValueError("Binary dataframe is empty after filtering to mutant/heldout_wt.")

    # 1) Group-aware logistic CV
    cv_result = _grouped_logreg_cv(
        binary_df,
        feature_cols=feat_cols,
        label_col="set_type",
        group_col="embryo_id",
        n_splits=3,
    )

    # 2) Embryo-level permutation p-value for AUROC
    perm_result = _embryo_label_shuffle_pvalue(
        binary_df,
        feature_cols=feat_cols,
        n_perm=args.n_permutations,
        random_state=42,
    )
    np.save(out_dir / f"logreg_groupcv_null_{args.run_id}.npy", perm_result["null_distribution"])

    # 3) Distribution-level difference test (energy distance) on PCA embeddings
    X_mut = binary_df.loc[binary_df["set_type"] == "mutant", feat_cols].to_numpy(dtype=np.float64)
    X_wt = binary_df.loc[binary_df["set_type"] == "heldout_wt", feat_cols].to_numpy(dtype=np.float64)
    energy_result = permutation_test_energy(
        X_mut,
        X_wt,
        n_permutations=args.n_permutations,
        random_state=42,
    )
    np.save(out_dir / f"energy_null_{args.run_id}.npy", energy_result.null_distribution)

    # 4) Time-resolved binary classification test from difference_detection module
    dd_df = binary_df.copy()
    dd_df["group"] = dd_df["set_type"]
    dd_df["predicted_stage_hpf"] = dd_df["bin_mid_hpf"]
    dd_results = run_binary_classification_test(
        dd_df,
        group_col="group",
        group1="mutant",
        group2="heldout_wt",
        features=feat_cols,
        morphology_metric=None,
        time_col="predicted_stage_hpf",
        embryo_id_col="embryo_id",
        bin_width=2.0,
        n_splits=3,
        n_permutations=args.n_permutations,
        n_jobs=1,
        min_samples_per_bin=2,
        within_bin_time_stratification=True,
        within_bin_time_strata_width=0.5,
        random_state=42,
        verbose=False,
    )
    dd_classification = dd_results["classification"]
    dd_classification.to_csv(out_dir / f"time_resolved_classification_{args.run_id}.csv", index=False)
    if dd_results.get("diagnostics") is not None:
        dd_results["diagnostics"].to_csv(out_dir / f"time_resolved_diagnostics_{args.run_id}.csv", index=False)

    # 5) Clustering
    km = KMeans(n_clusters=2, random_state=42, n_init=20)
    emb["cluster_k2"] = km.fit_predict(emb[feat_cols].to_numpy(dtype=np.float64))
    sil = float(silhouette_score(emb[feat_cols], emb["cluster_k2"]))
    cluster_ct = pd.crosstab(emb["cluster_k2"], emb["set_type"])
    emb.to_csv(out_dir / f"embeddings_with_clusters_{args.run_id}.csv", index=False)
    cluster_ct.to_csv(out_dir / f"cluster_vs_settype_{args.run_id}.csv")

    # Plots
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(emb["PC1"], emb["PC2"], c=emb["cluster_k2"], cmap="tab10", s=26, alpha=0.8)
    ax.set_title("KMeans(k=2) on Raw-Field PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / f"kmeans_k2_pca_scatter_{args.run_id}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(perm_result["null_distribution"], bins=30, alpha=0.7, color="#4C78A8")
    ax.axvline(perm_result["observed_auroc"], color="red", linestyle="--", linewidth=2)
    ax.set_title("Group-CV AUROC Permutation Null")
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / f"logreg_auroc_permutation_{args.run_id}.png", dpi=180)
    plt.close(fig)

    # Summary
    summary = {
        "run_id": args.run_id,
        "features_used": feat_cols,
        "group_cv": {
            "auroc": cv_result["auroc"],
            "accuracy": cv_result["accuracy"],
            "n_samples_eval": cv_result["n_samples_eval"],
        },
        "group_cv_permutation": {
            "observed_auroc": perm_result["observed_auroc"],
            "pvalue": perm_result["pvalue"],
            "n_permutations": int(args.n_permutations),
        },
        "energy_test": {
            "statistic": float(energy_result.observed),
            "pvalue": float(energy_result.pvalue),
            "n_permutations": int(args.n_permutations),
        },
        "difference_detection_summary": dd_results["summary"],
        "kmeans_k2": {
            "silhouette_score": sil,
            "cluster_vs_settype_csv": str(out_dir / f"cluster_vs_settype_{args.run_id}.csv"),
        },
    }
    summary = _jsonify(summary)
    with open(out_dir / f"difference_classification_summary_{args.run_id}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Difference/classification/clustering on PCA embeddings.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default="phase2_24_48_ott_v1")
    parser.add_argument("--n-pcs-for-tests", type=int, default=6)
    parser.add_argument("--n-permutations", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
