from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import pair_id
from .io import save_pair_model_bundle
from .support import add_support_weights, build_support_reference, score_support_metrics, summarize_feature_support


def _make_logistic_classifier(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        multi_class="ovr",
        class_weight="balanced",
        random_state=random_state,
    )


def _make_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", _make_logistic_classifier(random_state=random_state)),
        ]
    )


def _safe_logit(prob: np.ndarray) -> np.ndarray:
    return logit(np.clip(prob, 1e-6, 1.0 - 1e-6))


def _bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(y)
    if n <= 1:
        return np.arange(n, dtype=int)
    for _ in range(100):
        idx = rng.integers(0, n, size=n, endpoint=False)
        if len(np.unique(y[idx])) >= 2:
            return idx
    return np.arange(n, dtype=int)


def _fit_bootstrap_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstraps: int,
    seed: int,
) -> list[Pipeline]:
    rng = np.random.default_rng(seed)
    models: list[Pipeline] = []
    for bootstrap_idx in range(int(n_bootstraps)):
        sample_idx = _bootstrap_indices(y, rng)
        model = _make_pipeline(random_state=seed + bootstrap_idx)
        model.fit(X[sample_idx], y[sample_idx])
        models.append(model)
    return models


def _predict_ensemble(models: list[Pipeline], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        empty = np.array([], dtype=float)
        return empty, empty
    probs = np.stack([model.predict_proba(X)[:, 1] for model in models], axis=1)
    logits = _safe_logit(probs)
    return logits.mean(axis=1), logits.std(axis=1, ddof=0)


def _summarize_model_coefficients(models: list[Pipeline], feature_cols: list[str]) -> pd.DataFrame:
    if not models:
        return pd.DataFrame(
            columns=[
                "feature",
                "coef_mean",
                "coef_sd",
                "coef_abs_mean",
                "sign_consistency",
            ]
        )
    coefs = np.stack([model.named_steps["clf"].coef_[0] for model in models], axis=0)
    sign_consistency = np.abs(np.mean(np.sign(coefs), axis=0))
    return pd.DataFrame(
        {
            "feature": feature_cols,
            "coef_mean": coefs.mean(axis=0),
            "coef_sd": coefs.std(axis=0, ddof=0),
            "coef_abs_mean": np.abs(coefs).mean(axis=0),
            "sign_consistency": sign_consistency,
        }
    )


def _oof_bootstrap_predictions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    embryo_ids: np.ndarray,
    n_splits: int,
    n_bootstraps: int,
    random_state: int,
) -> pd.DataFrame:
    min_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits_actual = min(int(n_splits), min_count)
    if n_splits_actual < 2:
        raise ValueError("Need at least 2 samples per class for OOF scoring.")

    cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=random_state)
    mean_logits = np.full(len(y), np.nan, dtype=float)
    std_logits = np.full(len(y), np.nan, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        models = _fit_bootstrap_models(
            X[train_idx],
            y[train_idx],
            n_bootstraps=n_bootstraps,
            seed=random_state + 1000 * (fold_idx + 1),
        )
        fold_mean, fold_std = _predict_ensemble(models, X[test_idx])
        mean_logits[test_idx] = fold_mean
        std_logits[test_idx] = fold_std

    out = pd.DataFrame(
        {
            "embryo_id": embryo_ids.astype(str),
            "y_true_binary": y.astype(int),
            "position_logit_mean": mean_logits,
            "position_logit_sd": std_logits,
        }
    )
    out["position_probability_mean"] = expit(out["position_logit_mean"])
    return out


def _build_unavailable_rows(
    df_bin: pd.DataFrame,
    *,
    current_pair_id: str,
    group1: str,
    group2: str,
    counts: dict[str, int],
) -> pd.DataFrame:
    out = df_bin[["embryo_id", "genotype", "experiment_id", "_time_bin", "time_bin_center"]].copy()
    out["pair_id"] = current_pair_id
    out["group1"] = group1
    out["group2"] = group2
    out["score_role"] = "unavailable"
    out["model_available"] = False
    out["position_logit_mean"] = np.nan
    out["position_logit_sd"] = np.nan
    out["position_probability_mean"] = np.nan
    out["axis_projection"] = np.nan
    out["axis_residual"] = np.nan
    out["axis_residual_z"] = np.nan
    out["knn_novelty"] = np.nan
    out["knn_novelty_z"] = np.nan
    out["n_group1"] = int(counts.get(group1, 0))
    out["n_group2"] = int(counts.get(group2, 0))
    return out


def run_pairwise_support_analysis(
    df_binned: pd.DataFrame,
    *,
    feature_cols: list[str],
    genotypes: list[str],
    n_splits: int,
    n_bootstraps: int,
    random_state: int,
    k_neighbors: int,
    models_dir=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_axis_rows: list[pd.DataFrame] = []
    all_score_rows: list[dict[str, Any]] = []
    model_index_rows: list[dict[str, Any]] = []
    feature_support_rows: list[pd.DataFrame] = []
    coefficient_rows: list[pd.DataFrame] = []

    present = [g for g in genotypes if g in set(df_binned["genotype"].unique())]
    time_bins = sorted(df_binned["_time_bin"].unique())

    for pair_idx, (group1, group2) in enumerate(combinations(present, 2), start=1):
        current_pair_id = pair_id(group1, group2)
        time_models: dict[int, dict[str, Any]] = {}

        for time_idx, time_bin in enumerate(time_bins):
            df_bin = df_binned[df_binned["_time_bin"] == time_bin].reset_index(drop=True)
            df_pair = df_bin[df_bin["genotype"].isin([group1, group2])].reset_index(drop=True)
            counts = df_pair["genotype"].value_counts().to_dict()

            if counts.get(group1, 0) < 2 or counts.get(group2, 0) < 2:
                all_axis_rows.append(
                    _build_unavailable_rows(
                        df_bin,
                        current_pair_id=current_pair_id,
                        group1=group1,
                        group2=group2,
                        counts=counts,
                    )
                )
                continue

            X_pair = df_pair[feature_cols].to_numpy(dtype=float)
            y_pair = (df_pair["genotype"] == group2).astype(int).to_numpy()
            embryo_pair_ids = df_pair["embryo_id"].astype(str).to_numpy()

            oof_df = _oof_bootstrap_predictions(
                X_pair,
                y_pair,
                embryo_ids=embryo_pair_ids,
                n_splits=n_splits,
                n_bootstraps=n_bootstraps,
                random_state=random_state + 10000 * pair_idx + 100 * time_idx,
            )

            full_models = _fit_bootstrap_models(
                X_pair,
                y_pair,
                n_bootstraps=n_bootstraps,
                seed=random_state + 500000 + 10000 * pair_idx + 100 * time_idx,
            )

            support_reference = build_support_reference(
                X_pair,
                y_pair,
                feature_cols=feature_cols,
                group1=group1,
                group2=group2,
                k_neighbors=k_neighbors,
            )

            support_rows = score_support_metrics(
                support_reference,
                df_bin,
                feature_cols=feature_cols,
            )
            feature_summary = summarize_feature_support(
                support_reference,
                df_bin,
                feature_cols=feature_cols,
            )
            feature_summary["pair_id"] = current_pair_id
            feature_summary["group1"] = group1
            feature_summary["group2"] = group2
            feature_summary["_time_bin"] = int(time_bin)
            feature_summary["time_bin_center"] = float(df_bin["time_bin_center"].iloc[0])
            feature_support_rows.append(feature_summary)

            pair_member_ids = set(df_pair["embryo_id"].astype(str))
            df_out = df_bin[~df_bin["embryo_id"].astype(str).isin(pair_member_ids)].copy()
            if len(df_out) > 0:
                probe_mean, probe_sd = _predict_ensemble(full_models, df_out[feature_cols].to_numpy(dtype=float))
                probe_df = df_out[["embryo_id"]].copy()
                probe_df["position_logit_mean"] = probe_mean
                probe_df["position_logit_sd"] = probe_sd
                probe_df["position_probability_mean"] = expit(probe_mean)
            else:
                probe_df = pd.DataFrame(columns=["embryo_id", "position_logit_mean", "position_logit_sd", "position_probability_mean"])

            score_rows = pd.concat(
                [
                    oof_df.assign(score_role="in_pair_oof"),
                    probe_df.assign(score_role="out_pair_probe"),
                ],
                ignore_index=True,
            )
            merged = support_rows.merge(score_rows, on="embryo_id", how="left")
            merged["pair_id"] = current_pair_id
            merged["group1"] = group1
            merged["group2"] = group2
            merged["model_available"] = True
            merged["n_group1"] = int(counts.get(group1, 0))
            merged["n_group2"] = int(counts.get(group2, 0))
            merged["score_role"] = merged["score_role"].fillna("out_pair_probe")
            all_axis_rows.append(merged)

            oof_auc = float(roc_auc_score(oof_df["y_true_binary"], oof_df["position_probability_mean"]))
            all_score_rows.append(
                {
                    "pair_id": current_pair_id,
                    "group1": group1,
                    "group2": group2,
                    "time_bin": int(time_bin),
                    "time_bin_center": float(df_bin["time_bin_center"].iloc[0]),
                    "n_group1": int(counts.get(group1, 0)),
                    "n_group2": int(counts.get(group2, 0)),
                    "oof_auroc": oof_auc,
                    "n_bootstraps": int(n_bootstraps),
                }
            )

            time_models[int(time_bin)] = {
                "pair_id": current_pair_id,
                "group1": group1,
                "group2": group2,
                "time_bin": int(time_bin),
                "time_bin_center": float(df_bin["time_bin_center"].iloc[0]),
                "feature_cols": list(feature_cols),
                "n_group1": int(counts.get(group1, 0)),
                "n_group2": int(counts.get(group2, 0)),
                "bootstrap_models": full_models,
                "support_reference": support_reference,
                "training_embryo_ids": df_pair["embryo_id"].astype(str).tolist(),
                "training_genotypes": df_pair["genotype"].astype(str).tolist(),
            }
            coef_summary = _summarize_model_coefficients(full_models, feature_cols)
            coef_summary["pair_id"] = current_pair_id
            coef_summary["group1"] = group1
            coef_summary["group2"] = group2
            coef_summary["_time_bin"] = int(time_bin)
            coef_summary["time_bin_center"] = float(df_bin["time_bin_center"].iloc[0])
            coefficient_rows.append(coef_summary)

        if models_dir is not None:
            bundle = {
                "pair_id": current_pair_id,
                "group1": group1,
                "group2": group2,
                "feature_cols": list(feature_cols),
                "time_models": time_models,
                "n_bootstraps": int(n_bootstraps),
                "n_splits": int(n_splits),
                "random_state": int(random_state),
                "k_neighbors": int(k_neighbors),
            }
            model_path = save_pair_model_bundle(models_dir, current_pair_id, bundle)
            model_index_rows.append(
                {
                    "pair_id": current_pair_id,
                    "group1": group1,
                    "group2": group2,
                    "model_path": str(model_path),
                    "n_time_models": int(len(time_models)),
                }
            )

    axis_df = pd.concat(all_axis_rows, ignore_index=True)
    axis_df = add_support_weights(axis_df)
    score_df = pd.DataFrame(all_score_rows).sort_values(["pair_id", "time_bin"]).reset_index(drop=True)
    model_index_df = pd.DataFrame(model_index_rows).sort_values("pair_id").reset_index(drop=True)
    feature_support_df = (
        pd.concat(feature_support_rows, ignore_index=True)
        if feature_support_rows
        else pd.DataFrame()
    )
    if not feature_support_df.empty:
        feature_support_df = feature_support_df.sort_values(["pair_id", "_time_bin", "genotype", "feature"]).reset_index(drop=True)
    coefficient_df = pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else pd.DataFrame()
    if not coefficient_df.empty:
        coefficient_df = coefficient_df.sort_values(["pair_id", "_time_bin", "feature"]).reset_index(drop=True)
    return axis_df, score_df, model_index_df, feature_support_df, coefficient_df
