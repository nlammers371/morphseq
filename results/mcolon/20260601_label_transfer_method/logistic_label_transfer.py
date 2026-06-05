"""
Logistic label transfer for CEP290 phenotypes.

Uses the same LogisticRegression spec as src/analyze/classification (liblinear,
balanced class weight, StandardScaler) to transfer labels from a reference
DataFrame to a query DataFrame.

Three model granularities are benchmarked in a single call:

    A — Global:        one model fit on all reference images (no time binning)
    B — Per-bin/image: one model per HPF bin, one row per image in training
    C — Per-bin/embryo: one model per HPF bin, embryo-mean features in training
                        (standard run_classification behaviour; prevents embryo leakage)

Two rollup methods aggregate per-image probability vectors to embryo-level predictions:

    mean:           simple mean of q vectors across all images → argmax
    margin_weighted: weight each q vector by max(0, top1_prob - top2_prob) before
                     averaging; down-weights uncertain early timepoints automatically

All 6 combinations (3 modes × 2 rollups) are stored in embryo_label_transfer_summary.

Migration note
--------------
When this moves into src/analyze/classification/, the three fit+predict blocks become
a predict_transfer(query_df, mode=...) method on ClassificationAnalysis, and the
rollup helpers move into engine/. The model spec is identical to the existing
_make_logistic_classifier in directions/fit.py (liblinear, balanced, max_iter=1000).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MORPHSEQ_ROOT = HERE.parents[2]
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from label_transfer_core import (   # noqa: E402
    compute_reference_label_profile,
    _validate_feature_cols,
    _attach_label_profile,
    _assign_status,
)

MAIN_LABELS = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
MIN_BIN_SAMPLES = 20      # minimum reference images per bin to use a bin model
MIN_BIN_CLASSES = 2       # minimum classes represented per bin


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_logistic_label_transfer(
    reference_df: pd.DataFrame,
    query_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "cluster_categories",
    embryo_col: str = "embryo_id",
    snip_col: str = "snip_id",
    time_col: str = "predicted_stage_hpf",
    min_hpf: float = 0.0,
    max_hpf: float = 9999.0,
    bin_width: float = 4.0,
    random_state: int = 42,
    margin_threshold: float | None = None,
    agreement_threshold: float = 0.5,
    consistency_threshold: float = 0.6,
    distance_score_threshold: float = 0.05,
    top_probability_threshold: float = 0.4,
) -> dict:
    """Transfer phenotype labels from reference to query using logistic regression.

    Parameters
    ----------
    reference_df : Labeled reference images. Must have feature_cols, label_col,
        embryo_col, snip_col, time_col.
    query_df : Images to classify. Must have feature_cols, embryo_col, snip_col,
        time_col. Labels are never read from query_df.
    feature_cols : Numeric embedding columns (e.g. z_mu_b_*).
    label_col : Column in reference_df with phenotype labels.
    embryo_col, snip_col, time_col : Column names.
    min_hpf, max_hpf : Time window applied to both DataFrames.
    bin_width : HPF bin width for per-bin models (modes B and C).
    random_state : Sklearn random state.
    margin_threshold : If set, embryos with embryo-level argmax_margin < τ get
        status="ambiguous", reason="low_margin".
    agreement_threshold, consistency_threshold, distance_score_threshold,
    top_probability_threshold : Status assignment thresholds (passed to _assign_status).

    Returns
    -------
    dict with keys:
        embryo_label_transfer_summary : one row per query embryo, columns
            predicted_label_{A/B/C}_{mean/margin} and margin_{A/B/C}_{mean/margin},
            plus confidence and status columns (based on mode C margin rollup as primary).
        image_predictions : one row per (query image × mode), columns
            snip_id, embryo_id, bin_center, pred_proba_{label}..., argmax_label,
            argmax_margin, mode.
        embryo_label_probabilities : long format (embryo × label × mode × rollup).
        reference_label_profile : per-label geometry from compute_reference_label_profile.
        neighbor_long_table : empty DataFrame (no KNN in this method).
        image_prediction_summary : alias for image_predictions (compat with run_label_transfer).
    """
    _validate_feature_cols(reference_df, query_df, feature_cols)

    # ------------------------------------------------------------------
    # Step 1: filter + prep
    # ------------------------------------------------------------------
    ref = reference_df.copy()
    ref = ref[ref[label_col].notna()].copy()
    ref = ref[(ref[time_col] >= min_hpf) & (ref[time_col] <= max_hpf)].copy()
    ref = ref.reset_index(drop=True)

    qry = query_df.copy()
    qry = qry[(qry[time_col] >= min_hpf) & (qry[time_col] <= max_hpf)].copy()
    qry = qry.reset_index(drop=True)

    if len(ref) == 0:
        raise ValueError("No reference rows remain after filtering.")
    if len(qry) == 0:
        warnings.warn("No query rows remain after filtering.")
        return _empty_results()

    ref_X = ref[feature_cols].to_numpy(dtype=float)
    qry_X = qry[feature_cols].to_numpy(dtype=float)
    ref_labels = ref[label_col].to_numpy(dtype=str)

    # Determine label order from reference
    label_order = sorted(set(ref_labels))

    # snip → embryo lookup for query
    snip_to_embryo = dict(zip(qry[snip_col], qry[embryo_col]))

    # Reference label geometry (reused from label_transfer_core)
    label_profile = compute_reference_label_profile(ref_X, ref_labels, k=15)

    # ------------------------------------------------------------------
    # Step 2: fit and predict for each mode
    # ------------------------------------------------------------------
    mode_preds: dict[str, pd.DataFrame] = {}

    # Mode A — global model (one bin = all data)
    mode_preds["A"] = _predict_global(
        ref_X, ref_labels, qry_X, qry, label_order,
        snip_col, embryo_col, time_col, snip_to_embryo,
        random_state, feature_cols=feature_cols,
    )

    # Mode B — per-bin model, per-image training rows
    mode_preds["B"] = _predict_perbin(
        ref, qry, ref_X, ref_labels, qry_X, label_order,
        snip_col, embryo_col, time_col, None,
        snip_to_embryo, bin_width, min_hpf, max_hpf,
        random_state, use_embryo_mean=False,
        feature_cols=feature_cols, label_col=label_col,
    )

    # Mode C — per-bin model, embryo-mean training rows (standard)
    mode_preds["C"] = _predict_perbin(
        ref, qry, ref_X, ref_labels, qry_X, label_order,
        snip_col, embryo_col, time_col, None,
        snip_to_embryo, bin_width, min_hpf, max_hpf,
        random_state, use_embryo_mean=True,
        feature_cols=feature_cols, label_col=label_col,
    )

    # ------------------------------------------------------------------
    # Step 3: rollup to embryo level for each mode × rollup combination
    # ------------------------------------------------------------------
    all_embryo_rows: list[pd.DataFrame] = []
    prob_long_rows: list[pd.DataFrame] = []

    for mode, img_df in mode_preds.items():
        proba_cols = [c for c in img_df.columns if c.startswith("pred_proba_")]
        for rollup in ("mean", "margin"):
            emb_df = _rollup_embryo(img_df, embryo_col, proba_cols, label_order, rollup)
            emb_df = emb_df.rename(columns={
                "predicted_label": f"predicted_label_{mode}_{rollup}",
                "embryo_margin":   f"margin_{mode}_{rollup}",
                "top_probability": f"top_prob_{mode}_{rollup}",
            })
            all_embryo_rows.append(emb_df)

            # Long probabilities
            for _, row in emb_df.iterrows():
                for lbl in label_order:
                    col = f"prob_{lbl}"
                    prob_long_rows.append({
                        "query_embryo_id": row["query_embryo_id"],
                        "label": lbl,
                        "embryo_label_probability": row.get(col, np.nan),
                        "mode": mode,
                        "rollup": rollup,
                    })

    # Merge all rollup columns onto a single per-embryo table
    # Only carry the renamed prediction/margin/top_prob columns — not the raw prob_ or n_images
    all_embryos = qry[embryo_col].unique()
    summary = pd.DataFrame({"query_embryo_id": all_embryos})
    for emb_df in all_embryo_rows:
        keep = ["query_embryo_id"] + [
            c for c in emb_df.columns
            if c.startswith("predicted_label_") or c.startswith("margin_") or c.startswith("top_prob_")
        ]
        summary = summary.merge(emb_df[keep], on="query_embryo_id", how="left")

    # ------------------------------------------------------------------
    # Step 4: confidence + status (using mode C margin rollup as primary)
    # ------------------------------------------------------------------
    # Distance score: KNN distance of query images vs reference distribution
    summary = _add_distance_confidence(summary, ref_X, qry_X, qry, embryo_col, snip_col)

    # Consistency score: fraction of per-image argmaxes (mode C) matching embryo pred
    primary_pred_col = "predicted_label_C_margin"
    if primary_pred_col in summary.columns:
        img_c = mode_preds["C"][["query_embryo_id", "argmax_label"]].copy()
        img_c = img_c.merge(
            summary[["query_embryo_id", primary_pred_col]], on="query_embryo_id", how="left"
        )
        img_c["agrees"] = img_c["argmax_label"] == img_c[primary_pred_col]
        consistency = (
            img_c.groupby("query_embryo_id")["agrees"]
            .mean()
            .reset_index()
            .rename(columns={"agrees": "embryo_consistency_score"})
        )
        summary = summary.merge(consistency, on="query_embryo_id", how="left")
    else:
        summary["embryo_consistency_score"] = np.nan

    # top_label_probability: use mode C margin rollup top prob
    summary["top_label_probability"] = summary.get(
        "top_prob_C_margin", pd.Series(np.nan, index=summary.index)
    )
    # mean_image_neighbor_agreement: not applicable for logistic; set to 1.0 so it
    # never triggers a spurious "low_agreement" flag from the KNN-based threshold
    summary["mean_image_neighbor_agreement"] = 1.0
    # embryo_distance_score: no KNN distance; set to 1.0 so is_low_density is never True
    summary["embryo_distance_score"] = 1.0

    if "n_images" not in summary.columns:
        if not mode_preds.get("C", pd.DataFrame()).empty:
            n_img = (
                mode_preds["C"].groupby("query_embryo_id")["query_snip_id"]
                .nunique().reset_index().rename(columns={"query_snip_id": "n_images"})
            )
            summary = summary.merge(n_img, on="query_embryo_id", how="left")
        else:
            summary["n_images"] = 1

    # Logistic-specific status: flag on low top probability or low consistency only
    summary["is_low_density"] = False
    summary["is_low_agreement"] = False
    summary["is_low_consistency"] = (
        summary["embryo_consistency_score"].fillna(1.0) < consistency_threshold
    )
    summary["is_low_top_probability"] = (
        summary["top_label_probability"].fillna(0.0) < top_probability_threshold
    )
    ambiguous = summary["is_low_consistency"] | summary["is_low_top_probability"]
    summary["status"] = np.where(ambiguous, "ambiguous", "assigned")
    summary["status_reason"] = ""

    # Margin gate override
    if margin_threshold is not None and "margin_C_margin" in summary.columns:
        below = summary["margin_C_margin"] < margin_threshold
        summary.loc[below, "status"] = "ambiguous"
        summary.loc[below, "status_reason"] = "low_margin"

    # Primary predicted_label column (mode C, margin rollup)
    if "predicted_label_C_margin" in summary.columns:
        summary["predicted_label"] = summary["predicted_label_C_margin"]
    else:
        summary["predicted_label"] = np.nan

    # Attach reference label profile
    summary = _attach_label_profile(summary, label_profile)

    # ------------------------------------------------------------------
    # Step 5: assemble image_predictions (all modes stacked)
    # ------------------------------------------------------------------
    image_pred_frames = []
    for mode, img_df in mode_preds.items():
        img_df = img_df.copy()
        img_df["mode"] = mode
        image_pred_frames.append(img_df)
    image_predictions = pd.concat(image_pred_frames, ignore_index=True, sort=False)

    embryo_label_probs = pd.DataFrame(prob_long_rows)

    return {
        "embryo_label_transfer_summary": summary,
        "image_predictions": image_predictions,
        "image_prediction_summary": image_predictions,  # compat alias
        "embryo_label_probabilities": embryo_label_probs,
        "reference_label_profile": label_profile,
        "neighbor_long_table": pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# Internal: model fitting helpers
# ---------------------------------------------------------------------------

def _make_pipeline(n_classes: int, random_state: int) -> Any:
    """Multiclass logistic pipeline matching run_q_conformal_benchmark.py spec.

    Uses OneVsRestClassifier(LogisticRegression(liblinear)) — same as the benchmark
    that established logistic > KNN for this task.
    """
    return make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            )
        ),
    )


def _predict_proba_full(model: Any, X: np.ndarray, label_order: list[str]) -> np.ndarray:
    """Predict proba, expanding to full label_order even if some classes absent."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    out = np.zeros((len(X), len(label_order)), dtype=float)
    for j, lbl in enumerate(label_order):
        if lbl in classes:
            out[:, j] = proba[:, classes.index(lbl)]
    return out


def _image_df_from_proba(
    q: np.ndarray,
    qry: pd.DataFrame,
    label_order: list[str],
    snip_col: str,
    embryo_col: str,
    time_col: str,
    snip_to_embryo: dict,
    bin_center: float,
) -> pd.DataFrame:
    """Build image-level prediction rows from a (n_images, n_labels) proba matrix."""
    sorted_q = np.sort(q, axis=1)[:, ::-1]
    argmax_idx = q.argmax(axis=1)
    margin = sorted_q[:, 0] - sorted_q[:, 1]

    rows = []
    for i in range(len(qry)):
        row = {
            "query_snip_id": qry[snip_col].iloc[i],
            "query_embryo_id": snip_to_embryo.get(qry[snip_col].iloc[i],
                                                   qry[embryo_col].iloc[i]),
            "query_hpf": qry[time_col].iloc[i],
            "bin_center": bin_center,
            "argmax_label": label_order[argmax_idx[i]],
            "argmax_margin": float(margin[i]),
        }
        for j, lbl in enumerate(label_order):
            row[f"pred_proba_{lbl}"] = float(q[i, j])
        rows.append(row)
    return pd.DataFrame(rows)


def _predict_global(
    ref_X, ref_labels, qry_X, qry, label_order,
    snip_col, embryo_col, time_col, snip_to_embryo, random_state,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Mode A: one global model on all reference images."""
    clf = _make_pipeline(len(label_order), random_state)
    clf.fit(ref_X, ref_labels)
    q = _predict_proba_full(clf, qry_X, label_order)
    bin_center = float(qry[time_col].mean())
    return _image_df_from_proba(
        q, qry, label_order, snip_col, embryo_col, time_col, snip_to_embryo, bin_center
    )


def _get_hpf_bins(min_hpf: float, max_hpf: float, bin_width: float) -> list[float]:
    bins = []
    b = min_hpf
    while b < max_hpf:
        bins.append(b)
        b += bin_width
    return bins


def _predict_perbin(
    ref, qry, ref_X, ref_labels, qry_X, label_order,
    snip_col, embryo_col, time_col, _unused,
    snip_to_embryo, bin_width, min_hpf, max_hpf,
    random_state, use_embryo_mean: bool,
    feature_cols: list[str],
    label_col: str = "cluster_categories",
) -> pd.DataFrame:
    """Modes B and C: one model per HPF bin.

    use_embryo_mean=True  → mode C (embryo-averaged features per bin, standard)
    use_embryo_mean=False → mode B (per-image rows per bin)

    Falls back to a global model for bins with insufficient data.
    """
    bins = _get_hpf_bins(min_hpf, max_hpf, bin_width)

    global_clf = _make_pipeline(len(label_order), random_state)
    global_clf.fit(ref_X, ref_labels)

    all_img_dfs: list[pd.DataFrame] = []

    for bin_start in bins:
        bin_end = bin_start + bin_width
        bin_center = bin_start + bin_width / 2.0

        ref_bin = ref[(ref[time_col] >= bin_start) & (ref[time_col] < bin_end)].copy()
        qry_bin = qry[(qry[time_col] >= bin_start) & (qry[time_col] < bin_end)].copy()
        if len(qry_bin) == 0:
            continue

        # Build training X and y for this bin
        if len(ref_bin) < MIN_BIN_SAMPLES:
            clf_bin = global_clf
        else:
            if use_embryo_mean:
                # Mode C: mean features per embryo, then fit
                train_df = ref_bin.groupby(embryo_col, as_index=False)[feature_cols].mean()
                lbl_map = (
                    ref_bin.groupby(embryo_col)[label_col]
                    .agg(lambda x: x.mode().iloc[0])
                )
                train_X = train_df[feature_cols].to_numpy(dtype=float)
                train_y = np.array([lbl_map[eid] for eid in train_df[embryo_col]], dtype=str)
            else:
                # Mode B: one row per image
                train_X = ref_bin[feature_cols].to_numpy(dtype=float)
                train_y = ref_bin[label_col].to_numpy(dtype=str)

            n_classes_bin = len(np.unique(train_y))
            if n_classes_bin < MIN_BIN_CLASSES:
                clf_bin = global_clf
            else:
                clf_bin = _make_pipeline(len(label_order), random_state)
                try:
                    clf_bin.fit(train_X, train_y)
                except Exception:
                    clf_bin = global_clf

        q_bin = _predict_proba_full(
            clf_bin, qry_bin[feature_cols].to_numpy(dtype=float), label_order
        )
        img_df = _image_df_from_proba(
            q_bin, qry_bin, label_order, snip_col, embryo_col, time_col,
            snip_to_embryo, bin_center,
        )
        all_img_dfs.append(img_df)

    if not all_img_dfs:
        return pd.DataFrame()
    return pd.concat(all_img_dfs, ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Internal: rollup helpers
# ---------------------------------------------------------------------------

def _rollup_embryo(
    img_df: pd.DataFrame,
    embryo_col: str,
    proba_cols: list[str],
    label_order: list[str],
    rollup: str,
) -> pd.DataFrame:
    """Aggregate per-image probability vectors to embryo level.

    rollup='mean': straight mean of q vectors.
    rollup='margin_weighted': weight by max(0, argmax_margin) before averaging.
    """
    if img_df.empty:
        return pd.DataFrame(columns=["query_embryo_id", "predicted_label",
                                     "embryo_margin", "top_probability"])

    rows = []
    for embryo_id, grp in img_df.groupby("query_embryo_id"):
        q = grp[proba_cols].to_numpy(dtype=float)

        if rollup == "mean":
            q_emb = q.mean(axis=0)
        else:  # margin_weighted
            weights = np.clip(grp["argmax_margin"].to_numpy(dtype=float), 0, None)
            if weights.sum() == 0:
                q_emb = q.mean(axis=0)  # fallback to mean if all margins zero
            else:
                q_emb = (weights[:, None] * q).sum(axis=0) / weights.sum()

        sorted_q = np.sort(q_emb)[::-1]
        pred_idx = int(q_emb.argmax())
        pred_label = label_order[pred_idx] if pred_idx < len(label_order) else "unknown"
        margin = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else 0.0

        row = {
            "query_embryo_id": embryo_id,
            "predicted_label": pred_label,
            "embryo_margin": margin,
            "top_probability": float(sorted_q[0]),
            "n_images": int(len(grp)),
        }
        for j, lbl in enumerate(label_order):
            row[f"prob_{lbl}"] = float(q_emb[j])
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal: confidence helpers
# ---------------------------------------------------------------------------

def _add_distance_confidence(
    summary: pd.DataFrame,
    ref_X: np.ndarray,
    qry_X: np.ndarray,
    qry: pd.DataFrame,
    embryo_col: str,
    snip_col: str,
    k: int = 15,
) -> pd.DataFrame:
    """Add embryo_distance_score: percentile rank of query KNN distance vs reference."""
    from scipy import stats as scipy_stats
    from sklearn.neighbors import NearestNeighbors

    df = summary.copy()

    # Reference self-KNN (leave-one-out: k+1, drop self)
    k_ref = min(k + 1, len(ref_X))
    ref_nn = NearestNeighbors(n_neighbors=k_ref, metric="euclidean")
    ref_nn.fit(ref_X)
    ref_dists, _ = ref_nn.kneighbors(ref_X)
    ref_mean = ref_dists[:, 1:].mean(axis=1)  # drop self (col 0)

    # Query mean KNN distance
    k_qry = min(k, len(ref_X))
    qry_nn = NearestNeighbors(n_neighbors=k_qry, metric="euclidean")
    qry_nn.fit(ref_X)
    qry_dists, _ = qry_nn.kneighbors(qry_X)
    qry_mean = qry_dists.mean(axis=1)

    def to_score(d):
        return 1.0 - scipy_stats.percentileofscore(ref_mean, d, kind="rank") / 100.0

    dist_scores = np.array([to_score(d) for d in qry_mean])

    # Map image-level scores to embryo level
    snip_ids = qry[snip_col].to_numpy()
    embryo_ids = qry[embryo_col].to_numpy()
    dist_df = pd.DataFrame({
        "query_embryo_id": embryo_ids,
        "dist_score": dist_scores,
    })
    embryo_dist = (
        dist_df.groupby("query_embryo_id")["dist_score"]
        .mean()
        .reset_index()
        .rename(columns={"dist_score": "embryo_distance_score"})
    )
    df = df.merge(embryo_dist, on="query_embryo_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Internal: utilities
# ---------------------------------------------------------------------------

def _empty_results() -> dict:
    return {
        "embryo_label_transfer_summary": pd.DataFrame(),
        "image_predictions": pd.DataFrame(),
        "image_prediction_summary": pd.DataFrame(),
        "embryo_label_probabilities": pd.DataFrame(),
        "reference_label_profile": pd.DataFrame(),
        "neighbor_long_table": pd.DataFrame(),
    }
