"""
CEP290 phenotype label transfer via distance-weighted KNN.

Pass 1: MVP label transfer (neighbor table → image probs → embryo predictions)
Pass 2: Confidence scoring, diagnostic flags, status assignment
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors


VALID_STATUSES = ("assigned", "ambiguous", "low_density", "not_evaluated")
MAIN_LABELS = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_label_transfer(
    reference_df: pd.DataFrame,
    query_df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "cluster_categories",
    embryo_col: str = "embryo_id",
    snip_col: str = "snip_id",
    time_col: str = "predicted_stage_hpf",
    experiment_col: str = None,
    min_hpf: float = 30.0,
    max_hpf: float = 48.0,
    k: int = 15,
    metric: str = "euclidean",
    epsilon: float = 1e-8,
    agreement_threshold: float = 0.5,
    consistency_threshold: float = 0.6,
    distance_score_threshold: float = 0.05,
    top_probability_threshold: float = 0.4,
) -> dict:
    """
    Transfer phenotype labels from a labeled reference DataFrame to a query DataFrame.

    Parameters
    ----------
    reference_df : DataFrame with labeled rows. Must contain feature_cols, label_col,
        embryo_col, snip_col, time_col.
    query_df : DataFrame with rows to classify. Must contain feature_cols, embryo_col,
        snip_col, time_col. Labels are never read from query_df.
    feature_cols : List of numeric columns used to compute distances. No default —
        always specified by caller.
    label_col : Column in reference_df with phenotype labels to transfer.
    embryo_col, snip_col, time_col : Column names identifying embryos, images, and time.
    experiment_col : Optional column identifying experimental batch.
    min_hpf, max_hpf : Developmental time window applied to both DataFrames.
    k : Number of nearest reference neighbors per query image.
    metric : Distance metric for KNN ('euclidean', 'cosine', etc.).
    epsilon : Small value added to distances to prevent division by zero.
    agreement_threshold : Images below this neighbor agreement are flagged ambiguous.
    consistency_threshold : Embryos below this image consistency fraction are flagged.
    distance_score_threshold : Embryos below this in-distribution score are low_density.
    top_probability_threshold : Embryos below this top label probability are ambiguous.

    Returns
    -------
    dict with keys:
        neighbor_long_table, image_label_probabilities, image_prediction_summary,
        embryo_label_probabilities, embryo_label_transfer_summary,
        embryo_pred_label_dict, embryo_confidence_dict, embryo_status_dict
    """
    _validate_feature_cols(reference_df, query_df, feature_cols)

    # Track embryos excluded before KNN runs
    excluded_embryos = []

    # --- Step 1: filter reference ---
    ref = reference_df.copy()
    ref = ref[ref[label_col].notna()].copy()
    if min_hpf is not None:
        ref = ref[ref[time_col] >= min_hpf]
    if max_hpf is not None:
        ref = ref[ref[time_col] <= max_hpf]
    ref = ref.reset_index(drop=True)

    # --- Step 2: filter query, track excluded ---
    qry = query_df.copy()

    # Drop rows with missing features
    missing_feat_mask = qry[feature_cols].isna().any(axis=1)
    if missing_feat_mask.any():
        missing_embs = qry.loc[missing_feat_mask, embryo_col].unique()
        warnings.warn(
            f"{missing_feat_mask.sum()} query rows dropped for missing feature values "
            f"({len(missing_embs)} embryos affected)."
        )
        # Record embryos that have ALL rows missing (no remaining rows)
        remaining_after_feature_drop = qry.loc[~missing_feat_mask, embryo_col].unique()
        fully_missing = set(missing_embs) - set(remaining_after_feature_drop)
        for eid in fully_missing:
            excluded_embryos.append({
                embryo_col: eid,
                "status": "not_evaluated",
                "status_reason": "missing_features",
            })
        qry = qry[~missing_feat_mask].copy()

    # Time-window filter
    all_query_embryos = qry[embryo_col].unique()
    if min_hpf is not None:
        qry = qry[qry[time_col] >= min_hpf]
    if max_hpf is not None:
        qry = qry[qry[time_col] <= max_hpf]

    in_window_embryos = set(qry[embryo_col].unique())
    for eid in all_query_embryos:
        if eid not in in_window_embryos:
            excluded_embryos.append({
                embryo_col: eid,
                "status": "not_evaluated",
                "status_reason": "outside_time_window",
            })

    qry = qry.reset_index(drop=True)

    if len(ref) == 0:
        raise ValueError("No reference rows remain after filtering.")
    if len(qry) == 0:
        warnings.warn("No query rows remain after filtering. Returning empty results.")
        return _empty_results(excluded_embryos, embryo_col)

    # --- Step 3: KNN search ---
    ref_X = ref[feature_cols].values.astype(float)
    qry_X = qry[feature_cols].values.astype(float)

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(ref_X)
    distances, indices = nn.kneighbors(qry_X)  # (n_query, k)

    # --- Step 4: Build neighbor long table ---
    neighbor_records = []
    for qi in range(len(qry)):
        qrow = qry.iloc[qi]
        for rank, (dist, ref_idx) in enumerate(zip(distances[qi], indices[qi])):
            rrow = ref.iloc[ref_idx]
            rec = {
                "query_embryo_id": qrow[embryo_col],
                "query_snip_id": qrow[snip_col],
                "query_hpf": qrow[time_col],
                "neighbor_rank": rank,
                "ref_snip_id": rrow[snip_col],
                "ref_embryo_id": rrow[embryo_col],
                "ref_label": rrow[label_col],
                "ref_hpf": rrow[time_col],
                "distance": dist,
                "weight": 1.0 / (dist + epsilon),
            }
            if experiment_col is not None:
                rec["query_experiment_id"] = qrow.get(experiment_col, np.nan)
                rec["ref_experiment_id"] = rrow.get(experiment_col, np.nan)
            neighbor_records.append(rec)

    neighbor_df = pd.DataFrame(neighbor_records)

    # --- Step 5 & 6: Image-level label probabilities and prediction summary ---
    image_prob_df, image_summary_df = _compute_image_probabilities(neighbor_df, epsilon)

    # --- Step 7: Embryo-level aggregation ---
    embryo_prob_df, embryo_summary_df = _compute_embryo_probabilities(
        image_prob_df, image_summary_df, qry, embryo_col, snip_col, time_col
    )

    # --- Pass 2: Confidence scoring ---
    embryo_summary_df = _add_confidence_scores(
        embryo_summary_df, neighbor_df, ref_X, k, metric, epsilon
    )

    # --- Status assignment ---
    embryo_summary_df = _assign_status(
        embryo_summary_df,
        agreement_threshold=agreement_threshold,
        consistency_threshold=consistency_threshold,
        distance_score_threshold=distance_score_threshold,
        top_probability_threshold=top_probability_threshold,
    )

    # Append excluded embryos
    if excluded_embryos:
        excl_df = pd.DataFrame(excluded_embryos).rename(columns={embryo_col: "query_embryo_id"})
        excl_df["predicted_label"] = "not_assigned"
        embryo_summary_df = pd.concat([embryo_summary_df, excl_df], ignore_index=True, sort=False)

    # Simple dictionaries for easy downstream use
    embryo_pred_label_dict = dict(
        zip(embryo_summary_df["query_embryo_id"], embryo_summary_df["predicted_label"])
    )
    embryo_confidence_dict = dict(
        zip(embryo_summary_df["query_embryo_id"], embryo_summary_df.get("embryo_confidence", np.nan))
    )
    embryo_status_dict = dict(
        zip(embryo_summary_df["query_embryo_id"], embryo_summary_df.get("status", np.nan))
    )

    return {
        "neighbor_long_table": neighbor_df,
        "image_label_probabilities": image_prob_df,
        "image_prediction_summary": image_summary_df,
        "embryo_label_probabilities": embryo_prob_df,
        "embryo_label_transfer_summary": embryo_summary_df,
        "embryo_pred_label_dict": embryo_pred_label_dict,
        "embryo_confidence_dict": embryo_confidence_dict,
        "embryo_status_dict": embryo_status_dict,
    }


def add_label_transfer_predictions(
    query_df: pd.DataFrame,
    embryo_summary_df: pd.DataFrame,
    embryo_col: str = "embryo_id",
    label_col_out: str = "predicted_label",
    confidence_col_out: str = "label_transfer_confidence",
    status_col_out: str = "label_transfer_status",
) -> pd.DataFrame:
    """Merge embryo-level predictions back onto query_df rows."""
    cols = ["query_embryo_id", "predicted_label"]
    if "embryo_confidence" in embryo_summary_df.columns:
        cols.append("embryo_confidence")
    if "status" in embryo_summary_df.columns:
        cols.append("status")

    merge_df = embryo_summary_df[cols].rename(columns={
        "query_embryo_id": embryo_col,
        "predicted_label": label_col_out,
        "embryo_confidence": confidence_col_out,
        "status": status_col_out,
    })
    return query_df.merge(merge_df, on=embryo_col, how="left")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_feature_cols(reference_df, query_df, feature_cols):
    missing_ref = [c for c in feature_cols if c not in reference_df.columns]
    missing_qry = [c for c in feature_cols if c not in query_df.columns]
    if missing_ref:
        raise ValueError(f"feature_cols missing from reference_df: {missing_ref}")
    if missing_qry:
        raise ValueError(f"feature_cols missing from query_df: {missing_qry}")


def _compute_image_probabilities(neighbor_df: pd.DataFrame, epsilon: float):
    """
    Compute distance-weighted label probabilities for each query image.

    Groups by query_snip_id (globally unique). Returns:
      image_prob_df  : long format, one row per (snip × label)
      image_summary_df : one row per snip
    """
    records = []
    summary_records = []

    # Determine label universe from neighbor data
    all_labels = sorted(neighbor_df["ref_label"].dropna().unique())

    for snip_id, grp in neighbor_df.groupby("query_snip_id"):
        meta = grp.iloc[0]
        total_weight = grp["weight"].sum()

        label_probs = {}
        for label in all_labels:
            label_weight = grp.loc[grp["ref_label"] == label, "weight"].sum()
            label_probs[label] = label_weight / total_weight if total_weight > 0 else 0.0

        for label, prob in label_probs.items():
            records.append({
                "query_embryo_id": meta["query_embryo_id"],
                "query_snip_id": snip_id,
                "query_hpf": meta["query_hpf"],
                "label": label,
                "image_label_probability": prob,
            })

        image_pred_label = max(label_probs, key=label_probs.get)
        image_neighbor_agreement = max(label_probs.values())

        summary_records.append({
            "query_embryo_id": meta["query_embryo_id"],
            "query_snip_id": snip_id,
            "query_hpf": meta["query_hpf"],
            "image_pred_label": image_pred_label,
            "image_neighbor_agreement": image_neighbor_agreement,
        })

    image_prob_df = pd.DataFrame(records)
    image_summary_df = pd.DataFrame(summary_records)
    return image_prob_df, image_summary_df


def _compute_embryo_probabilities(
    image_prob_df: pd.DataFrame,
    image_summary_df: pd.DataFrame,
    qry: pd.DataFrame,
    embryo_col: str,
    snip_col: str,
    time_col: str,
):
    """
    Average image-level label probabilities across images per embryo.
    Returns embryo_prob_df (long) and embryo_summary_df (wide, one row per embryo).
    """
    # Map snip → embryo
    snip_to_embryo = dict(zip(qry[snip_col], qry[embryo_col]))
    image_prob_df = image_prob_df.copy()
    image_prob_df["query_embryo_id"] = image_prob_df["query_snip_id"].map(snip_to_embryo)

    # Mean probability per embryo per label
    embryo_prob_df = (
        image_prob_df
        .groupby(["query_embryo_id", "label"], as_index=False)["image_label_probability"]
        .mean()
        .rename(columns={"image_label_probability": "embryo_label_probability"})
    )

    # Wide summary per embryo
    embryo_wide = embryo_prob_df.pivot(
        index="query_embryo_id", columns="label", values="embryo_label_probability"
    ).reset_index()

    all_labels = sorted(image_prob_df["label"].unique())
    embryo_wide["predicted_label"] = embryo_wide[all_labels].idxmax(axis=1)
    embryo_wide["top_label_probability"] = embryo_wide[all_labels].max(axis=1)

    # Image count metadata per embryo
    snip_hpf = image_summary_df[["query_embryo_id", "query_snip_id", "query_hpf"]]
    img_meta = (
        snip_hpf
        .groupby("query_embryo_id")
        .agg(
            n_images=("query_snip_id", "nunique"),
            min_query_hpf=("query_hpf", "min"),
            max_query_hpf=("query_hpf", "max"),
        )
        .reset_index()
    )
    img_meta["query_hpf_range"] = img_meta["max_query_hpf"] - img_meta["min_query_hpf"]

    embryo_summary_df = embryo_wide.merge(img_meta, on="query_embryo_id", how="left")

    # Keep only the metadata + prediction columns (drop per-label probability cols from summary)
    keep_cols = [
        "query_embryo_id", "n_images", "min_query_hpf", "max_query_hpf", "query_hpf_range",
        "predicted_label", "top_label_probability",
    ]
    embryo_summary_df = embryo_summary_df[keep_cols]

    return embryo_prob_df, embryo_summary_df


def _add_confidence_scores(
    embryo_summary_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    ref_X: np.ndarray,
    k: int,
    metric: str,
    epsilon: float,
) -> pd.DataFrame:
    """
    Add confidence components to embryo_summary_df:
      - mean_image_neighbor_agreement
      - embryo_distance_score
      - embryo_consistency_score
      - embryo_confidence
    """
    df = embryo_summary_df.copy()

    # --- Neighbor agreement: mean per embryo ---
    image_agreement = (
        neighbor_df
        .groupby("query_snip_id")
        .apply(_image_neighbor_agreement)
        .reset_index()
        .rename(columns={0: "image_neighbor_agreement"})
    )
    image_agreement["query_embryo_id"] = neighbor_df.groupby("query_snip_id")["query_embryo_id"].first().values

    embryo_agreement = (
        image_agreement
        .groupby("query_embryo_id")["image_neighbor_agreement"]
        .mean()
        .reset_index()
        .rename(columns={"image_neighbor_agreement": "mean_image_neighbor_agreement"})
    )
    df = df.merge(embryo_agreement, on="query_embryo_id", how="left")

    # --- Distance confidence ---
    # Reference self-KNN: k+1 neighbors, drop self (rank 0)
    ref_nn = NearestNeighbors(n_neighbors=min(k + 1, len(ref_X)), metric=metric)
    ref_nn.fit(ref_X)
    ref_dists, _ = ref_nn.kneighbors(ref_X)
    # Drop first column (self-match, distance≈0)
    ref_knn_dists = ref_dists[:, 1:]
    ref_mean_knn_dist = ref_knn_dists.mean(axis=1)  # (n_ref,)

    # Query mean KNN distance (already from the k neighbors)
    query_mean_knn = (
        neighbor_df
        .groupby("query_snip_id")["distance"]
        .mean()
        .reset_index()
        .rename(columns={"distance": "mean_knn_distance"})
    )
    query_mean_knn["query_embryo_id"] = (
        neighbor_df.groupby("query_snip_id")["query_embryo_id"].first().values
    )

    # Percentile rank of each query image against reference distribution
    def dist_to_score(d):
        pct = stats.percentileofscore(ref_mean_knn_dist, d, kind="rank")
        return 1.0 - pct / 100.0

    query_mean_knn["distance_in_distribution_score"] = query_mean_knn["mean_knn_distance"].map(
        dist_to_score
    )

    embryo_dist_score = (
        query_mean_knn
        .groupby("query_embryo_id")["distance_in_distribution_score"]
        .mean()
        .reset_index()
        .rename(columns={"distance_in_distribution_score": "embryo_distance_score"})
    )
    df = df.merge(embryo_dist_score, on="query_embryo_id", how="left")

    # --- Embryo consistency: fraction of images agreeing with embryo_pred_label ---
    image_pred_labels = (
        neighbor_df
        .groupby("query_snip_id")
        .apply(_image_pred_label_from_group)
        .reset_index()
        .rename(columns={0: "image_pred_label"})
    )
    image_pred_labels["query_embryo_id"] = (
        neighbor_df.groupby("query_snip_id")["query_embryo_id"].first().values
    )
    image_pred_labels = image_pred_labels.merge(
        df[["query_embryo_id", "predicted_label"]], on="query_embryo_id", how="left"
    )
    image_pred_labels["agrees"] = (
        image_pred_labels["image_pred_label"] == image_pred_labels["predicted_label"]
    )
    embryo_consistency = (
        image_pred_labels
        .groupby("query_embryo_id")["agrees"]
        .mean()
        .reset_index()
        .rename(columns={"agrees": "embryo_consistency_score"})
    )
    df = df.merge(embryo_consistency, on="query_embryo_id", how="left")

    # Single-image embryos: consistency set to 1.0 by definition
    df.loc[df["n_images"] == 1, "embryo_consistency_score"] = 1.0

    # --- Aggregate confidence ---
    df["embryo_confidence"] = (
        df["mean_image_neighbor_agreement"]
        * df["embryo_distance_score"]
        * df["embryo_consistency_score"]
    )

    return df


def _image_neighbor_agreement(grp):
    """Max weighted-vote probability for the image represented by grp."""
    total = grp["weight"].sum()
    if total == 0:
        return 0.0
    label_probs = grp.groupby("ref_label")["weight"].sum() / total
    return float(label_probs.max())


def _image_pred_label_from_group(grp):
    """Argmax weighted-vote label for a single image group."""
    total = grp["weight"].sum()
    if total == 0:
        return np.nan
    label_probs = grp.groupby("ref_label")["weight"].sum() / total
    return label_probs.idxmax()


def _assign_status(
    embryo_summary_df: pd.DataFrame,
    agreement_threshold: float,
    consistency_threshold: float,
    distance_score_threshold: float,
    top_probability_threshold: float,
) -> pd.DataFrame:
    df = embryo_summary_df.copy()

    # Diagnostic boolean flags
    df["is_low_density"] = df["embryo_distance_score"] < distance_score_threshold
    df["is_low_agreement"] = df["mean_image_neighbor_agreement"] < agreement_threshold
    df["is_low_consistency"] = df["embryo_consistency_score"] < consistency_threshold
    df["is_low_top_probability"] = df["top_label_probability"] < top_probability_threshold

    # Compact status (priority order)
    conditions = [
        df["is_low_density"],
        df["is_low_agreement"] | df["is_low_consistency"] | df["is_low_top_probability"],
    ]
    choices = ["low_density", "ambiguous"]
    df["status"] = np.select(conditions, choices, default="assigned")
    df["status_reason"] = ""

    return df


def _empty_results(excluded_embryos, embryo_col):
    excl_df = pd.DataFrame(excluded_embryos).rename(columns={embryo_col: "query_embryo_id"})
    excl_df["predicted_label"] = "not_assigned"
    return {
        "neighbor_long_table": pd.DataFrame(),
        "image_label_probabilities": pd.DataFrame(),
        "image_prediction_summary": pd.DataFrame(),
        "embryo_label_probabilities": pd.DataFrame(),
        "embryo_label_transfer_summary": excl_df,
        "embryo_pred_label_dict": dict(zip(excl_df["query_embryo_id"], excl_df["predicted_label"])),
        "embryo_confidence_dict": {},
        "embryo_status_dict": dict(zip(excl_df["query_embryo_id"], excl_df.get("status", "not_evaluated"))),
    }
