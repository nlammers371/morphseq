#!/usr/bin/env python
"""Embryo-first persistence discovery for subtle phenotype methods.

This script uses standardized `input_core.csv` tables and the shared time-binning
and resampling utilities to:
1. Load + filter frame-level observations
2. Build embryo x 2hpf bins
3. Construct soft-mutual persistence matrices across time windows
4. Scan cluster resolution with bootstrap stability
5. Emit cohort assignments + drift summaries
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze.utils import resampling as resample  # noqa: E402
from src.analyze.utils.binning import add_time_bins, bin_embryos_by_time, filter_binned_data  # noqa: E402


DEFAULT_INPUT_ROOT = Path(__file__).resolve().parent / "input_data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "embryo_first_persistence"

CORE_FEATURES = [
    "total_length_um",
    "baseline_deviation_um",
    "mean_curvature_per_um",
    "std_curvature_per_um",
    "max_curvature_per_um",
    "surface_area_um",
    "area_um2",
]

BOOL_TRUE_TOKENS = {"1", "true", "t", "yes", "y"}
MISSING_TOKENS = {"", "na", "nan", "none", "null"}


@dataclass
class ScopeRunResult:
    scope_key: str
    output_dir: Path
    n_embryos: int
    n_windows: int
    best_k: int


def _normalize_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_")


def _is_wildtype_like(value: Any) -> bool:
    token = _normalize_token(value)
    if token in {"ab", "a/b", "wt", "wildtype", "wild_type", "b92_wildtype", "cep290_wildtype"}:
        return True
    if "wild" in token and "type" in token:
        return True
    if re.search(r"(^|[_\s/])wt($|[_\s/])", token):
        return True
    return False


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.map(lambda x: _normalize_token(x) in BOOL_TRUE_TOKENS)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _cluster_distance_matrix(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    kwargs = {"n_clusters": int(n_clusters), "linkage": "average"}
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return model.fit_predict(distance_matrix)


def _resolve_dataset_ids(input_root: Path, dataset_ids: Sequence[str] | None) -> List[str]:
    manifest_path = input_root / "datasets_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    available = sorted(manifest["dataset_id"].dropna().astype(str).unique().tolist())
    if not available:
        raise ValueError(f"No dataset_id entries found in {manifest_path}")

    if dataset_ids:
        requested = [x.strip() for x in dataset_ids if x.strip()]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Unknown dataset_ids: {missing}. Available: {available}")
        return requested

    return available


def _load_input_tables(input_root: Path, dataset_ids: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for dataset_id in dataset_ids:
        csv_path = input_root / "experiments" / dataset_id / "input_core.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input_core.csv for {dataset_id}: {csv_path}")

        df = pd.read_csv(csv_path, low_memory=False)
        df["dataset_id"] = dataset_id
        if "experiment_id" not in df.columns:
            df["experiment_id"] = df["embryo_id"].astype(str).str.split("_", n=1).str[0]
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["genotype_group"] = out.get("genotype", pd.Series(["unknown"] * len(out))).astype(str)
    out["phenotype_group"] = out.get("phenotype", pd.Series(["unknown"] * len(out))).astype(str)

    wt_mask = out["genotype_group"].map(_is_wildtype_like) | out["phenotype_group"].map(_is_wildtype_like)
    out.loc[wt_mask, "genotype_group"] = "wildtype"

    unlabeled_pheno = out["phenotype_group"].map(_normalize_token).isin(MISSING_TOKENS | {"unlabeled", "unknown"})
    out.loc[wt_mask & unlabeled_pheno, "phenotype_group"] = "wildtype"

    return out


def _apply_filters(
    df: pd.DataFrame,
    *,
    drop_dead: bool,
    require_focus_flag: bool,
    require_well_qc: bool,
    require_sam2_qc: bool,
) -> pd.DataFrame:
    out = df.copy()

    if "use_embryo_flag" in out.columns:
        out = out.loc[_to_bool_series(out["use_embryo_flag"])].copy()

    if drop_dead and "dead_flag2" in out.columns:
        out = out.loc[~_to_bool_series(out["dead_flag2"])].copy()

    if require_focus_flag and "focus_flag" in out.columns:
        out = out.loc[~_to_bool_series(out["focus_flag"])].copy()

    if require_well_qc and "well_qc_flag" in out.columns:
        out = out.loc[~_to_bool_series(out["well_qc_flag"])].copy()

    if require_sam2_qc and "sam2_qc_flag" in out.columns:
        out = out.loc[~_to_bool_series(out["sam2_qc_flag"])].copy()

    out = out.dropna(subset=["embryo_id", "predicted_stage_hpf"])
    return out.reset_index(drop=True)


def _resolve_feature_mode(df: pd.DataFrame, feature_mode: str) -> tuple[str, List[str], Dict[str, Any]]:
    requested = feature_mode.strip().lower()
    zmu_cols = [c for c in df.columns if "z_mu_b" in c]
    core_cols = [c for c in CORE_FEATURES if c in df.columns]

    meta: Dict[str, Any] = {
        "requested_feature_mode": requested,
        "zmu_cols_available": len(zmu_cols),
        "core_cols_available": len(core_cols),
    }

    if requested in {"auto", "zmu_b", "zmu"}:
        if zmu_cols:
            meta["resolved_feature_mode"] = "zmu_b"
            return "zmu_b", zmu_cols, meta
        if requested in {"zmu_b", "zmu"}:
            meta["feature_warning"] = "Requested zmu_b but none found in input_core.csv; falling back to core features"
        if not core_cols:
            raise ValueError("No z_mu_b columns and no core morphology columns found.")
        meta["resolved_feature_mode"] = "core"
        return "core", core_cols, meta

    if requested == "core":
        if not core_cols:
            raise ValueError("Requested core features but none were found.")
        meta["resolved_feature_mode"] = "core"
        return "core", core_cols, meta

    if requested == "pca95":
        base_mode = "zmu_b" if zmu_cols else "core"
        base_cols = zmu_cols if zmu_cols else core_cols
        if not base_cols:
            raise ValueError("Cannot run pca95: no numeric base feature columns found.")
        meta["resolved_feature_mode"] = "pca95"
        meta["pca_base_mode"] = base_mode
        return "pca95", base_cols, meta

    raise ValueError("feature_mode must be one of: auto, zmu_b, core, pca95")


def _add_pca95(df: pd.DataFrame, feature_cols: Sequence[str], random_state: int) -> tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    out = df.copy()
    X = out[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    medians = X.median(axis=0)
    X_filled = X.fillna(medians)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled.to_numpy(dtype=float))

    pca = PCA(n_components=0.95, svd_solver="full", random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    pca_cols = [f"pca95_{i + 1:02d}" for i in range(X_pca.shape[1])]
    out[pca_cols] = X_pca

    info = {
        "pca_components": int(X_pca.shape[1]),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return out, pca_cols, info


def _build_scopes(df: pd.DataFrame, scope_mode: str) -> List[tuple[str, pd.DataFrame]]:
    scope_mode = scope_mode.strip().lower()

    if scope_mode == "cross_dataset":
        return [("cross_dataset_all", df.copy())]

    if scope_mode == "within_dataset":
        scopes = []
        for dataset_id, gdf in df.groupby("dataset_id", sort=True):
            scopes.append((f"dataset_{dataset_id}", gdf.copy()))
        return scopes

    if scope_mode == "within_experiment":
        scopes = []
        grouped = df.groupby(["dataset_id", "experiment_id"], sort=True)
        for (dataset_id, experiment_id), gdf in grouped:
            scopes.append((f"dataset_{dataset_id}__experiment_{experiment_id}", gdf.copy()))
        return scopes

    raise ValueError("scope_mode must be one of: within_experiment, within_dataset, cross_dataset")


def _build_window_adjacency(window_df: pd.DataFrame, feature_cols: Sequence[str], k_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    embryo_ids = window_df["embryo_id"].astype(str).to_numpy()
    X = window_df[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(axis=0))

    values = X.to_numpy(dtype=float)
    if len(values) < 2:
        return embryo_ids, np.eye(len(values), dtype=float)

    scaled = StandardScaler().fit_transform(values)

    k_eff = max(1, min(int(k_neighbors), len(values) - 1))
    # `algorithm='kd_tree'` avoids OpenMP SHM failures seen with default backend
    # in restricted environments while still scaling well for this feature count.
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", algorithm="kd_tree")
    nn.fit(scaled)
    distances, indices = nn.kneighbors(scaled)

    neighbor_dist = distances[:, 1:]
    neighbor_idx = indices[:, 1:]

    sigma = neighbor_dist[:, -1]
    eps = 1e-8
    positive_sigma = sigma[sigma > eps]
    sigma_floor = float(np.median(positive_sigma)) if positive_sigma.size else 1.0
    sigma = np.where(sigma > eps, sigma, sigma_floor)

    directed = np.zeros((len(values), len(values)), dtype=float)
    weights = np.exp(-(neighbor_dist ** 2) / (sigma[:, None] ** 2 + eps))

    row_idx = np.repeat(np.arange(len(values)), k_eff)
    directed[row_idx, neighbor_idx.reshape(-1)] = weights.reshape(-1)

    sym = np.sqrt(directed * directed.T)
    np.fill_diagonal(sym, 1.0)
    return embryo_ids, sym


def _compute_persistence(
    df_binned: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    k_neighbors: int,
    topq: float,
) -> tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    embryo_index = sorted(df_binned["embryo_id"].astype(str).unique().tolist())
    emb_to_idx = {emb: i for i, emb in enumerate(embryo_index)}
    n_emb = len(embryo_index)

    window_records: List[Dict[str, Any]] = []
    window_mats: List[np.ndarray] = []

    for time_bin, gdf in df_binned.groupby("time_bin", sort=True):
        gdf = gdf.dropna(subset=list(feature_cols))
        if gdf["embryo_id"].nunique() < 2:
            continue

        local_embryos, local_adj = _build_window_adjacency(gdf, feature_cols, k_neighbors)

        mat = np.full((n_emb, n_emb), np.nan, dtype=np.float32)
        local_idx = np.array([emb_to_idx[e] for e in local_embryos], dtype=int)
        mat[np.ix_(local_idx, local_idx)] = local_adj.astype(np.float32)
        window_mats.append(mat)

        window_records.append(
            {
                "time_bin": float(time_bin),
                "n_embryos": int(len(local_embryos)),
                "embryo_ids": local_embryos.tolist(),
                "adjacency": local_adj,
            }
        )

    if not window_mats:
        raise ValueError("No valid time windows available after binning/filtering.")

    stack = np.stack(window_mats, axis=0).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p_mean = np.nanmean(stack, axis=0)
    p_mean = np.where(np.isnan(p_mean), 0.0, p_mean).astype(np.float32)
    np.fill_diagonal(p_mean, 1.0)

    neg_inf = np.float32(-np.inf)
    sorted_desc = np.sort(np.where(np.isnan(stack), neg_inf, stack), axis=0)[::-1]
    valid_counts = np.sum(sorted_desc > neg_inf, axis=0)
    q_counts = np.ceil(valid_counts * float(topq)).astype(int)
    q_counts = np.where(valid_counts > 0, np.maximum(1, q_counts), 0)

    safe_values = np.where(sorted_desc > neg_inf, sorted_desc, 0.0)
    cumsums = np.cumsum(safe_values, axis=0)

    gather_idx = np.clip(q_counts - 1, 0, max(sorted_desc.shape[0] - 1, 0))
    top_sums = np.take_along_axis(cumsums, gather_idx[None, :, :], axis=0)[0]

    p_topq = np.zeros_like(p_mean, dtype=np.float32)
    valid_mask = q_counts > 0
    p_topq[valid_mask] = (top_sums[valid_mask] / q_counts[valid_mask]).astype(np.float32)
    np.fill_diagonal(p_topq, 1.0)

    return p_mean, p_topq, embryo_index, window_records


def _compute_drift(window_records: List[Dict[str, Any]], k_neighbors: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []

    for a, b in zip(window_records[:-1], window_records[1:]):
        emb_a = list(a["embryo_ids"])
        emb_b = list(b["embryo_ids"])
        overlap = sorted(set(emb_a) & set(emb_b))
        if not overlap:
            continue

        idx_map_a = {e: i for i, e in enumerate(emb_a)}
        idx_map_b = {e: i for i, e in enumerate(emb_b)}

        adj_a = a["adjacency"]
        adj_b = b["adjacency"]

        for emb in overlap:
            i = idx_map_a[emb]
            j = idx_map_b[emb]

            neigh_a_rank = np.argsort(adj_a[i])[::-1]
            neigh_b_rank = np.argsort(adj_b[j])[::-1]

            set_a = [emb_a[idx] for idx in neigh_a_rank if emb_a[idx] != emb and adj_a[i, idx] > 0]
            set_b = [emb_b[idx] for idx in neigh_b_rank if emb_b[idx] != emb and adj_b[j, idx] > 0]

            set_a = set(set_a[:k_neighbors])
            set_b = set(set_b[:k_neighbors])

            union = set_a | set_b
            inter = set_a & set_b
            jaccard = 1.0 if not union else len(inter) / len(union)

            rows.append(
                {
                    "embryo_id": emb,
                    "time_bin": float(a["time_bin"]),
                    "time_bin_next": float(b["time_bin"]),
                    "jaccard_neighbors": float(jaccard),
                    "drift": float(1.0 - jaccard),
                    "gain": int(len(set_b - set_a)),
                    "loss": int(len(set_a - set_b)),
                }
            )

    drift_df = pd.DataFrame(rows)
    if drift_df.empty:
        return drift_df, drift_df

    summary = (
        drift_df.groupby("embryo_id", as_index=False)
        .agg(
            mean_drift=("drift", "mean"),
            max_drift=("drift", "max"),
            mean_gain=("gain", "mean"),
            mean_loss=("loss", "mean"),
            n_transitions=("drift", "size"),
        )
        .sort_values(["mean_drift", "max_drift"], ascending=False)
        .reset_index(drop=True)
    )

    def _taxonomy(row: pd.Series) -> str:
        if row["mean_drift"] < 0.25:
            return "stable"
        if row["mean_drift"] > 0.60:
            if row["mean_gain"] > row["mean_loss"]:
                return "joiner"
            if row["mean_loss"] > row["mean_gain"]:
                return "expelled"
            return "transitioner"
        return "transitioner"

    summary["taxonomy"] = summary.apply(_taxonomy, axis=1)
    return drift_df, summary


def _run_resolution_scan(
    p_mean: np.ndarray,
    *,
    k_values: Sequence[int],
    n_bootstrap: int,
    bootstrap_frac: float,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, int, np.ndarray]:
    D = 1.0 - np.clip(p_mean, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)

    rows: List[Dict[str, Any]] = []
    labels_by_k: Dict[int, np.ndarray] = {}

    for k in sorted(set(int(x) for x in k_values if int(x) > 1)):
        if k >= D.shape[0]:
            continue

        full_labels = _cluster_distance_matrix(D, n_clusters=k)
        labels_by_k[k] = full_labels

        sil = np.nan
        if len(np.unique(full_labels)) > 1:
            try:
                sil = float(silhouette_score(D, full_labels, metric="precomputed"))
            except Exception:
                sil = np.nan

        spec = resample.subsample(frac=float(bootstrap_frac))

        def _stat_fn(data: dict, _rng) -> float:
            if "indices" in data:
                idx = np.asarray(data["indices"], dtype=int)
                idx = np.unique(idx)
            else:
                idx = np.arange(len(data["reference_labels"]), dtype=int)
            if len(idx) <= k:
                raise ValueError("Bootstrap sample too small for selected k")

            D_sub = data["distance_matrix"][np.ix_(idx, idx)]
            ref_sub = data["reference_labels"][idx]
            boot_labels = _cluster_distance_matrix(D_sub, n_clusters=k)
            return float(adjusted_rand_score(ref_sub, boot_labels))

        run = resample.run(
            data={"n": int(D.shape[0]), "distance_matrix": D, "reference_labels": full_labels},
            spec=spec,
            statistic=resample.statistic(name="ari_vs_full", fn=_stat_fn),
            n_iters=int(n_bootstrap),
            seed=int(seed + k),
            n_jobs=int(n_jobs),
            store="all",
            max_retries_per_iter=2,
        )
        summary = resample.aggregate(run, alpha=0.05)

        rows.append(
            {
                "k": int(k),
                "n_clusters_observed": int(len(np.unique(full_labels))),
                "silhouette": float(sil) if np.isfinite(sil) else np.nan,
                "bootstrap_mean_ari": float(summary.mean),
                "bootstrap_ci_low": float(summary.ci_low),
                "bootstrap_ci_high": float(summary.ci_high),
                "bootstrap_se": float(summary.se),
                "bootstrap_n_success": int(run.n_success),
                "bootstrap_n_failed": int(run.n_failed),
            }
        )

    scan_df = pd.DataFrame(rows)
    if scan_df.empty:
        raise ValueError("Resolution scan failed: no valid k values produced outputs.")

    scan_df = scan_df.sort_values(
        ["bootstrap_mean_ari", "silhouette", "k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_k = int(scan_df.loc[0, "k"])
    return scan_df, best_k, labels_by_k[best_k]


def _prepare_run_dir(base_dir: Path, scope_key: str, feature_mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{_safe_slug(scope_key)}_{_safe_slug(feature_mode)}"
    out_dir = base_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_scope(
    scope_key: str,
    df_scope: pd.DataFrame,
    *,
    feature_mode: str,
    bin_width: float,
    k_neighbors: int,
    topq: float,
    k_values: Sequence[int],
    n_bootstrap: int,
    bootstrap_frac: float,
    min_time_bins: int,
    min_embryos: int,
    seed: int,
    n_jobs: int,
    output_root: Path,
) -> ScopeRunResult:
    start = time.time()

    mode_used, base_features, mode_meta = _resolve_feature_mode(df_scope, feature_mode)
    work = df_scope.copy()

    pca_meta: Dict[str, Any] = {}
    if mode_used == "pca95":
        work, analysis_features, pca_meta = _add_pca95(work, base_features, random_state=seed)
    else:
        analysis_features = list(base_features)

    work = work.dropna(subset=["predicted_stage_hpf", "embryo_id"])
    work = work.loc[work[analysis_features].notna().any(axis=1)].copy()

    work = add_time_bins(work, time_col="predicted_stage_hpf", bin_width=bin_width, bin_col="time_bin")
    binned = bin_embryos_by_time(
        work,
        time_col="predicted_stage_hpf",
        z_cols=analysis_features,
        bin_width=bin_width,
        suffix="_binned",
    )
    binned = filter_binned_data(binned, min_time_bins=min_time_bins, min_embryos=min_embryos)

    binned_feature_cols = [f"{c}_binned" for c in analysis_features]
    binned = binned.dropna(subset=binned_feature_cols, how="any").reset_index(drop=True)

    p_mean, p_topq, embryo_ids, window_records = _compute_persistence(
        binned,
        binned_feature_cols,
        k_neighbors=k_neighbors,
        topq=topq,
    )

    scan_df, best_k, best_labels = _run_resolution_scan(
        p_mean,
        k_values=k_values,
        n_bootstrap=n_bootstrap,
        bootstrap_frac=bootstrap_frac,
        seed=seed,
        n_jobs=n_jobs,
    )

    D_topq = 1.0 - np.clip(p_topq, 0.0, 1.0)
    np.fill_diagonal(D_topq, 0.0)
    topq_labels = _cluster_distance_matrix(D_topq, n_clusters=best_k)

    assignments = pd.DataFrame(
        {
            "embryo_id": embryo_ids,
            "cluster": [f"C{int(x) + 1:02d}" for x in best_labels],
            "cluster_topq": [f"C{int(x) + 1:02d}" for x in topq_labels],
        }
    )

    meta_cols = [
        c
        for c in ["dataset_id", "experiment_id", "genotype", "genotype_group", "phenotype", "phenotype_group", "video_id", "well_id"]
        if c in binned.columns
    ]
    embryo_meta = binned[["embryo_id", *meta_cols]].drop_duplicates(subset=["embryo_id"]) if meta_cols else binned[["embryo_id"]].drop_duplicates()
    assignments = assignments.merge(embryo_meta, on="embryo_id", how="left")

    binned_out = binned.merge(assignments[["embryo_id", "cluster", "cluster_topq"]], on="embryo_id", how="left")

    drift_df, drift_summary_df = _compute_drift(window_records, k_neighbors=k_neighbors)

    out_dir = _prepare_run_dir(output_root, scope_key=scope_key, feature_mode=feature_mode)

    np.save(out_dir / "persistence_mean.npy", p_mean)
    np.save(out_dir / "persistence_topq.npy", p_topq)

    assignments.to_csv(out_dir / "cohort_assignments.tsv", sep="\t", index=False)
    binned_out.to_csv(out_dir / "binned_data.tsv", sep="\t", index=False)

    scan_df.to_csv(out_dir / "resolution_scan.tsv", sep="\t", index=False)

    if not drift_df.empty:
        drift_df.to_csv(out_dir / "drift_per_window.tsv", sep="\t", index=False)
    else:
        pd.DataFrame(columns=["embryo_id", "time_bin", "time_bin_next", "jaccard_neighbors", "drift", "gain", "loss"]).to_csv(
            out_dir / "drift_per_window.tsv", sep="\t", index=False
        )

    if not drift_summary_df.empty:
        drift_summary_df.to_csv(out_dir / "drift_summary.tsv", sep="\t", index=False)
    else:
        pd.DataFrame(columns=["embryo_id", "mean_drift", "max_drift", "mean_gain", "mean_loss", "n_transitions", "taxonomy"]).to_csv(
            out_dir / "drift_summary.tsv", sep="\t", index=False
        )

    feature_benchmark = pd.DataFrame(
        [
            {
                "scope_key": scope_key,
                "requested_feature_mode": feature_mode,
                "resolved_feature_mode": mode_meta.get("resolved_feature_mode"),
                "feature_warning": mode_meta.get("feature_warning", ""),
                "n_scope_rows": int(len(df_scope)),
                "n_rows_after_feature_filter": int(len(work)),
                "n_binned_rows": int(len(binned_out)),
                "n_embryos": int(assignments["embryo_id"].nunique()),
                "n_windows": int(len(window_records)),
                "n_base_features": int(len(base_features)),
                "n_analysis_features": int(len(analysis_features)),
                "n_binned_features": int(len(binned_feature_cols)),
                "best_k": int(best_k),
                "n_jobs": int(n_jobs),
                "elapsed_seconds": float(time.time() - start),
                **pca_meta,
            }
        ]
    )
    feature_benchmark.to_csv(out_dir / "feature_benchmark.tsv", sep="\t", index=False)

    config = {
        "scope_key": scope_key,
        "feature_mode": feature_mode,
        "resolved_feature_mode": mode_meta.get("resolved_feature_mode"),
        "analysis_features": analysis_features,
        "binned_feature_cols": binned_feature_cols,
        "bin_width": bin_width,
        "k_neighbors": k_neighbors,
        "topq": topq,
        "k_values": list(map(int, k_values)),
        "n_bootstrap": n_bootstrap,
        "bootstrap_frac": bootstrap_frac,
        "min_time_bins": min_time_bins,
        "min_embryos": min_embryos,
        "seed": seed,
        "n_jobs": n_jobs,
        "input_rows": int(len(df_scope)),
        "n_embryos": int(assignments["embryo_id"].nunique()),
        "n_windows": int(len(window_records)),
        "best_k": int(best_k),
        **mode_meta,
        **pca_meta,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    return ScopeRunResult(
        scope_key=scope_key,
        output_dir=out_dir,
        n_embryos=int(assignments["embryo_id"].nunique()),
        n_windows=int(len(window_records)),
        best_k=int(best_k),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Path containing datasets_manifest.csv and experiments/*/input_core.csv",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default="",
        help="Comma-separated dataset_ids from datasets_manifest.csv (default: all).",
    )
    parser.add_argument(
        "--scope-mode",
        choices=["within_experiment", "within_dataset", "cross_dataset"],
        default="within_experiment",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["auto", "zmu_b", "core", "pca95"],
        default="auto",
    )
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--topq", type=float, default=0.25)
    parser.add_argument("--k-values", type=str, default="2,3,4,5,6,7,8")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--bootstrap-frac", type=float, default=0.8)
    parser.add_argument("--min-time-bins", type=int, default=3)
    parser.add_argument("--min-embryos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    parser.add_argument("--drop-dead", action="store_true", default=True)
    parser.add_argument("--keep-dead", action="store_true", help="Disable dead_flag2 filtering")
    parser.add_argument("--require-focus-flag", action="store_true", help="Drop rows where focus_flag is True")
    parser.add_argument("--require-well-qc", action="store_true", help="Drop rows where well_qc_flag is True")
    parser.add_argument("--require-sam2-qc", action="store_true", help="Drop rows where sam2_qc_flag is True")

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for run outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_ids = [x.strip() for x in args.dataset_ids.split(",") if x.strip()] if args.dataset_ids else None
    resolved_ids = _resolve_dataset_ids(args.input_root, dataset_ids)

    df_all = _load_input_tables(args.input_root, resolved_ids)
    df_all = _apply_filters(
        df_all,
        drop_dead=bool(args.drop_dead and not args.keep_dead),
        require_focus_flag=bool(args.require_focus_flag),
        require_well_qc=bool(args.require_well_qc),
        require_sam2_qc=bool(args.require_sam2_qc),
    )

    scopes = _build_scopes(df_all, args.scope_mode)
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    results: List[ScopeRunResult] = []
    for i, (scope_key, df_scope) in enumerate(scopes):
        if df_scope["embryo_id"].nunique() < max(k_values, default=2):
            print(f"Skipping {scope_key}: not enough embryos for requested k range")
            continue

        result = run_scope(
            scope_key=scope_key,
            df_scope=df_scope,
            feature_mode=args.feature_mode,
            bin_width=float(args.bin_width),
            k_neighbors=int(args.k_neighbors),
            topq=float(args.topq),
            k_values=k_values,
            n_bootstrap=int(args.n_bootstrap),
            bootstrap_frac=float(args.bootstrap_frac),
            min_time_bins=int(args.min_time_bins),
            min_embryos=int(args.min_embryos),
            seed=int(args.seed + i),
            n_jobs=int(args.n_jobs),
            output_root=args.output_root,
        )
        results.append(result)
        print(
            f"[{scope_key}] wrote {result.output_dir} | embryos={result.n_embryos} "
            f"windows={result.n_windows} best_k={result.best_k}"
        )

    if not results:
        raise RuntimeError("No scope runs completed successfully.")

    summary = pd.DataFrame(
        [
            {
                "scope_key": r.scope_key,
                "output_dir": str(r.output_dir),
                "n_embryos": r.n_embryos,
                "n_windows": r.n_windows,
                "best_k": r.best_k,
            }
            for r in results
        ]
    )
    summary_path = args.output_root / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
