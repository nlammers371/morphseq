#!/usr/bin/env python
"""Generate manual-review diagnostics for embryo-first persistence outputs.

Creates a compact plot bundle for one run directory, including:
- resolution scan metrics (bootstrap stability + silhouette)
- resolution-over-time heatmap (ARI vs global clusters, time x k)
- cluster presence over time
- persistence matrix heatmap sorted by cluster
- classification AUROC over time (if validation outputs exist)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PERSISTENCE_ROOT = Path(__file__).resolve().parent / "output" / "embryo_first_persistence"


def _resolve_latest_run(root: Path) -> Path:
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories found in: {root}")
    return runs[-1]


def _cluster_distance_matrix(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    kwargs = {"n_clusters": int(n_clusters), "linkage": "average"}
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return model.fit_predict(distance_matrix)


def _build_window_adjacency(window_df: pd.DataFrame, feature_cols: Sequence[str], k_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    embryo_ids = window_df["embryo_id"].astype(str).to_numpy()
    X = window_df[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(axis=0))
    values = X.to_numpy(dtype=float)

    if len(values) < 2:
        return embryo_ids, np.eye(len(values), dtype=float)

    scaled = StandardScaler().fit_transform(values)
    k_eff = max(1, min(int(k_neighbors), len(values) - 1))

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


def _load_required(run_dir: Path) -> Dict[str, object]:
    req = {
        "config": run_dir / "config.json",
        "resolution_scan": run_dir / "resolution_scan.tsv",
        "assignments": run_dir / "cohort_assignments.tsv",
        "binned": run_dir / "binned_data.tsv",
        "persistence_mean": run_dir / "persistence_mean.npy",
    }
    missing = [str(p) for p in req.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in run dir:\n" + "\n".join(missing))

    with req["config"].open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    return {
        "config": config,
        "resolution_scan": pd.read_csv(req["resolution_scan"], sep="\t"),
        "assignments": pd.read_csv(req["assignments"], sep="\t"),
        "binned": pd.read_csv(req["binned"], sep="\t"),
        "persistence_mean": np.load(req["persistence_mean"]),
    }


def _plot_resolution_scan(scan: pd.DataFrame, out_path: Path) -> None:
    df = scan.sort_values("k").copy()

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df["k"], df["bootstrap_mean_ari"], marker="o", color="#1f77b4", label="Bootstrap mean ARI")
    if {"bootstrap_ci_low", "bootstrap_ci_high"}.issubset(set(df.columns)):
        ax1.fill_between(df["k"], df["bootstrap_ci_low"], df["bootstrap_ci_high"], alpha=0.2, color="#1f77b4")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Bootstrap ARI", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["silhouette"], marker="s", color="#ff7f0e", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _compute_resolution_over_time(
    binned: pd.DataFrame,
    p_mean: np.ndarray,
    assignments: pd.DataFrame,
    k_values: Sequence[int],
    feature_cols: Sequence[str],
    k_neighbors: int,
) -> pd.DataFrame:
    embryo_order = assignments["embryo_id"].astype(str).tolist()
    emb_to_idx = {e: i for i, e in enumerate(embryo_order)}

    D_global = 1.0 - np.clip(p_mean, 0.0, 1.0)
    np.fill_diagonal(D_global, 0.0)

    global_labels_by_k: Dict[int, np.ndarray] = {}
    for k in sorted(set(int(x) for x in k_values if int(x) > 1 and int(x) < len(embryo_order))):
        global_labels_by_k[k] = _cluster_distance_matrix(D_global, n_clusters=k)

    rows: List[Dict[str, float]] = []

    for time_bin, gdf in binned.groupby("time_bin", sort=True):
        gdf = gdf.dropna(subset=list(feature_cols)).copy()
        gdf = gdf.drop_duplicates(subset=["embryo_id"])  # one row per embryo in this bin
        if gdf["embryo_id"].nunique() < 4:
            continue

        local_emb, local_adj = _build_window_adjacency(gdf, feature_cols, k_neighbors=k_neighbors)
        D_local = 1.0 - np.clip(local_adj, 0.0, 1.0)
        np.fill_diagonal(D_local, 0.0)

        present_idx = np.array([emb_to_idx[e] for e in local_emb if e in emb_to_idx], dtype=int)
        if present_idx.size < 4:
            continue

        for k, global_labels in global_labels_by_k.items():
            if present_idx.size <= k:
                continue
            local_labels = _cluster_distance_matrix(D_local, n_clusters=k)
            global_sub = global_labels[present_idx]
            ari = adjusted_rand_score(global_sub, local_labels)
            rows.append(
                {
                    "time_bin": float(time_bin),
                    "k": int(k),
                    "n_embryos_in_bin": int(len(local_emb)),
                    "ari_vs_global": float(ari),
                }
            )

    return pd.DataFrame(rows)


def _plot_resolution_over_time(heat_df: pd.DataFrame, out_path: Path) -> None:
    if heat_df.empty:
        return

    pivot = heat_df.pivot_table(index="k", columns="time_bin", values="ari_vs_global", aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(max(8, 0.22 * pivot.shape[1]), max(4, 0.7 * pivot.shape[0])))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis", vmin=-0.1, vmax=1.0)

    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(k) for k in pivot.index.tolist()])

    xticks = np.arange(pivot.shape[1])
    ax.set_xticks(xticks[:: max(1, len(xticks) // 12)])
    ax.set_xticklabels([f"{pivot.columns[i]:.0f}" for i in xticks[:: max(1, len(xticks) // 12)]])

    ax.set_xlabel("Time bin (hpf)")
    ax.set_ylabel("k")
    ax.set_title("Resolution-over-time: ARI(window clusters vs global clusters)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ARI")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_presence_over_time(binned: pd.DataFrame, out_path: Path) -> None:
    work = binned[["time_bin", "embryo_id", "cluster"]].dropna().drop_duplicates(subset=["time_bin", "embryo_id"])
    if work.empty:
        return

    table = pd.crosstab(work["time_bin"], work["cluster"])
    prop = table.div(table.sum(axis=1), axis=0).sort_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in prop.columns:
        ax.plot(prop.index.to_numpy(dtype=float), prop[col].to_numpy(dtype=float), marker="o", linewidth=1.5, label=str(col))

    ax.set_xlabel("Time bin (hpf)")
    ax.set_ylabel("Proportion in bin")
    ax.set_title("Cluster composition over developmental time")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(title="Cluster", loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_persistence_matrix(p_mean: np.ndarray, assignments: pd.DataFrame, out_path: Path) -> None:
    work = assignments[["embryo_id", "cluster"]].copy()
    work["cluster"] = work["cluster"].astype(str)
    order = work.sort_values(["cluster", "embryo_id"]).index.to_numpy()

    sorted_mat = p_mean[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(sorted_mat, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Persistence matrix sorted by cluster")
    ax.set_xlabel("Embryos (sorted)")
    ax.set_ylabel("Embryos (sorted)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Persistence weight")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_classification_from_validation(run_dir: Path, out_path: Path) -> bool:
    cls_path = run_dir / "validation" / "classification_validation.tsv"
    if not cls_path.exists():
        return False

    cls_df = pd.read_csv(cls_path, sep="\t")
    if cls_df.empty:
        return False

    time_col = "time_bin_center" if "time_bin_center" in cls_df.columns else "time_bin"
    auroc_col = "auroc_obs" if "auroc_obs" in cls_df.columns else "auroc_observed"
    p_col = "pval" if "pval" in cls_df.columns else "p_value"
    class_col = "positive" if "positive" in cls_df.columns else "positive_class"

    if class_col not in cls_df.columns:
        return False

    labels = sorted(cls_df[class_col].dropna().astype(str).unique().tolist())
    if not labels:
        return False

    fig, axes = plt.subplots(len(labels), 1, figsize=(10, max(3.0, 2.4 * len(labels))), sharex=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, lab in zip(axes, labels):
        sub = cls_df[cls_df[class_col].astype(str) == lab].sort_values(time_col)
        x = sub[time_col].to_numpy(dtype=float)
        y = sub[auroc_col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.5, label=f"{lab} vs rest")

        if {"auroc_null_mean", "auroc_null_std"}.issubset(set(sub.columns)):
            mu = sub["auroc_null_mean"].to_numpy(dtype=float)
            sd = sub["auroc_null_std"].to_numpy(dtype=float)
            ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), alpha=0.2, color="gray", label="null ±1 sd")
            ax.plot(x, mu, color="gray", linewidth=1.0, alpha=0.9)

        if p_col in sub.columns:
            sig = sub[p_col].to_numpy(dtype=float) < 0.05
            ax.scatter(x[sig], y[sig], color="crimson", s=26, zorder=3)

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        ax.set_title(f"One-vs-rest: {lab}")

    axes[-1].set_xlabel("Time bin center (hpf)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--persistence-root", type=Path, default=DEFAULT_PERSISTENCE_ROOT)
    parser.add_argument("--k-neighbors", type=int, default=None, help="Override kNN used for per-window diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir else _resolve_latest_run(args.persistence_root)

    payload = _load_required(run_dir)
    config = payload["config"]
    scan = payload["resolution_scan"]
    assignments = payload["assignments"]
    binned = payload["binned"]
    p_mean = payload["persistence_mean"]

    fig_dir = run_dir / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_resolution_scan(scan, fig_dir / "resolution_scan_metrics.png")

    k_values = scan["k"].astype(int).tolist()
    feature_cols = [c for c in config.get("binned_feature_cols", []) if c in binned.columns]
    if not feature_cols:
        feature_cols = [c for c in binned.columns if c.endswith("_binned")][:8]

    k_neighbors = int(args.k_neighbors) if args.k_neighbors is not None else int(config.get("k_neighbors", 15))
    over_time = _compute_resolution_over_time(
        binned,
        p_mean=p_mean,
        assignments=assignments,
        k_values=k_values,
        feature_cols=feature_cols,
        k_neighbors=k_neighbors,
    )
    over_time.to_csv(fig_dir / "resolution_scan_over_time.tsv", sep="\t", index=False)
    _plot_resolution_over_time(over_time, fig_dir / "resolution_scan_over_time_heatmap.png")

    _plot_cluster_presence_over_time(binned, fig_dir / "cluster_presence_over_time.png")
    _plot_persistence_matrix(p_mean, assignments, fig_dir / "persistence_matrix_sorted.png")
    _plot_classification_from_validation(run_dir, fig_dir / "classification_ovr_auroc_over_time.png")

    print(f"Run directory: {run_dir}")
    print(f"Wrote plot bundle to: {fig_dir}")


if __name__ == "__main__":
    main()
