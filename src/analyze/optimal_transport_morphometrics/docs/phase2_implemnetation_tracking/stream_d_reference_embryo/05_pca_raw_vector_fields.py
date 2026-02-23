#!/usr/bin/env python3
"""PCA on raw OT velocity fields (projected with fixed WT-reference support union)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


DEFAULT_EXPORT_ROOT = Path(__file__).resolve().parent / "ot_24_48_exports"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent


def _load_reference_union_mask(reference_dir: Path) -> np.ndarray:
    masks = []
    for p in sorted(reference_dir.glob("ref_*_to_*.npz")):
        data = np.load(p)
        masks.append(np.asarray(data["support_mask"], dtype=bool))
    if not masks:
        raise ValueError(f"No reference masks found in {reference_dir}")
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= m
    return union


def run(args: argparse.Namespace) -> None:
    export_root = Path(args.export_root).resolve()
    output_root = Path(args.output_root).resolve()
    pca_dir = output_root / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_parquet(export_root / "ot_pair_metrics.parquet")
    m = metrics[
        (metrics["run_id"].astype(str) == args.run_id)
        & (metrics["success"] == True)
        & (metrics["is_control_pair"] == False)
    ].copy()
    if m.empty:
        raise ValueError(f"No successful non-control rows for run_id={args.run_id}")

    union_mask = _load_reference_union_mask(output_root / "reference_fields")
    n_union = int(union_mask.sum())
    artifact_root = export_root / "pair_artifacts"

    vectors: List[np.ndarray] = []
    meta_rows = []
    for row in m.itertuples(index=False):
        pair_id = str(row.pair_id)
        fields_path = artifact_root / pair_id / "fields.npz"
        if not fields_path.exists():
            continue
        arr = np.load(fields_path)
        vel = np.asarray(arr["velocity_px_per_frame_yx"], dtype=np.float32)
        vec = vel[union_mask].reshape(-1).astype(np.float32)
        vectors.append(vec)
        meta_rows.append(
            {
                "run_id": str(row.run_id),
                "pair_id": pair_id,
                "embryo_id": str(row.src_embryo_id_manifest)
                if hasattr(row, "src_embryo_id_manifest")
                else str(row.src_embryo_id),
                "set_type": str(row.set_type),
                "set_rank": int(row.set_rank) if pd.notna(row.set_rank) else np.nan,
                "genotype": str(row.genotype),
                "bin_src_hpf": float(row.bin_src_hpf),
                "bin_tgt_hpf": float(row.bin_tgt_hpf),
                "bin_mid_hpf": (float(row.bin_src_hpf) + float(row.bin_tgt_hpf)) / 2.0,
            }
        )

    X = np.stack(vectors, axis=0)
    X = X - X.mean(axis=0, keepdims=True)
    n_components = int(min(args.n_components, X.shape[0] - 1))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    Z = pca.fit_transform(X)

    meta = pd.DataFrame(meta_rows)
    for i in range(Z.shape[1]):
        meta[f"PC{i+1}"] = Z[:, i]
    emb_csv = pca_dir / f"raw_velocity_pca_embeddings_{args.run_id}.csv"
    meta.to_csv(emb_csv, index=False)

    var_df = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_ratio_cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    var_csv = pca_dir / f"raw_velocity_pca_variance_{args.run_id}.csv"
    var_df.to_csv(var_csv, index=False)

    # Plot PC1/PC2 by cohort.
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    color_map = {"reference_wt": "#1f77b4", "heldout_wt": "#2ca02c", "mutant": "#d62728"}
    for set_type, g in meta.groupby("set_type"):
        ax.scatter(g["PC1"], g["PC2"], s=24, alpha=0.75, label=set_type, c=color_map.get(set_type, "#666666"))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA on Raw Velocity Fields")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(pca_dir / f"raw_velocity_pca_pc1_pc2_{args.run_id}.png", dpi=180)
    plt.close(fig)

    # Plot temporal trend in PCA space (group means by time bin).
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for set_type, g in meta.groupby("set_type"):
        s = (
            g.groupby("bin_mid_hpf")[["PC1", "PC2"]]
            .mean()
            .reset_index()
            .sort_values("bin_mid_hpf")
        )
        ax.plot(s["PC1"], s["PC2"], marker="o", label=set_type)
    ax.set_xlabel("PC1 mean")
    ax.set_ylabel("PC2 mean")
    ax.set_title("PCA Trajectory by Cohort (Bin Means)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(pca_dir / f"raw_velocity_pca_group_trajectories_{args.run_id}.png", dpi=180)
    plt.close(fig)

    print(f"Wrote PCA embeddings: {emb_csv}")
    print(f"Wrote variance table: {var_csv}")
    print(f"Union support pixels used: {n_union}")
    print(var_df.head(min(10, len(var_df))).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA on raw OT velocity fields.")
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default="phase2_24_48_ott_v1")
    parser.add_argument("--n-components", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
