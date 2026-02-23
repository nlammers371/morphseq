#!/usr/bin/env python3
"""Compute WT-reference deviation metrics for exported cohort OT results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.optimal_transport_morphometrics.uot_masks.reference_embryo import (
    ReferenceField,
    compute_deviation_from_reference,
)


DEFAULT_EXPORT_ROOT = Path(__file__).resolve().parent / "ot_24_48_exports"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent


@dataclass
class _ResultLike:
    mass_created_px: np.ndarray
    mass_destroyed_px: np.ndarray
    velocity_px_per_frame_yx: np.ndarray


def _load_result_like(fields_path: Path) -> _ResultLike:
    data = np.load(fields_path)
    return _ResultLike(
        mass_created_px=np.asarray(data["mass_created_px"], dtype=np.float32),
        mass_destroyed_px=np.asarray(data["mass_destroyed_px"], dtype=np.float32),
        velocity_px_per_frame_yx=np.asarray(data["velocity_px_per_frame_yx"], dtype=np.float32),
    )


def _load_reference_fields(reference_dir: Path) -> Dict[Tuple[float, float], ReferenceField]:
    refs: Dict[Tuple[float, float], ReferenceField] = {}
    for p in sorted(reference_dir.glob("ref_*_to_*.npz")):
        data = np.load(p)
        b0 = float(data["bin_src_hpf"][0])
        b1 = float(data["bin_tgt_hpf"][0])
        refs[(b0, b1)] = ReferenceField(
            velocity_yx=np.asarray(data["velocity_yx"], dtype=np.float32),
            mass_created=np.asarray(data["mass_created"], dtype=np.float32),
            mass_destroyed=np.asarray(data["mass_destroyed"], dtype=np.float32),
            support_mask=np.asarray(data["support_mask"], dtype=bool),
            n_embryos=int(data["n_embryos"][0]),
        )
    return refs


def _plot_group_trends(dev_df: pd.DataFrame, out_root: Path) -> None:
    plot_dir = out_root / "deviation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = dev_df.copy()
    df["bin_mid_hpf"] = (df["bin_src_hpf"] + df["bin_tgt_hpf"]) / 2.0
    group_stats = (
        df.groupby(["set_type", "bin_mid_hpf"])
        .agg(
            rmse_velocity_mean=("rmse_velocity", "mean"),
            rmse_velocity_std=("rmse_velocity", "std"),
            cosine_mean=("cosine_similarity", "mean"),
            cosine_std=("cosine_similarity", "std"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for set_type, g in group_stats.groupby("set_type"):
        x = g["bin_mid_hpf"].to_numpy()
        y = g["rmse_velocity_mean"].to_numpy()
        yerr = g["rmse_velocity_std"].fillna(0.0).to_numpy()
        axes[0].plot(x, y, marker="o", label=set_type)
        axes[0].fill_between(x, y - yerr, y + yerr, alpha=0.2)

        y2 = g["cosine_mean"].to_numpy()
        yerr2 = g["cosine_std"].fillna(0.0).to_numpy()
        axes[1].plot(x, y2, marker="o", label=set_type)
        axes[1].fill_between(x, y2 - yerr2, y2 + yerr2, alpha=0.2)

    axes[0].set_title("Deviation from WT Reference: RMSE")
    axes[1].set_title("Deviation from WT Reference: Cosine Similarity")
    axes[0].set_xlabel("Bin Midpoint (hpf)")
    axes[1].set_xlabel("Bin Midpoint (hpf)")
    axes[0].set_ylabel("rmse_velocity")
    axes[1].set_ylabel("cosine_similarity")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.savefig(plot_dir / "deviation_group_trends.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    order = ["reference_wt", "heldout_wt", "mutant"]
    data = [df.loc[df["set_type"] == s, "rmse_velocity"].to_numpy() for s in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_title("RMSE Distribution by Cohort")
    ax.set_ylabel("rmse_velocity")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(plot_dir / "deviation_rmse_boxplot.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    export_root = Path(args.export_root).resolve()
    output_root = Path(args.output_root).resolve()
    reference_dir = output_root / "reference_fields"
    refs = _load_reference_fields(reference_dir)
    if not refs:
        raise ValueError(f"No reference fields found in {reference_dir}")

    metrics_path = export_root / "ot_pair_metrics.parquet"
    artifact_root = export_root / "pair_artifacts"
    metrics = pd.read_parquet(metrics_path)
    m = metrics[
        (metrics["run_id"].astype(str) == args.run_id)
        & (metrics["success"] == True)
        & (metrics["is_control_pair"] == False)
    ].copy()
    if m.empty:
        raise ValueError(f"No successful non-control rows for run_id={args.run_id}")

    rows = []
    for row in m.itertuples(index=False):
        b0 = float(row.bin_src_hpf)
        b1 = float(row.bin_tgt_hpf)
        key = (b0, b1)
        if key not in refs:
            continue
        pair_id = str(row.pair_id)
        fields_path = artifact_root / pair_id / "fields.npz"
        if not fields_path.exists():
            continue
        result = _load_result_like(fields_path)
        dev = compute_deviation_from_reference(result, refs[key])
        rows.append(
            {
                "run_id": str(row.run_id),
                "pair_id": pair_id,
                "embryo_id": str(row.src_embryo_id_manifest)
                if hasattr(row, "src_embryo_id_manifest")
                else str(row.src_embryo_id),
                "set_type": str(row.set_type),
                "set_rank": int(row.set_rank) if pd.notna(row.set_rank) else np.nan,
                "genotype": str(row.genotype),
                "bin_src_hpf": b0,
                "bin_tgt_hpf": b1,
                **dev,
            }
        )
    dev_df = pd.DataFrame(rows).sort_values(["set_type", "embryo_id", "bin_src_hpf"]).reset_index(drop=True)
    out_csv = output_root / "deviation_plots" / f"deviation_metrics_{args.run_id}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dev_df.to_csv(out_csv, index=False)
    _plot_group_trends(dev_df, out_root=output_root)
    print(f"Wrote deviations: {out_csv}")
    print(dev_df.groupby("set_type")[["rmse_velocity", "cosine_similarity"]].mean().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute deviations vs WT reference.")
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default="phase2_24_48_ott_v1")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
