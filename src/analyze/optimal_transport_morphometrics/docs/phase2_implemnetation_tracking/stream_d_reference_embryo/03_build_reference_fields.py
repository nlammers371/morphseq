#!/usr/bin/env python3
"""Build per-time-bin WT reference fields from exported OT artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.optimal_transport_morphometrics.uot_masks.reference_embryo import build_reference_field


DEFAULT_EXPORT_ROOT = Path(__file__).resolve().parent / "ot_24_48_exports"
DEFAULT_TRANSITIONS = Path(__file__).resolve().parent / "cohort_selection" / "cohort_transition_manifest.csv"
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


def _plot_reference_field(ref_npz: Path, png_out: Path, title: str) -> None:
    data = np.load(ref_npz)
    vel = np.asarray(data["velocity_yx"], dtype=np.float32)
    support = np.asarray(data["support_mask"], dtype=bool)
    created = np.asarray(data["mass_created"], dtype=np.float32)
    destroyed = np.asarray(data["mass_destroyed"], dtype=np.float32)
    speed = np.linalg.norm(vel, axis=-1)
    speed = np.where(support, speed, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    im0 = axes[0].imshow(speed, cmap="inferno")
    axes[0].set_title("Reference Speed")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(created, cmap="magma")
    axes[1].set_title("Reference Mass Created")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(destroyed, cmap="viridis")
    axes[2].set_title("Reference Mass Destroyed")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.savefig(png_out, dpi=180)
    plt.close(fig)


def run_build(args: argparse.Namespace) -> None:
    export_root = Path(args.export_root).resolve()
    output_root = Path(args.output_root).resolve()
    reference_dir = output_root / "reference_fields"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_plot_dir = output_root / "deviation_plots" / "reference_visuals"
    reference_plot_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = export_root / "ot_pair_metrics.parquet"
    artifact_root = export_root / "pair_artifacts"
    metrics = pd.read_parquet(metrics_path)
    transitions = pd.read_csv(args.transitions, low_memory=False)

    m = metrics[(metrics["run_id"].astype(str) == args.run_id) & (metrics["success"] == True)].copy()
    t = transitions[(transitions["analysis_use"] == True) & (transitions["set_type"] == "reference_wt")].copy()
    merged = t.merge(
        m[["pair_id", "run_id", "set_type", "bin_src_hpf", "bin_tgt_hpf"]],
        on=["pair_id"],
        how="inner",
        suffixes=("_manifest", "_metrics"),
    )
    if merged.empty:
        raise ValueError(f"No successful reference rows found for run_id={args.run_id}")

    summary_rows: List[Dict] = []
    for (b0, b1), g in merged.groupby(["bin_src_hpf_manifest", "bin_tgt_hpf_manifest"], sort=True):
        results = []
        pair_ids = []
        for row in g.itertuples(index=False):
            pair_id = str(row.pair_id)
            fields_path = artifact_root / pair_id / "fields.npz"
            if not fields_path.exists():
                continue
            results.append(_load_result_like(fields_path))
            pair_ids.append(pair_id)
        if len(results) < args.min_refs_per_bin:
            continue

        ref = build_reference_field(results)
        stem = f"ref_{int(round(float(b0))):02d}_to_{int(round(float(b1))):02d}"
        ref_npz = reference_dir / f"{stem}.npz"
        np.savez_compressed(
            ref_npz,
            velocity_yx=ref.velocity_yx.astype(np.float32),
            mass_created=ref.mass_created.astype(np.float32),
            mass_destroyed=ref.mass_destroyed.astype(np.float32),
            support_mask=ref.support_mask.astype(np.uint8),
            n_embryos=np.array([ref.n_embryos], dtype=np.int32),
            pair_ids=np.array(pair_ids, dtype=object),
            bin_src_hpf=np.array([float(b0)], dtype=np.float32),
            bin_tgt_hpf=np.array([float(b1)], dtype=np.float32),
        )
        _plot_reference_field(
            ref_npz=ref_npz,
            png_out=reference_plot_dir / f"{stem}.png",
            title=f"Reference WT: {int(round(float(b0)))}->{int(round(float(b1)))} hpf (n={ref.n_embryos})",
        )
        summary_rows.append(
            {
                "bin_src_hpf": float(b0),
                "bin_tgt_hpf": float(b1),
                "n_embryos": int(ref.n_embryos),
                "support_pixels": int(ref.support_mask.sum()),
                "reference_npz": str(ref_npz),
                "pair_ids": ";".join(pair_ids),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["bin_src_hpf", "bin_tgt_hpf"]).reset_index(drop=True)
    summary_path = reference_dir / f"reference_summary_{args.run_id}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote reference summary: {summary_path}")
    print(f"Wrote reference fields in: {reference_dir}")
    print(f"Wrote reference plots in: {reference_plot_dir}")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WT reference fields from batch OT exports.")
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default="phase2_24_48_ott_v1")
    parser.add_argument("--min-refs-per-bin", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run_build(parse_args())
