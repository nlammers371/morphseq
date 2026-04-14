"""Generate a per-genotype × per-time-bin support table.

Outputs:
  results/support_table.csv         — unique embryo counts per (genotype, time_bin)
  results/supported_window.json     — {t_min, t_max, min_support} for condensation
  figures/support_heatmap.png       — annotated heatmap for visual inspection

Run:
  conda run -n segmentation_grounded_sam --no-capture-output \\
      python results/mcolon/20260413_tfap2_followup/scripts/01_support_table.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BIN_WIDTH, MIN_SUPPORT, load_aggregate_dataframe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    run_dir = Path(__file__).resolve().parents[1]
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_aggregate_dataframe()
    print(f"Loaded {len(df):,} rows, {df['embryo_id'].nunique():,} unique embryos, "
          f"{df['genotype'].nunique()} genotypes")

    # Assign 2 hpf time bins
    df["time_bin"] = (df["predicted_stage_hpf"] // BIN_WIDTH * BIN_WIDTH).astype(float)

    # Count unique embryos per (genotype, time_bin)
    support = (
        df.groupby(["genotype", "time_bin"])["embryo_id"]
        .nunique()
        .unstack(level="time_bin", fill_value=0)
        .sort_index()
    )

    support_path = results_dir / "support_table.csv"
    support.to_csv(support_path)
    print(f"\nSupport table saved: {support_path}")
    print(f"  Genotypes: {len(support)}")
    print(f"  Time bins: {len(support.columns)} "
          f"({float(support.columns.min()):.0f}–{float(support.columns.max()):.0f} hpf)")

    # Print table to console
    print("\n" + "=" * 80)
    print("Support table (unique embryos per genotype per 2-hpf bin):")
    print("=" * 80)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(support.to_string())
    print("=" * 80)

    # Find contiguous window where ALL genotypes have >= MIN_SUPPORT
    all_covered = (support >= MIN_SUPPORT).all(axis=0)
    covered_bins = sorted(float(b) for b in support.columns[all_covered])

    if not covered_bins:
        print(f"\nWARNING: No time bin has all genotypes with >= {MIN_SUPPORT} embryos.")
        print("Minimum support per genotype across all bins:")
        print(support.min(axis=1).sort_values().to_string())
        return

    t_min, t_max = covered_bins[0], covered_bins[-1]
    n_bins = len(covered_bins)
    print(f"\nSupported window (all genotypes >= {MIN_SUPPORT} embryos):")
    print(f"  t_min = {t_min} hpf, t_max = {t_max} hpf  ({n_bins} bins)")

    window = {"t_min": t_min, "t_max": t_max, "min_support": MIN_SUPPORT}
    window_path = results_dir / "supported_window.json"
    window_path.write_text(json.dumps(window, indent=2) + "\n")
    print(f"  Saved: {window_path}")

    # Summary stats within window
    window_support = support[covered_bins]
    print(f"\nWithin-window support summary:")
    print(f"  Min embryos across all cells: {int(window_support.min().min())}")
    print(f"  Median embryos per cell: {window_support.median().median():.1f}")
    per_genotype_min = window_support.min(axis=1).sort_values()
    print(f"\n  Minimum within-window support per genotype (ascending):")
    for gt, val in per_genotype_min.items():
        flag = " <-- smallest" if val == per_genotype_min.min() else ""
        print(f"    {gt}: {int(val)}{flag}")

    # ---------------------------------------------------------------------------
    # Heatmap
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(12, len(support.columns) * 0.35), max(6, len(support) * 0.4)))

    vmax = int(support.values.max())
    im = ax.imshow(
        support.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        origin="upper",
    )

    # Annotate cells
    for i in range(len(support.index)):
        for j in range(len(support.columns)):
            val = int(support.iloc[i, j])
            color = "white" if val > vmax * 0.65 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color=color)

    # Highlight the supported window with a border
    if covered_bins:
        col_indices = [list(support.columns).index(b) for b in covered_bins]
        j0, j1 = col_indices[0] - 0.5, col_indices[-1] + 0.5
        rect = plt.Rectangle(
            (j0, -0.5), j1 - j0, len(support) + 0.0,
            linewidth=2.5, edgecolor="#2166AC", facecolor="none",
            label=f"Supported window (≥{MIN_SUPPORT} per cell)",
        )
        ax.add_patch(rect)
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xticks(range(len(support.columns)))
    ax.set_xticklabels([f"{float(b):.0f}" for b in support.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(support.index)))
    ax.set_yticklabels(support.index.tolist(), fontsize=8)
    ax.set_xlabel("Time bin (hpf)", fontsize=10)
    ax.set_ylabel("Genotype", fontsize=10)
    ax.set_title(
        f"TFAP2 embryo support per genotype × time bin (bin_width={BIN_WIDTH} hpf)\n"
        f"Blue box: window where all genotypes ≥ {MIN_SUPPORT} embryos  "
        f"({t_min:.0f}–{t_max:.0f} hpf)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label="Unique embryos", shrink=0.6)
    plt.tight_layout()

    heatmap_path = figures_dir / "support_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved: {heatmap_path}")


if __name__ == "__main__":
    main()
