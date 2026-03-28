"""
PBX crispant classification heatmap visualization.

Loads pre-computed classification CSVs and generates AUROC heatmaps
with significance borders for:
1. One-vs-all comparisons
2. All crispants vs wik-ab (true null control)

Generates 3 panels per mode (curvature, length, embedding).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def plot_auroc_heatmap(ax, comparisons_df, title, sig_threshold=0.01):
    """
    Plot AUROC heatmap with genotype x time_bin, bordered significance cells.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    comparisons_df : pd.DataFrame
        Must have columns: positive, time_bin_center, auroc_obs, pval
    title : str
    sig_threshold : float
    """
    pivot = comparisons_df.pivot_table(
        index="positive", columns="time_bin_center", values="auroc_obs"
    )
    sig = comparisons_df.pivot_table(
        index="positive", columns="time_bin_center", values="pval"
    ) <= sig_threshold

    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    # Keep controls first, then the crispants in a stable biological order.
    desired_order = [
        "inj_ctrl",
        "wik_ab",
        "pbx1b_crispant",
        "pbx4_crispant",
        "pbx1b_pbx4_crispant",
    ]
    row_order = [g for g in desired_order if g in pivot.index]
    # Add any remaining genotypes not in desired_order
    row_order.extend([g for g in pivot.index if g not in row_order])
    pivot = pivot.loc[row_order]
    sig = sig.loc[row_order]

    # Show missing genotype-time bins in grey so low-sample gaps are explicit.
    cmap = plt.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")
    heatmap_values = np.ma.masked_invalid(pivot.values.astype(float))
    im = ax.imshow(heatmap_values, vmin=0.3, vmax=1.0, cmap=cmap, aspect="auto")

    # Border significant cells with bold black Rectangle patches
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            if pd.notna(pivot.loc[row, col]) and sig.loc[row, col]:
                rect = Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=2.5, edgecolor="black", facecolor="none"
                )
                ax.add_patch(rect)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], rotation=60, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Genotype")
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=8)
    plt.colorbar(im, ax=ax, label="AUROC")


def main() -> None:
    # Get bin_width from argument or autodetect from results directory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-width", type=float, default=None, help="Time bin width in hpf (e.g., 2.0, 4.0)")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent.parent.parent

    # If no bin_width specified, try to find the latest one
    if args.bin_width is None:
        results_base = run_dir / "results"
        bin_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith("bin_width_")])
        if bin_dirs:
            latest_dir = bin_dirs[-1]
            bin_width_str = latest_dir.name
            print(f"Auto-detected: {bin_width_str}")
        else:
            print("No bin_width_* directories found. Please run 02_classification_only.py first.")
            return
    else:
        bin_width_str = f"bin_width_{args.bin_width:.1f}hpf"

    figures_dir = run_dir / "figures" / bin_width_str
    results_dir = run_dir / "results" / bin_width_str
    classification_dir = results_dir / "classification"
    classification_fig_dir = figures_dir / "classification"

    if not classification_dir.exists():
        print(f"Classification results not found at {classification_dir}")
        print("Run scripts/02_classification_only.py first.")
        return

    classification_fig_dir.mkdir(parents=True, exist_ok=True)

    # Load classification CSV files
    print(f"Loading classification results from {classification_dir}...")

    experiment_label = "20260304_20260306"
    vs_wik_ab_file = classification_dir / f"{experiment_label}_all_crispants_vs_wik_ab_curvature_comparisons.csv"

    if not vs_wik_ab_file.exists():
        print(f"Classification file not found: {vs_wik_ab_file}")
        return

    feature_sets = ["curvature", "length", "embedding"]

    # =========================================================================
    # Plot all crispants vs wik-ab heatmaps
    # =========================================================================
    print("Generating all crispants vs wik-ab AUROC heatmaps...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for ax, feature_set in zip(axes, feature_sets):
        comparisons_file = classification_dir / f"{experiment_label}_all_crispants_vs_wik_ab_{feature_set}_comparisons.csv"
        if comparisons_file.exists():
            comparisons_df = pd.read_csv(comparisons_file)
            if not comparisons_df.empty:
                plot_auroc_heatmap(
                    ax,
                    comparisons_df,
                    f"All Crispants vs wik-ab: {feature_set}",
                    sig_threshold=0.01
                )
        else:
            ax.text(0.5, 0.5, f"No {feature_set} data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"All Crispants vs wik-ab: {feature_set}")

    fig.tight_layout()
    vs_wik_ab_heatmap_path = classification_fig_dir / f"{experiment_label}_all_crispants_vs_wik_ab_heatmaps.png"
    fig.savefig(vs_wik_ab_heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {vs_wik_ab_heatmap_path}")

    print(f"\nAll heatmaps saved to: {classification_fig_dir}")


if __name__ == "__main__":
    main()
