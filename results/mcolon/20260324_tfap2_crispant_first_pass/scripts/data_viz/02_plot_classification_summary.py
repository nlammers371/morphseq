"""
TFAP2 crispant classification summary plots.

Loads pre-computed classification CSVs and generates:
1. AUROC-over-time plots (3 panels per mode: curvature, length, embedding)
2. AUROC heatmaps with significance borders (3 panels per mode)

Two modes: one_vs_all and each_vs_inj_ctrl
Two figures per mode: AUROC-over-time + heatmap
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def plot_auroc_heatmap(ax, comparisons_df, title, color_lookup=None, sig_threshold=0.01):
    """
    Plot AUROC heatmap with genotype x time_bin, bordered significance cells.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    comparisons_df : pd.DataFrame
        Must have columns: positive, time_bin_center, auroc_obs, pval
    title : str
    color_lookup : dict | None
        Map from genotype to color (for row labels, optional)
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

    # Sort rows by max auroc descending
    row_order = pivot.max(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]
    sig = sig.loc[row_order]

    # imshow with diverging colormap centered at 0.5
    im = ax.imshow(pivot.values, vmin=0.3, vmax=1.0, cmap="RdBu_r", aspect="auto")

    # Border significant cells with bold black Rectangle patches
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            if sig.loc[row, col]:
                rect = Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=2.5, edgecolor="black", facecolor="none"
                )
                ax.add_patch(rect)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Genotype")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="AUROC")


def main() -> None:
    run_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = run_dir / "figures"
    results_dir = run_dir / "results"
    classification_dir = results_dir / "classification"
    classification_fig_dir = figures_dir / "classification"

    if not classification_dir.exists():
        print(f"Classification results not found at {classification_dir}")
        print("Run scripts/data_gen/02_run_classification.py first.")
        return

    classification_fig_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(run_dir))
    sys.path.insert(0, str(project_root / "src"))

    from analyze.viz.styling import build_genotype_color_lookup
    from scripts.common import EXPERIMENT_LABEL
    from analyze.classification.viz.auroc_over_time import plot_aurocs_over_time
    from analyze.viz.plotting.faceting_engine import StyleSpec

    feature_sets = ["curvature", "length", "embedding"]
    feature_labels = {
        "curvature": "Curvature",
        "length": "Length",
        "embedding": "Embedding",
    }

    # =========================================================================
    # Process one_vs_all and each_vs_inj_ctrl modes
    # =========================================================================
    for mode in ["one_vs_all", "each_vs_inj_ctrl"]:
        print(f"\n=== {mode.upper()} ===")

        # Load comparisons CSVs for all 3 features
        mode_dfs = {}
        for feat in feature_sets:
            csv_path = classification_dir / f"{EXPERIMENT_LABEL}_{mode}_{feat}_comparisons.csv"
            if csv_path.exists():
                mode_dfs[feat] = pd.read_csv(csv_path)
                print(f"Loaded: {csv_path.name}")
            else:
                print(f"Warning: Missing {csv_path.name}")

        if not mode_dfs:
            print(f"  No CSV files found for {mode}, skipping.")
            continue

        # Build color lookup from unique genotypes across all features
        # Use a distinct color palette rather than genotype-based colors
        all_genotypes = set()
        for df in mode_dfs.values():
            if "positive" in df.columns:
                all_genotypes.update(df["positive"].dropna().unique())
        all_genotypes = sorted(all_genotypes)
        color_lookup = build_genotype_color_lookup(all_genotypes, warn_on_collision=False)

        # =====================================================================
        # Plot A: AUROC-over-time (3 panels, one per feature)
        # =====================================================================
        print(f"Plotting AUROC over time ({mode})...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, feat in enumerate(feature_sets):
            if feat not in mode_dfs:
                axes[idx].text(0.5, 0.5, "No data", ha="center", va="center",
                               transform=axes[idx].transAxes)
                axes[idx].set_title(feature_labels[feat])
                continue

            df = mode_dfs[feat]
            panel_title = feature_labels[feat]

            # Use plot_aurocs_over_time to render on pre-made axis
            # Note: plot_aurocs_over_time returns a figure, so we call it differently
            # For now, let's use a simplified inline version
            for genotype in sorted(df["positive"].dropna().unique()):
                gdf = df[df["positive"] == genotype]
                ax = axes[idx]
                color = color_lookup.get(genotype, "#000000")

                x = gdf["time_bin_center"].values
                y = gdf["auroc_obs"].values

                ax.plot(x, y, "o-", label=genotype, color=color, linewidth=2, markersize=5)

                # Null band
                if "auroc_null_mean" in gdf.columns and "auroc_null_std" in gdf.columns:
                    m = gdf["auroc_null_mean"].values
                    s = gdf["auroc_null_std"].values
                    ax.fill_between(x, m - s, m + s, color=color, alpha=0.1)

                # Significance markers
                if "pval" in gdf.columns:
                    sig_mask = (gdf["pval"] <= 0.01).values
                    if sig_mask.any():
                        ax.scatter(x[sig_mask], y[sig_mask], s=200, facecolors="none",
                                  edgecolors=color, linewidths=2.5, zorder=5)

            # Chance line
            x_min, x_max = axes[idx].get_xlim()
            axes[idx].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance (0.5)")
            axes[idx].set_ylim(0.3, 1.05)
            axes[idx].set_xlabel("Time (hpf)")
            axes[idx].set_ylabel("AUROC")
            axes[idx].set_title(panel_title)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].spines["top"].set_visible(False)
            axes[idx].spines["right"].set_visible(False)

        fig.suptitle(f"{EXPERIMENT_LABEL} {mode.replace('_', ' ').title()} AUROC over Time",
                     fontsize=14, fontweight="bold", y=1.02)

        # Place legend outside the rightmost panel
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
        fig.subplots_adjust(right=0.8)
        fig.tight_layout()

        auroc_ot_path = classification_fig_dir / f"{EXPERIMENT_LABEL}_{mode}_auroc_over_time.png"
        fig.savefig(auroc_ot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {auroc_ot_path.name}")

        # =====================================================================
        # Plot B: AUROC heatmap (3 panels, one per feature, sig cells bordered)
        # =====================================================================
        print(f"Plotting AUROC heatmap ({mode})...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, feat in enumerate(feature_sets):
            if feat not in mode_dfs:
                axes[idx].text(0.5, 0.5, "No data", ha="center", va="center",
                               transform=axes[idx].transAxes)
                axes[idx].set_title(feature_labels[feat])
                continue

            df = mode_dfs[feat]
            plot_auroc_heatmap(
                axes[idx],
                df,
                title=feature_labels[feat],
                color_lookup=color_lookup,
                sig_threshold=0.01,
            )

        fig.suptitle(f"{EXPERIMENT_LABEL} {mode.replace('_', ' ').title()} AUROC Heatmap",
                     fontsize=14, fontweight="bold", y=1.00)
        fig.tight_layout()

        heatmap_path = classification_fig_dir / f"{EXPERIMENT_LABEL}_{mode}_auroc_heatmap.png"
        fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {heatmap_path.name}")

    print(f"\nClassification summary plots saved to: {classification_fig_dir}")


if __name__ == "__main__":
    main()
