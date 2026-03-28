"""
PBX crispant AUROC-over-time curves visualization.

Loads pre-computed classification CSVs and generates AUROC curves
for all crispants vs wik-ab comparison across time bins.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-width", type=float, default=None, help="Time bin width in hpf (e.g., 2.0, 4.0)")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent.parent.parent

    # If no bin_width specified, autodetect
    if args.bin_width is None:
        results_base = run_dir / "results"
        bin_dirs = sorted([d for d in results_base.iterdir() if d.is_dir() and d.name.startswith("bin_width_")])
        if bin_dirs:
            latest_dir = bin_dirs[-1]
            bin_width_str = latest_dir.name
            print(f"Auto-detected: {bin_width_str}")
        else:
            print("No bin_width_* directories found.")
            return
    else:
        bin_width_str = f"bin_width_{args.bin_width:.1f}hpf"

    figures_dir = run_dir / "figures" / bin_width_str
    results_dir = run_dir / "results" / bin_width_str
    classification_dir = results_dir / "classification"
    classification_fig_dir = figures_dir / "classification"

    if not classification_dir.exists():
        print(f"Classification results not found at {classification_dir}")
        return

    classification_fig_dir.mkdir(parents=True, exist_ok=True)

    experiment_label = "20260304_20260306"
    feature_sets = ["curvature", "length", "embedding"]
    feature_labels = {
        "curvature": "Curvature",
        "length": "Length",
        "embedding": "Embedding",
    }
    mode = "all_crispants_vs_wik_ab"

    # Load comparisons CSVs for all 3 features
    mode_dfs = {}
    for feat in feature_sets:
        csv_path = classification_dir / f"{experiment_label}_{mode}_{feat}_comparisons.csv"
        if csv_path.exists():
            mode_dfs[feat] = pd.read_csv(csv_path)
            print(f"Loaded: {csv_path.name}")
        else:
            print(f"Warning: Missing {csv_path.name}")

    if not mode_dfs:
        print(f"No CSV files found, skipping.")
        return

    # Get all unique genotypes and order them
    all_genotypes = set()
    for df in mode_dfs.values():
        all_genotypes.update(df["positive_label"].dropna().unique())

    desired_order = ["inj_ctrl", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant"]
    ordered_genotypes = [g for g in desired_order if g in all_genotypes]
    ordered_genotypes.extend([g for g in sorted(all_genotypes) if g not in ordered_genotypes])

    # Color mapping for genotypes
    color_map = {
        "inj_ctrl": "#1f77b4",
        "pbx1b_crispant": "#ff7f0e",
        "pbx4_crispant": "#d62728",
        "pbx1b_pbx4_crispant": "#2ca02c",
    }

    # =========================================================================
    # Plot AUROC-over-time (3 panels, one per feature)
    # =========================================================================
    print(f"Plotting AUROC over time...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, feat in enumerate(feature_sets):
        if feat not in mode_dfs:
            axes[idx].text(0.5, 0.5, "No data", ha="center", va="center",
                          transform=axes[idx].transAxes)
            axes[idx].set_title(feature_labels[feat])
            continue

        df = mode_dfs[feat]
        panel_title = feature_labels[feat]

        # Plot each genotype
        for genotype in ordered_genotypes:
            gdf = df[df["positive_label"] == genotype]
            if gdf.empty:
                continue

            ax = axes[idx]
            color = color_map.get(genotype, "#808080")

            x = gdf["time_bin_center"].values
            y = gdf["auroc_obs"].values

            ax.plot(x, y, "o-", label=genotype, color=color, linewidth=2, markersize=5)

            # Null band (if available)
            if "auroc_null_mean" in gdf.columns and "auroc_null_std" in gdf.columns:
                m = gdf["auroc_null_mean"].values
                s = gdf["auroc_null_std"].values
                ax.fill_between(x, m - s, m + s, color=color, alpha=0.1)

            # Significance markers (p < 0.01)
            if "pval" in gdf.columns:
                sig_mask = (gdf["pval"] <= 0.01).values
                if sig_mask.any():
                    ax.scatter(x[sig_mask], y[sig_mask], s=200, facecolors="none",
                              edgecolors=color, linewidths=2.5, zorder=5)

        # Chance line
        axes[idx].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance (0.5)")
        axes[idx].set_ylim(0.3, 1.05)
        axes[idx].set_xlabel("Time (hpf)")
        axes[idx].set_ylabel("AUROC")
        axes[idx].set_title(panel_title)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)

    fig.suptitle(f"{experiment_label} {mode.replace('_', ' ').title()} AUROC over Time",
                 fontsize=14, fontweight="bold", y=1.02)

    # Place legend outside the rightmost panel
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.subplots_adjust(right=0.8)
    fig.tight_layout()

    auroc_curve_path = classification_fig_dir / f"{experiment_label}_{mode}_aurocs.png"
    fig.savefig(auroc_curve_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {auroc_curve_path}")

    print(f"\nAUROC curves saved to: {classification_fig_dir}")


if __name__ == "__main__":
    main()
