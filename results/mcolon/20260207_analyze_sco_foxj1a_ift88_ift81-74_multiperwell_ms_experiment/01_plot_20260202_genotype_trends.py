"""
Quick genotype trend plots for experiment 20260202.

Outputs:
- Feature-over-time plot (length + curvature), faceted by genotype
- Raw genotype proportion plot (embryo-level)
- Raw proportion tables and summary text with embryo counts
- Classification comparisons:
  - one-vs-all (each genotype vs rest)
  - each-vs-ab (each non-ab genotype vs ab)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_ID = "20260202"
FEATURES = ["total_length_um", "baseline_deviation_normalized"]
OVERLAP_FEATURE = "baseline_deviation_normalized"


def main() -> None:
    run_dir = Path(__file__).resolve().parent
    figures_dir = run_dir / "figures"
    results_dir = run_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root / "src"))

    from analyze.trajectory_analysis.viz.styling import get_color_for_genotype
    from analyze.difference_detection import (
        run_classification_test,
        plot_feature_comparison_grid,
    )
    from analyze.viz.plotting import plot_feature_over_time, plot_proportions

    data_path = (
        project_root
        / "morphseq_playground"
        / "metadata"
        / "build06_output"
        / f"df03_final_output_with_latents_{EXPERIMENT_ID}.csv"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    if "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str) == EXPERIMENT_ID].copy()
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()

    required_cols = {"embryo_id", "genotype", "predicted_stage_hpf", *FEATURES}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").astype(str)

    embryo_df = df.drop_duplicates(subset="embryo_id")[["embryo_id", "genotype"]].copy()
    total_embryos = embryo_df["embryo_id"].nunique()

    genotype_counts = (
        embryo_df.groupby("genotype", observed=True)["embryo_id"]
        .nunique()
        .rename("n_embryos")
        .sort_values(ascending=False)
        .reset_index()
    )
    genotype_counts["proportion"] = genotype_counts["n_embryos"] / total_embryos
    genotype_counts["percent"] = 100.0 * genotype_counts["proportion"]
    genotype_counts.to_csv(
        results_dir / f"raw_genotype_proportions_{EXPERIMENT_ID}.csv",
        index=False,
    )

    embryo_df["genotype_class"] = embryo_df["genotype"].where(
        embryo_df["genotype"].str.lower().eq("ab"),
        "crispant",
    )
    class_counts = (
        embryo_df.groupby("genotype_class", observed=True)["embryo_id"]
        .nunique()
        .rename("n_embryos")
        .sort_values(ascending=False)
        .reset_index()
    )
    class_counts["proportion"] = class_counts["n_embryos"] / total_embryos
    class_counts["percent"] = 100.0 * class_counts["proportion"]
    class_counts.to_csv(
        results_dir / f"raw_ab_vs_crispant_proportions_{EXPERIMENT_ID}.csv",
        index=False,
    )

    genotype_order = genotype_counts["genotype"].tolist()
    color_lookup = {gt: get_color_for_genotype(gt) for gt in genotype_order}

    figs = plot_feature_over_time(
        df,
        features=FEATURES,
        color_by="genotype",
        color_lookup=color_lookup,
        facet_col="genotype",
        show_individual=True,
        show_error_band=True,
        trend_statistic="median",
        backend="both",
    )
    figs["plotly"].write_html(
        figures_dir / f"{EXPERIMENT_ID}_length_curvature_by_genotype.html"
    )
    figs["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_ID}_length_curvature_by_genotype.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figs["matplotlib"])

    # Overlapping view in a single panel: one feature colored by genotype.
    overlap_figs = plot_feature_over_time(
        df,
        features=OVERLAP_FEATURE,
        color_by="genotype",
        color_lookup=color_lookup,
        show_individual=True,
        show_error_band=True,
        trend_statistic="median",
        backend="both",
    )
    overlap_figs["plotly"].write_html(
        figures_dir / f"{EXPERIMENT_ID}_{OVERLAP_FEATURE}_overlap_by_genotype.html"
    )
    overlap_figs["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_ID}_{OVERLAP_FEATURE}_overlap_by_genotype.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(overlap_figs["matplotlib"])

    fig = plot_proportions(
        embryo_df,
        color_by_grouping="genotype",
        count_by="embryo_id",
        color_order=genotype_order,
        color_palette=color_lookup,
        normalize=True,
        bar_mode="grouped",
        title=f"Raw Genotype Proportions ({EXPERIMENT_ID})",
        show_counts=True,
    )
    fig.savefig(
        figures_dir / f"{EXPERIMENT_ID}_raw_genotype_proportions.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---------------------------------------------------------------------
    # Classification comparisons
    # ---------------------------------------------------------------------
    classification_dir = results_dir / "classification"
    classification_fig_dir = figures_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)
    classification_fig_dir.mkdir(parents=True, exist_ok=True)

    class_feature_sets = {
        "curvature": ["baseline_deviation_normalized"],
        "length": ["total_length_um"],
        "embedding": "z_mu_b",
    }
    class_feature_labels = {
        "curvature": "Curvature",
        "length": "Length",
        "embedding": "Embedding",
    }
    class_colors = {gt: color_lookup.get(gt, "#808080") for gt in genotype_order}
    non_ab_genotypes = [g for g in genotype_order if g.lower() != "ab"]
    n_permutations = 100
    bin_width = 2.0
    min_samples_per_class = 3

    # Mode 1: one-vs-all
    ovr_results_by_feature = {}
    ovr_summary_tables = []
    for feat_key, feat_spec in class_feature_sets.items():
        res = run_classification_test(
            df,
            groupby="genotype",
            groups="all",
            reference="rest",
            features=feat_spec,
            n_jobs=-1,
            n_permutations=n_permutations,
            bin_width=bin_width,
            min_samples_per_class=min_samples_per_class,
            verbose=False,
        )
        ovr_results_by_feature[feat_key] = res
        res.comparisons.to_csv(
            classification_dir / f"{EXPERIMENT_ID}_one_vs_all_{feat_key}_comparisons.csv",
            index=False,
        )
        feat_summary = res.summary().sort_values(
            ["min_pval", "max_auroc"], ascending=[True, False]
        )
        feat_summary["feature"] = feat_key
        ovr_summary_tables.append(feat_summary)
        feat_summary.to_csv(
            classification_dir / f"{EXPERIMENT_ID}_one_vs_all_{feat_key}_summary.csv",
            index=False,
        )

    ovr_summary = pd.concat(ovr_summary_tables, ignore_index=True)
    ovr_summary.to_csv(
        classification_dir / f"{EXPERIMENT_ID}_one_vs_all_summary_all_features.csv",
        index=False,
    )
    fig_ovr_grid = plot_feature_comparison_grid(
        results_by_feature=ovr_results_by_feature,
        feature_labels=class_feature_labels,
        cluster_colors=class_colors,
        title=f"{EXPERIMENT_ID} One-vs-All (Embedding vs Length vs Curvature)",
        save_path=classification_fig_dir / f"{EXPERIMENT_ID}_one_vs_all_feature_grid.png",
    )
    plt.close(fig_ovr_grid)

    # Mode 2: each non-ab genotype vs ab
    if "ab" in df["genotype"].str.lower().unique():
        vs_ab_results_by_feature = {}
        vs_ab_summary_tables = []
        for feat_key, feat_spec in class_feature_sets.items():
            res = run_classification_test(
                df,
                groupby="genotype",
                groups=non_ab_genotypes,
                reference="ab",
                features=feat_spec,
                n_jobs=-1,
                n_permutations=n_permutations,
                bin_width=bin_width,
                min_samples_per_class=min_samples_per_class,
                verbose=False,
            )
            vs_ab_results_by_feature[feat_key] = res
            res.comparisons.to_csv(
                classification_dir / f"{EXPERIMENT_ID}_each_vs_ab_{feat_key}_comparisons.csv",
                index=False,
            )
            feat_summary = res.summary().sort_values(
                ["min_pval", "max_auroc"], ascending=[True, False]
            )
            feat_summary["feature"] = feat_key
            vs_ab_summary_tables.append(feat_summary)
            feat_summary.to_csv(
                classification_dir / f"{EXPERIMENT_ID}_each_vs_ab_{feat_key}_summary.csv",
                index=False,
            )

        vs_ab_summary = pd.concat(vs_ab_summary_tables, ignore_index=True)
        vs_ab_summary.to_csv(
            classification_dir / f"{EXPERIMENT_ID}_each_vs_ab_summary_all_features.csv",
            index=False,
        )
        fig_vs_ab_grid = plot_feature_comparison_grid(
            results_by_feature=vs_ab_results_by_feature,
            feature_labels=class_feature_labels,
            cluster_colors=class_colors,
            title=f"{EXPERIMENT_ID} Each-vs-AB (Embedding vs Length vs Curvature)",
            save_path=classification_fig_dir / f"{EXPERIMENT_ID}_each_vs_ab_feature_grid.png",
        )
        plt.close(fig_vs_ab_grid)
    else:
        vs_ab_summary = pd.DataFrame()

    summary_lines = [
        f"experiment_id: {EXPERIMENT_ID}",
        f"total_rows: {len(df)}",
        f"total_embryo_ids: {total_embryos}",
        "",
        "genotype_raw_proportions:",
    ]
    for _, row in genotype_counts.iterrows():
        summary_lines.append(
            f"- {row['genotype']}: n={int(row['n_embryos'])}, "
            f"proportion={row['proportion']:.6f}, percent={row['percent']:.2f}"
        )
    summary_lines += ["", "ab_vs_crispant_raw_proportions:"]
    for _, row in class_counts.iterrows():
        summary_lines.append(
            f"- {row['genotype_class']}: n={int(row['n_embryos'])}, "
            f"proportion={row['proportion']:.6f}, percent={row['percent']:.2f}"
        )
    summary_lines += ["", "classification_top_hits_one_vs_all:"]
    if not ovr_summary.empty:
        for _, row in ovr_summary.head(5).iterrows():
            summary_lines.append(
                f"- [{row['feature']}] {row['positive']} vs {row['negative']}: "
                f"max_auroc={row.get('max_auroc', float('nan')):.3f}, "
                f"min_pval={row.get('min_pval', float('nan')):.4f}"
            )
    else:
        summary_lines.append("- none")
    summary_lines += ["", "classification_top_hits_each_vs_ab:"]
    if not vs_ab_summary.empty:
        for _, row in vs_ab_summary.head(5).iterrows():
            summary_lines.append(
                f"- [{row['feature']}] {row['positive']} vs {row['negative']}: "
                f"max_auroc={row.get('max_auroc', float('nan')):.3f}, "
                f"min_pval={row.get('min_pval', float('nan')):.4f}"
            )
    else:
        summary_lines.append("- none")

    summary_text = "\n".join(summary_lines)
    (results_dir / f"summary_{EXPERIMENT_ID}.txt").write_text(summary_text + "\n")

    print(summary_text)
    print("\nSaved outputs:")
    print(f"- {figures_dir / f'{EXPERIMENT_ID}_length_curvature_by_genotype.html'}")
    print(f"- {figures_dir / f'{EXPERIMENT_ID}_length_curvature_by_genotype.png'}")
    print(f"- {figures_dir / f'{EXPERIMENT_ID}_{OVERLAP_FEATURE}_overlap_by_genotype.html'}")
    print(f"- {figures_dir / f'{EXPERIMENT_ID}_{OVERLAP_FEATURE}_overlap_by_genotype.png'}")
    print(f"- {figures_dir / f'{EXPERIMENT_ID}_raw_genotype_proportions.png'}")
    print(f"- {results_dir / f'raw_genotype_proportions_{EXPERIMENT_ID}.csv'}")
    print(f"- {results_dir / f'raw_ab_vs_crispant_proportions_{EXPERIMENT_ID}.csv'}")
    print(f"- {classification_dir / f'{EXPERIMENT_ID}_one_vs_all_summary_all_features.csv'}")
    print(f"- {classification_fig_dir / f'{EXPERIMENT_ID}_one_vs_all_feature_grid.png'}")
    if not vs_ab_summary.empty:
        print(f"- {classification_dir / f'{EXPERIMENT_ID}_each_vs_ab_summary_all_features.csv'}")
        print(f"- {classification_fig_dir / f'{EXPERIMENT_ID}_each_vs_ab_feature_grid.png'}")
    print(f"- {results_dir / f'summary_{EXPERIMENT_ID}.txt'}")


if __name__ == "__main__":
    main()
