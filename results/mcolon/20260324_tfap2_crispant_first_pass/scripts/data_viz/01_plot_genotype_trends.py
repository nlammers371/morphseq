"""TFAP2 crispant genotype trend plots from the aggregated combined dataset."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    run_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = run_dir / "figures"
    results_dir = run_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(run_dir))
    sys.path.insert(0, str(project_root / "src"))

    from analyze.viz.styling import build_genotype_color_lookup
    from analyze.viz.plotting import plot_feature_over_time, plot_proportions
    from analyze.viz.plotting.faceting_engine import FacetSpec
    from scripts.common import (
        EXPERIMENT_IDS,
        EXPERIMENT_LABEL,
        FEATURES,
        MULTIMETRIC_FACET_FEATURES,
        OVERLAP_FEATURE,
        load_aggregate_dataframe,
    )

    df = load_aggregate_dataframe(
        run_dir,
        required_cols={"embryo_id", "genotype", "experiment_id", "predicted_stage_hpf", *FEATURES},
    )

    embryo_df = df.drop_duplicates(subset="embryo_id")[["embryo_id", "genotype", "experiment_id"]].copy()
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
        results_dir / f"raw_genotype_proportions_{EXPERIMENT_LABEL}.csv",
        index=False,
    )

    genotype_order = genotype_counts["genotype"].tolist()
    control_genotypes = {"inj_ctrl", "non_inj_ctrl"}

    embryo_df["genotype_class"] = embryo_df["genotype"].where(
        embryo_df["genotype"].isin(control_genotypes),
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
        results_dir / f"raw_control_vs_crispant_proportions_{EXPERIMENT_LABEL}.csv",
        index=False,
    )

    color_lookup = build_genotype_color_lookup(genotype_order, warn_on_collision=False)

    print("Plotting features by genotype...")
    figs = plot_feature_over_time(
        df,
        features=FEATURES,
        id_col="embryo_id",
        color_by="genotype",
        color_lookup=color_lookup,
        facet_col="genotype",
        show_individual=True,
        show_error_band=True,
        trend_statistic="median",
        backend="both",
        legend_loc="outside",
    )
    figs["plotly"].write_html(
        figures_dir / f"{EXPERIMENT_LABEL}_length_curvature_by_genotype.html"
    )
    figs["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_LABEL}_length_curvature_by_genotype.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figs["matplotlib"])

    overlap_figs = plot_feature_over_time(
        df,
        features=OVERLAP_FEATURE,
        id_col="embryo_id",
        color_by="genotype",
        color_lookup=color_lookup,
        show_individual=True,
        show_error_band=True,
        trend_statistic="median",
        backend="both",
        legend_loc="outside",
    )
    overlap_figs["plotly"].write_html(
        figures_dir / f"{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_overlap_by_genotype.html"
    )
    overlap_figs["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_overlap_by_genotype.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(overlap_figs["matplotlib"])

    print("Plotting batch effect check...")
    batch_figs = plot_feature_over_time(
        df,
        features=OVERLAP_FEATURE,
        id_col="embryo_id",
        color_by="genotype",
        color_lookup=color_lookup,
        facet_col="experiment_id",
        show_individual=True,
        show_error_band=True,
        trend_statistic="median",
        backend="both",
        legend_loc="outside",
    )
    batch_figs["plotly"].write_html(
        figures_dir / f"{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_batch_check.html"
    )
    batch_figs["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_batch_check.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(batch_figs["matplotlib"])

    multimetric_layout = FacetSpec(col_order=EXPERIMENT_IDS)
    print("Plotting multimetric by experiment...")
    multimetric_by_experiment = plot_feature_over_time(
        df,
        features=MULTIMETRIC_FACET_FEATURES,
        id_col="embryo_id",
        color_by="genotype",
        color_lookup=color_lookup,
        facet_col="experiment_id",
        layout=multimetric_layout,
        show_individual=True,
        show_error_band=True,
        trend_statistic="median",
        backend="both",
        legend_loc="outside",
        title=f"{EXPERIMENT_LABEL} Curvature and Length by Experiment",
    )
    multimetric_by_experiment["plotly"].write_html(
        figures_dir / f"{EXPERIMENT_LABEL}_curvature_length_by_experiment.html"
    )
    multimetric_by_experiment["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_LABEL}_curvature_length_by_experiment.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(multimetric_by_experiment["matplotlib"])

    print("Plotting multimetric stacked...")
    multimetric_stacked = plot_feature_over_time(
        df,
        features=MULTIMETRIC_FACET_FEATURES,
        id_col="embryo_id",
        color_by="genotype",
        color_lookup=color_lookup,
        show_individual=True,
        show_error_band=True,
        trend_statistic="median",
        backend="both",
        legend_loc="outside",
        title=f"{EXPERIMENT_LABEL} Curvature and Length",
    )
    multimetric_stacked["plotly"].write_html(
        figures_dir / f"{EXPERIMENT_LABEL}_curvature_length_stacked.html"
    )
    multimetric_stacked["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_LABEL}_curvature_length_stacked.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(multimetric_stacked["matplotlib"])

    print("Plotting proportions...")
    fig = plot_proportions(
        embryo_df,
        color_by_grouping="genotype",
        count_by="embryo_id",
        color_order=genotype_order,
        color_palette=color_lookup,
        normalize=True,
        bar_mode="grouped",
        title=f"Raw Genotype Proportions ({EXPERIMENT_LABEL})",
        show_counts=True,
    )
    fig.savefig(
        figures_dir / f"{EXPERIMENT_LABEL}_raw_genotype_proportions.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nTrend plots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
