"""
All crispants vs wik-ab control — 20260304 + 20260306 haspbx analysis.

Loads both 20260304 and 20260306 experiments (pbx crispant data),
normalizes control labels, and runs classification comparing ALL crispants
(including injection control) against wik-ab (non-injected wildtype) as the true null.

Outputs:
- Feature-over-time plots (length + baseline_deviation), faceted by genotype
- Feature-over-time plot faceted by experiment_id (batch-effect check)
- Multimetric faceted plots with rows = curvature,length and cols = experiment_id
- Multimetric stacked plot (rows = curvature,length, no column facet)
- Raw genotype proportion plot (embryo-level)
- Raw proportion tables and summary text with embryo counts
- Classification comparisons:
  - one-vs-all (each genotype vs rest)
  - all crispants vs wik-ab (the true null)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import BUILD06_DIR, EXPERIMENT_IDS, EXPERIMENT_LABEL, REPO_ROOT, resolve_bin_width_roots
OVERLAP_FEATURE = "baseline_deviation_normalized"
FEATURES = ["total_length_um", OVERLAP_FEATURE]
MULTIMETRIC_FACET_FEATURES = [OVERLAP_FEATURE, "total_length_um"]
THREE_FEATURE_PLOT = ["baseline_deviation_normalized", "total_length_um", "surface_area_um"]


def _normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower()
    g = g.replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")

    # Normalize injection control variants to a single label
    if g in (
        "ab_inj_ctrl",
        "wik-ab_inj_ctrl",
        "wik-ab_ctrl_inj",
        "wik_ab_inj_ctrl",
        "wik_ab_ctrl_inj",
    ):
        return "inj_ctrl"

    # Normalize uninjected wik-ab -> "wik_ab"
    g = g.replace("wik-ab", "wik_ab")

    return g


def _min_or_nan(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.min())


def _summarize_classification_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_set",
                "comparison_id",
                "positive",
                "negative",
                "max_auroc",
                "min_pval",
                "peak_time_bin_center",
                "n_time_bins",
            ]
        )

    group_cols = ["feature_set", "comparison_id", "positive_label", "negative_label"]
    summary = (
        scores.groupby(group_cols, as_index=False, observed=True)
        .agg(
            max_auroc=("auroc_obs", "max"),
            min_pval=("pval", _min_or_nan),
            n_time_bins=("time_bin_center", "nunique"),
        )
    )

    peak_rows = scores.loc[
        scores.groupby(group_cols, observed=True)["auroc_obs"].idxmax()
    ][group_cols + ["time_bin_center"]].rename(
        columns={"time_bin_center": "peak_time_bin_center"}
    )

    summary = summary.merge(peak_rows, on=group_cols, how="left")
    summary["feature"] = summary["feature_set"]
    summary["positive"] = summary["positive_label"]
    summary["negative"] = summary["negative_label"]
    return summary[
        [
            "feature",
            "feature_set",
            "comparison_id",
            "positive",
            "negative",
            "max_auroc",
            "min_pval",
            "peak_time_bin_center",
            "n_time_bins",
        ]
    ]


def _outside_legend_style():
    from analyze.viz.plotting.faceting_engine import default_style

    style = default_style()
    style.legend_loc = "outside"
    return style


def _write_classification_outputs(
    *,
    analysis,
    mode_stem: str,
    experiment_label: str,
    classification_dir: Path,
    classification_fig_dir: Path,
    color_lookup: dict[str, str],
    title: str,
) -> pd.DataFrame:
    scores = analysis.scores.copy()
    scores["positive"] = scores["positive_label"]
    scores["negative"] = scores["negative_label"]

    summary_all = _summarize_classification_scores(scores).sort_values(
        ["min_pval", "max_auroc"], ascending=[True, False], na_position="last"
    )
    summary_all.to_csv(
        classification_dir / f"{experiment_label}_{mode_stem}_summary_all_features.csv",
        index=False,
    )

    for feature_set in analysis.feature_sets:
        feat_scores = scores[scores["feature_set"] == feature_set].copy()
        feat_scores.to_csv(
            classification_dir / f"{experiment_label}_{mode_stem}_{feature_set}_comparisons.csv",
            index=False,
        )

        feat_summary = summary_all[summary_all["feature_set"] == feature_set].copy()
        feat_summary.to_csv(
            classification_dir / f"{experiment_label}_{mode_stem}_{feature_set}_summary.csv",
            index=False,
        )

    fig = analysis.plot_aurocs(
        curve_col="positive_label",
        facet_col="feature_set",
        color_lookup=color_lookup,
        show_null_band=True,
        show_significance=True,
        sig_threshold=0.05,
        backend="both",
        title=title,
        style=_outside_legend_style(),
    )
    fig["plotly"].write_html(
        classification_fig_dir / f"{experiment_label}_{mode_stem}_aurocs.html"
    )
    fig["matplotlib"].savefig(
        classification_fig_dir / f"{experiment_label}_{mode_stem}_aurocs.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig["matplotlib"])

    return summary_all


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PBX no-yolk rerun: all crispants vs wik_ab summary.")
    parser.add_argument("--bin-width", type=float, default=2.0, help="Classification time bin width in hpf.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of workers for classification.")
    parser.add_argument("--n-permutations", type=int, default=100, help="Number of label permutations.")
    parser.add_argument("--results-subdir", default=None, help="Relative results subdir under the PBX analysis root.")
    parser.add_argument("--figures-subdir", default=None, help="Relative figures subdir under the PBX analysis root.")
    args = parser.parse_args()

    results_dir, figures_dir = resolve_bin_width_roots(
        bin_width=args.bin_width,
        results_subdir=args.results_subdir,
        figures_subdir=args.figures_subdir,
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    project_root = REPO_ROOT
    sys.path.insert(0, str(project_root / "src"))

    from analyze.classification import run_classification
    from analyze.viz.styling import build_genotype_color_lookup
    from analyze.viz.plotting import plot_feature_over_time, plot_proportions
    from analyze.viz.plotting.faceting_engine import FacetSpec

    frames = []
    for exp_id in EXPERIMENT_IDS:
        data_path = BUILD06_DIR / f"df03_final_output_with_latents_{exp_id}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        part = pd.read_csv(data_path, low_memory=False)
        if "experiment_id" in part.columns:
            part = part[part["experiment_id"].astype(str) == exp_id].copy()
        else:
            part["experiment_id"] = exp_id
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)

    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()

    required_cols = {"embryo_id", "genotype", "predicted_stage_hpf", *FEATURES}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)

    embryo_df = df.drop_duplicates(subset="embryo_id")[
        ["embryo_id", "genotype", "experiment_id"]
    ].copy()
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

    # Identify wik_ab control (the true null)
    wik_ab_candidates = [g for g in genotype_order if "wik_ab" in g.lower()]
    wik_ab_genotype = wik_ab_candidates[0] if wik_ab_candidates else None

    # All non-wik_ab genotypes are positives (all crispants, including inj_ctrl)
    all_crispants = [
        g for g in genotype_order
        if "wik_ab" not in g.lower()
    ]

    color_lookup = build_genotype_color_lookup(genotype_order, warn_on_collision=False)

    # Faceted by genotype (primary view)
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
        figures_dir / f"{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_by_genotype.html"
    )
    overlap_figs["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_by_genotype.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(overlap_figs["matplotlib"])

    # Batch-effect check: facet by experiment_id, color by genotype
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

    # Three-feature plot: curvature, length, surface_area
    three_feature_plot = plot_feature_over_time(
        df,
        features=THREE_FEATURE_PLOT,
        id_col="embryo_id",
        color_by="genotype",
        color_lookup=color_lookup,
        show_individual=True,
        show_error_band=True,
        trend_statistic="median",
        backend="both",
        legend_loc="outside",
        title=f"{EXPERIMENT_LABEL} Curvature, Length, and Surface Area",
    )
    three_feature_plot["plotly"].write_html(
        figures_dir / f"{EXPERIMENT_LABEL}_three_features_stacked.html"
    )
    three_feature_plot["matplotlib"].savefig(
        figures_dir / f"{EXPERIMENT_LABEL}_three_features_stacked.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(three_feature_plot["matplotlib"])

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

    # -------------------------------------------------------------------------
    # Classification comparisons
    # -------------------------------------------------------------------------
    classification_dir = results_dir / "classification"
    classification_fig_dir = figures_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)
    classification_fig_dir.mkdir(parents=True, exist_ok=True)

    class_feature_sets = {
        "curvature": ["baseline_deviation_normalized"],
        "length": ["total_length_um"],
        "embedding": "z_mu_b",
    }
    class_colors = {gt: color_lookup.get(gt, "#808080") for gt in genotype_order}
    n_permutations = args.n_permutations
    bin_width = args.bin_width
    min_samples_per_class = 3

    # Mode 1: one-vs-all
    ovr_analysis = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        comparisons="all_vs_rest",
        features=class_feature_sets,
        n_jobs=args.n_jobs,
        n_permutations=n_permutations,
        bin_width=bin_width,
        min_samples_per_group=min_samples_per_class,
        verbose=True,
    )
    ovr_summary = _write_classification_outputs(
        analysis=ovr_analysis,
        mode_stem="one_vs_all",
        experiment_label=EXPERIMENT_LABEL,
        classification_dir=classification_dir,
        classification_fig_dir=classification_fig_dir,
        color_lookup=class_colors,
        title=f"{EXPERIMENT_LABEL} One-vs-All AUROC",
    )

    # Mode 2: all crispants vs wik-ab (the true null)
    vs_wik_ab_summary = pd.DataFrame()
    if wik_ab_genotype is not None and all_crispants:
        vs_wik_ab_analysis = run_classification(
            df,
            class_col="genotype",
            id_col="embryo_id",
            time_col="predicted_stage_hpf",
            positive=all_crispants,
            negative=wik_ab_genotype,
            features=class_feature_sets,
            n_jobs=args.n_jobs,
            n_permutations=n_permutations,
            bin_width=bin_width,
            min_samples_per_group=min_samples_per_class,
            verbose=True,
        )
        vs_wik_ab_summary = _write_classification_outputs(
            analysis=vs_wik_ab_analysis,
            mode_stem="all_crispants_vs_wik_ab",
            experiment_label=EXPERIMENT_LABEL,
            classification_dir=classification_dir,
            classification_fig_dir=classification_fig_dir,
            color_lookup=class_colors,
            title=f"{EXPERIMENT_LABEL} All Crispants vs wik-ab Control",
        )

    summary_lines = [
        f"experiment_ids: {EXPERIMENT_IDS}",
        f"experiment_label: {EXPERIMENT_LABEL}",
        f"total_rows: {len(df)}",
        f"total_embryos: {total_embryos}",
        "",
        "classification_control: wik_ab (non-injected wildtype)",
        "",
        "genotype_raw_proportions:",
    ]
    for _, row in genotype_counts.iterrows():
        summary_lines.append(
            f"- {row['genotype']}: n={int(row['n_embryos'])}, "
            f"proportion={row['proportion']:.6f}, percent={row['percent']:.2f}"
        )
    summary_lines += ["", "classification_top_hits_one_vs_all:"]
    if not ovr_summary.empty:
        for _, row in ovr_summary.head(5).iterrows():
            summary_lines.append(
                f"- [{row['feature']}] {row['positive']} vs {row['negative']}: "
                f"max_auroc={float(row['max_auroc']):.3f}, "
                f"min_pval={float(row['min_pval']):.4f}"
            )
    else:
        summary_lines.append("- none")
    summary_lines += ["", "classification_top_hits_all_crispants_vs_wik_ab:"]
    if not vs_wik_ab_summary.empty:
        for _, row in vs_wik_ab_summary.head(5).iterrows():
            summary_lines.append(
                f"- [{row['feature']}] {row['positive']} vs {row['negative']}: "
                f"max_auroc={float(row['max_auroc']):.3f}, "
                f"min_pval={float(row['min_pval']):.4f}"
            )
    else:
        summary_lines.append("- none")

    summary_text = "\n".join(summary_lines)
    (results_dir / f"summary_{EXPERIMENT_LABEL}.txt").write_text(summary_text + "\n")

    print(summary_text)
    print("\nSaved outputs:")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_length_curvature_by_genotype.html'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_length_curvature_by_genotype.png'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_by_genotype.html'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_by_genotype.png'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_batch_check.html'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_{OVERLAP_FEATURE}_batch_check.png'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_curvature_length_by_experiment.html'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_curvature_length_by_experiment.png'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_curvature_length_stacked.html'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_curvature_length_stacked.png'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_three_features_stacked.html'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_three_features_stacked.png'}")
    print(f"- {figures_dir / f'{EXPERIMENT_LABEL}_raw_genotype_proportions.png'}")
    print(f"- {results_dir / f'raw_genotype_proportions_{EXPERIMENT_LABEL}.csv'}")
    print(f"- {classification_dir / f'{EXPERIMENT_LABEL}_one_vs_all_summary_all_features.csv'}")
    print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_one_vs_all_aurocs.html'}")
    print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_one_vs_all_aurocs.png'}")
    if not vs_wik_ab_summary.empty:
        print(f"- {classification_dir / f'{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_summary_all_features.csv'}")
        print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_aurocs.html'}")
        print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_aurocs.png'}")
    print(f"- {results_dir / f'summary_{EXPERIMENT_LABEL}.txt'}")


if __name__ == "__main__":
    main()
