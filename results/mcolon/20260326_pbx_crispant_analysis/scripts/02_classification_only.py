"""
Classification analysis only — 20260304 + 20260306 haspbx data.

Runs classification on pre-loaded data comparing all crispants vs wik-ab.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


EXPERIMENT_IDS = ["20260304", "20260306"]
EXPERIMENT_LABEL = "20260304_20260306"


def plot_auroc_heatmap(ax, comparisons_df, title, color_lookup=None, sig_threshold=0.01):
    """
    Plot AUROC heatmap with genotype x time_bin, bordered significance cells.
    """
    positive_col = "positive"
    if positive_col not in comparisons_df.columns:
        positive_col = "positive_label"

    pivot = comparisons_df.pivot_table(
        index=positive_col, columns="time_bin_center", values="auroc_obs"
    )
    sig = comparisons_df.pivot_table(
        index=positive_col, columns="time_bin_center", values="pval"
    ) <= sig_threshold

    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    desired_order = [
        "inj_ctrl",
        "wik_ab",
        "pbx1b_crispant",
        "pbx4_crispant",
        "pbx1b_pbx4_crispant",
    ]
    row_order = [g for g in desired_order if g in pivot.index]
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
        sig_threshold=0.01,
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
    run_dir = Path(__file__).resolve().parent

    # Get bin_width from argument or use default
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-width", type=float, default=4.0, help="Time bin width in hpf")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of workers for classification.")
    parser.add_argument("--n-permutations", type=int, default=500, help="Number of label permutations.")
    parser.add_argument(
        "--mode",
        choices=["all", "vs_wik_ab", "vs_inj_ctrl"],
        default="all",
        help="Which classification block to run.",
    )
    args = parser.parse_args()
    bin_width_arg = args.bin_width
    run_mode = args.mode
    n_jobs_arg = args.n_jobs
    n_permutations_arg = args.n_permutations

    figures_dir = run_dir.parent / "figures" / f"bin_width_{bin_width_arg:.1f}hpf"
    results_dir = run_dir.parent / "results" / f"bin_width_{bin_width_arg:.1f}hpf"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root / "src"))

    from analyze.classification import run_classification
    from analyze.viz.styling import build_genotype_color_lookup

    frames = []
    for exp_id in EXPERIMENT_IDS:
        data_path = (
            project_root
            / "morphseq_playground"
            / "metadata"
            / "build06_output"
            / f"df03_final_output_with_latents_{exp_id}.csv"
        )
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

    required_cols = {"embryo_id", "genotype", "predicted_stage_hpf", "baseline_deviation_normalized", "total_length_um"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)

    embryo_df = df.drop_duplicates(subset="embryo_id")[
        ["embryo_id", "genotype"]
    ].copy()

    genotype_counts = (
        embryo_df.groupby("genotype", observed=True)["embryo_id"]
        .nunique()
        .rename("n_embryos")
        .sort_values(ascending=False)
        .reset_index()
    )
    total_embryos = genotype_counts["n_embryos"].sum()
    genotype_counts["proportion"] = genotype_counts["n_embryos"] / total_embryos
    genotype_counts["percent"] = 100.0 * genotype_counts["proportion"]

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
    n_permutations = n_permutations_arg
    bin_width = bin_width_arg
    min_samples_per_class = 3
    n_jobs = n_jobs_arg

    print(f"Running classification with bin_width={bin_width}hpf")
    print(f"n_jobs={n_jobs}, n_permutations={n_permutations}")
    print(f"Data summary: {len(df)} rows, {df['embryo_id'].nunique()} embryos")
    print(f"Genotypes: {genotype_order}")

    # All crispants vs wik-ab (the true null)
    vs_wik_ab_summary = pd.DataFrame()
    if run_mode in {"all", "vs_wik_ab"} and wik_ab_genotype is not None and all_crispants:
        print(f"\n=== Running all crispants vs wik-ab classification ===")
        print(f"Positive (crispants): {all_crispants}")
        print(f"Negative (control): {wik_ab_genotype}")
        vs_wik_ab_analysis = run_classification(
            df,
            class_col="genotype",
            id_col="embryo_id",
            time_col="predicted_stage_hpf",
            positive=all_crispants,
            negative=wik_ab_genotype,
            features=class_feature_sets,
            n_jobs=n_jobs,
            n_permutations=n_permutations,
            bin_width=bin_width,
            min_samples_per_group=min_samples_per_class,
            verbose=True,
        )
        print("All crispants vs wik-ab completed.")
        vs_wik_ab_summary = _write_classification_outputs(
            analysis=vs_wik_ab_analysis,
            mode_stem="all_crispants_vs_wik_ab",
            experiment_label=EXPERIMENT_LABEL,
            classification_dir=classification_dir,
            classification_fig_dir=classification_fig_dir,
            color_lookup=class_colors,
            title=f"{EXPERIMENT_LABEL} All Crispants vs wik-ab Control",
        )

    # All non-inj_ctrl genotypes vs inj_ctrl (replication check)
    inj_ctrl_genotype = next((g for g in genotype_order if g == "inj_ctrl"), None)
    vs_inj_ctrl_positives = [g for g in genotype_order if g != "inj_ctrl"]
    vs_inj_ctrl_summary = pd.DataFrame()
    if run_mode in {"all", "vs_inj_ctrl"} and inj_ctrl_genotype is not None and vs_inj_ctrl_positives:
        print(f"\n=== Running all genotypes vs inj_ctrl classification ===")
        print(f"Positive (all non-inj_ctrl genotypes): {vs_inj_ctrl_positives}")
        print(f"Negative (control): {inj_ctrl_genotype}")
        vs_inj_ctrl_analysis = run_classification(
            df,
            class_col="genotype",
            id_col="embryo_id",
            time_col="predicted_stage_hpf",
            positive=vs_inj_ctrl_positives,
            negative=inj_ctrl_genotype,
            features=class_feature_sets,
            n_jobs=n_jobs,
            n_permutations=n_permutations,
            bin_width=bin_width,
            min_samples_per_group=min_samples_per_class,
            verbose=True,
        )
        print("All genotypes vs inj_ctrl completed.")
        vs_inj_ctrl_summary = _write_classification_outputs(
            analysis=vs_inj_ctrl_analysis,
            mode_stem="all_genotypes_vs_inj_ctrl",
            experiment_label=EXPERIMENT_LABEL,
            classification_dir=classification_dir,
            classification_fig_dir=classification_fig_dir,
            color_lookup=class_colors,
            title=f"{EXPERIMENT_LABEL} All Genotypes vs inj_ctrl",
        )

    summary_lines = [
        f"experiment_ids: {EXPERIMENT_IDS}",
        f"experiment_label: {EXPERIMENT_LABEL}",
        f"total_rows: {len(df)}",
        f"total_embryos: {df['embryo_id'].nunique()}",
        "",
        "classification_control: wik_ab (non-injected wildtype)",
        "comparison: all crispants vs wik_ab",
        "",
        "genotype_raw_proportions:",
    ]
    for _, row in genotype_counts.iterrows():
        summary_lines.append(
            f"- {row['genotype']}: n={int(row['n_embryos'])}, "
            f"proportion={row['proportion']:.6f}, percent={row['percent']:.2f}"
        )
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

    summary_lines += ["", "classification_top_hits_all_genotypes_vs_inj_ctrl:"]
    if not vs_inj_ctrl_summary.empty:
        for _, row in vs_inj_ctrl_summary.head(5).iterrows():
            summary_lines.append(
                f"- [{row['feature']}] {row['positive']} vs {row['negative']}: "
                f"max_auroc={float(row['max_auroc']):.3f}, "
                f"min_pval={float(row['min_pval']):.4f}"
            )
    else:
        summary_lines.append("- none")

    summary_text = "\n".join(summary_lines)
    (results_dir / f"summary_classification_{EXPERIMENT_LABEL}.txt").write_text(summary_text + "\n")

    print(summary_text)

    # -------------------------------------------------------------------------
    # Plot AUROC heatmaps (all crispants vs wik-ab)
    if run_mode in {"all", "vs_wik_ab"} and not vs_wik_ab_summary.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        vs_wik_ab_scores = vs_wik_ab_analysis.scores.copy()
        for ax, feature_set in zip(axes, ["curvature", "length", "embedding"]):
            feat_data = vs_wik_ab_scores[vs_wik_ab_scores["feature_set"] == feature_set].copy()
            if not feat_data.empty:
                plot_auroc_heatmap(
                    ax,
                    feat_data,
                    f"All Crispants vs wik-ab: {feature_set}",
                    sig_threshold=0.01
                )
        fig.tight_layout()
        fig.savefig(
            classification_fig_dir / f"{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_heatmaps.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"Saved: {classification_fig_dir / f'{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_heatmaps.png'}")

    print("\nSaved classification outputs:")
    if not vs_wik_ab_summary.empty:
        print(f"- {classification_dir / f'{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_summary_all_features.csv'}")
        print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_aurocs.html'}")
        print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_aurocs.png'}")
        if run_mode in {'all', 'vs_wik_ab'}:
            print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_all_crispants_vs_wik_ab_heatmaps.png'}")
    if not vs_inj_ctrl_summary.empty:
        print(f"- {classification_dir / f'{EXPERIMENT_LABEL}_all_genotypes_vs_inj_ctrl_summary_all_features.csv'}")
        print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_all_genotypes_vs_inj_ctrl_aurocs.html'}")
        print(f"- {classification_fig_dir / f'{EXPERIMENT_LABEL}_all_genotypes_vs_inj_ctrl_aurocs.png'}")
    print(f"- {results_dir / f'summary_classification_{EXPERIMENT_LABEL}.txt'}")


if __name__ == "__main__":
    main()
