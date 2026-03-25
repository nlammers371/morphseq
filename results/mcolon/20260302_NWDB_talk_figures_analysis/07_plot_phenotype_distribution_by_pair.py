"""
NWDB talk: phenotype distribution figures by breeding pair (homozygous only).

Generates 4 figures:
  01 - curvature by pair, colored homozygous (crimson)
  02 - curvature by pair, colored by experiment_id (repeatability)
  03 - curvature by pair, colored by phenotype (cluster_categories)
  04 - phenotype proportions per pair (stacked bar)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PHENOTYPE_COLORS = {
    "High_to_Low": "#E76FA2",    # pink/rose
    "Low_to_High": "#2FB7B0",    # teal
    "Not Penetrant": "#3A3A3A",  # charcoal
}
AXIS_WIDTH_PER_COL = 500
AXIS_HEIGHT_PER_ROW = 450
PANEL_BORDER_COLOR = "black"
PANEL_BORDER_WIDTH = 1.0
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 1.8


def _apply_black_panel_borders(fig: plt.Figure) -> None:
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(PANEL_BORDER_COLOR)
            spine.set_linewidth(PANEL_BORDER_WIDTH)


def main() -> None:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))

    from analyze.utils.stats import normalize_arbitrary_feature
    from analyze.viz.plotting.feature_over_time import plot_feature_over_time
    from analyze.viz.plotting.proportions import plot_proportions
    from analyze.viz.styling.color_mapping_config import GENOTYPE_SUFFIX_COLORS
    from _plot_nwdb_genotype_classification_utils import save_figure

    out_dir = here / "figures" / "phenotype_distribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    CEP290_REF_DIR = _PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data"
    df_ref_data = pd.read_csv(CEP290_REF_DIR / "embryo_data_with_labels.csv", low_memory=False)
    labels_valid = pd.read_csv(CEP290_REF_DIR / "embryo_cluster_labels.csv", low_memory=False)
    labels_valid = labels_valid.drop_duplicates(subset="embryo_id")
    labels_valid = labels_valid[labels_valid["cluster_categories"].notna()].copy()

    df_ref = df_ref_data[df_ref_data["embryo_id"].isin(labels_valid["embryo_id"])].copy()

    df_ref["curvature"] = normalize_arbitrary_feature(
        df_ref["baseline_deviation_normalized"], low=0, high_percentile=100, clip=False
    )
    df_ref = df_ref.loc[df_ref["predicted_stage_hpf"] >= 24].copy()
    df_ref = df_ref.loc[df_ref["predicted_stage_hpf"] <= 120].copy()

    col = "cluster_categories"
    df_ref.loc[df_ref[col] == "Intermediate", col] = "Low_to_High"
    df_ref = df_ref[~df_ref["genotype"].isin(["cep290_unknown"])].copy()

    # Homozygous + pairs 1-3 + only the two specified experiments
    KEEP_EXPERIMENTS = ["20251106", "20251113"]
    df_homo = df_ref[df_ref["genotype"] == "cep290_homozygous"].copy()
    df_homo = df_homo[df_homo["pair"].str.contains(r"cep290_pair_[123]", regex=True, na=False)].copy()

    # Determine experiment date column and filter
    exp_col = "experiment_date" if "experiment_date" in df_homo.columns else "experiment_id"
    df_homo = df_homo[
        df_homo[exp_col].astype(str).str.replace("-", "").str[:8].isin(KEEP_EXPERIMENTS)
    ].copy()

    df_homo["pair_label"] = (
        df_homo["pair"].str.extract(r"pair_([123])")[0].map(lambda x: f"Pair {x}")
    )

    print(f"Loaded {len(df_homo)} rows (homozygous, pairs 1-3, expts {KEEP_EXPERIMENTS})")
    print(f"  Pairs: {sorted(df_homo['pair_label'].unique())}")
    print(f"  Experiments: {sorted(df_homo[exp_col].unique())}")
    print(f"  Phenotypes: {sorted(df_homo['cluster_categories'].dropna().unique())}")

    homo_color = str(GENOTYPE_SUFFIX_COLORS.get("homozygous", "#B2182B"))

    # -------------------------------------------------------------------------
    # Figure 1: curvature by pair, all crimson (homozygous)
    # -------------------------------------------------------------------------
    print("Generating Figure 1: curvature by pair (homozygous color)...")
    fig1 = plot_feature_over_time(
        df_homo,
        features="curvature",
        facet_col="pair_label",
        color_by="genotype",
        color_lookup={"cep290_homozygous": homo_color},
        show_individual=True,
        show_trend=True,
        backend="matplotlib",
        ylim=(0, 1),
        legend_loc="outside",
    )
    _apply_black_panel_borders(fig1)
    save_figure(
        fig1,
        out_dir / "01_curvature_by_pair_homozygous.png",
        out_dir / "01_curvature_by_pair_homozygous.pdf",
    )
    print("  Saved: 01_curvature_by_pair_homozygous.png/.pdf")

    # -------------------------------------------------------------------------
    # Figure 2: curvature by pair, colored by experiment_id (repeatability)
    # -------------------------------------------------------------------------
    print("Generating Figure 2: curvature by pair (colored by experiment_id)...")

    print(f"  Using experiment column: {exp_col}")

    fig2 = plot_feature_over_time(
        df_homo,
        features="curvature",
        facet_col="pair_label",
        color_by=exp_col,
        color_lookup=None,
        show_individual=True,
        show_trend=True,
        backend="matplotlib",
        ylim=(0, 1),
        legend_loc="outside",
    )
    _apply_black_panel_borders(fig2)
    save_figure(
        fig2,
        out_dir / "02_curvature_by_pair_colored_by_experiment.png",
        out_dir / "02_curvature_by_pair_colored_by_experiment.pdf",
    )
    print("  Saved: 02_curvature_by_pair_colored_by_experiment.png/.pdf")

    # -------------------------------------------------------------------------
    # Figure 3: curvature by pair, colored by phenotype (cluster_categories)
    # -------------------------------------------------------------------------
    print("Generating Figure 3: curvature by pair (colored by phenotype)...")
    fig3 = plot_feature_over_time(
        df_homo,
        features="curvature",
        facet_col="pair_label",
        color_by="cluster_categories",
        color_lookup=PHENOTYPE_COLORS,
        show_individual=True,
        show_trend=True,
        backend="matplotlib",
        ylim=(0, 1),
        legend_loc="outside",
    )
    _apply_black_panel_borders(fig3)
    save_figure(
        fig3,
        out_dir / "03_curvature_by_pair_colored_by_phenotype.png",
        out_dir / "03_curvature_by_pair_colored_by_phenotype.pdf",
    )
    print("  Saved: 03_curvature_by_pair_colored_by_phenotype.png/.pdf")

    # -------------------------------------------------------------------------
    # Figure 4: phenotype proportions per pair (stacked bar)
    # -------------------------------------------------------------------------
    print("Generating Figure 4: phenotype proportions per pair...")

    # One row per embryo (deduplicated on embryo_id)
    df_homo_embryo_level = (
        df_homo[["embryo_id", "pair_label", "cluster_categories"]]
        .drop_duplicates(subset="embryo_id")
        .copy()
    )
    print(f"  Embryo-level rows: {len(df_homo_embryo_level)}")

    fig4 = plot_proportions(
        df_homo_embryo_level,
        color_by_grouping="cluster_categories",
        col_by="pair_label",
        count_by="embryo_id",
        normalize=True,
        color_palette=PHENOTYPE_COLORS,
        show_counts=True,
        width_per_col=AXIS_WIDTH_PER_COL,
        height_per_row=AXIS_HEIGHT_PER_ROW,
        show_panel_border=True,
        panel_border_color=PANEL_BORDER_COLOR,
        panel_border_width=PANEL_BORDER_WIDTH,
        bar_edgecolor=BAR_EDGE_COLOR,
        bar_edgewidth=BAR_EDGE_WIDTH,
    )
    save_figure(
        fig4,
        out_dir / "04_phenotype_proportions_by_pair.png",
        out_dir / "04_phenotype_proportions_by_pair.pdf",
    )
    print("  Saved: 04_phenotype_proportions_by_pair.png/.pdf")

    print("\nDone. All figures saved to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
