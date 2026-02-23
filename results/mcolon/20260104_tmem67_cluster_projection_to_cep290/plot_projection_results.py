"""
Plot TMEM67→CEP290 projection outputs.

This is a post-processing script that ingests the CSV outputs written by
`run_projection.py` and generates:
  1) Cluster frequency comparison bar plot
  2) Faceted TMEM67 trajectories (rows=genotype, cols=assigned group)
  3) Cluster composition grid (genotype/pair/experiment_id)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path (matches run_projection.py pattern)
import sys
sys.path.insert(0, "/net/trapnell/vol1/home/mdcolon/proj/morphseq")

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe
from src.analyze.trajectory_analysis.facetted_plotting import (
    plot_proportion_grid,
    plot_trajectories_faceted,
)
import matplotlib.colors as mcolors

# Define ordering (most important!)
PHENOTYPE_ORDER = ["High_to_Low", 'Intermediate', 'Low_to_High', 'Not Penetrant']

cmap = plt.cm.tab10
# Convert to hex strings for Plotly compatibility
PHENOTYPE_COLORS = {
    'Not Penetrant': mcolors.to_hex(cmap(0)),   # Blue
    'Low_to_High': mcolors.to_hex(cmap(1)),     # Orange  
    'Intermediate': mcolors.to_hex(cmap(2)),    # Green
    'High_to_Low': mcolors.to_hex(cmap(3)),     # Red
}
# Convert to ordered palette list
PHENOTYPE_PALETTE = [PHENOTYPE_COLORS[cat] for cat in PHENOTYPE_ORDER]

# Keep defaults aligned with run_projection.py
TMEM67_EXPERIMENTS = ["20250711", "20251205"]
TIME_COL = "predicted_stage_hpf"
METRIC_COL = "baseline_deviation_normalized"
PAIR_FILTER = "tmem67_spawn"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_csv(path, low_memory=False)


def plot_cluster_frequency_comparison(freq_csv: Path, out_png: Path, title: str) -> None:
    df = _read_csv(freq_csv).rename(columns={"Unnamed: 0": "cluster_category"})
    if "cluster_category" not in df.columns:
        df = df.reset_index().rename(columns={"index": "cluster_category"})

    value_cols = [c for c in df.columns if c != "cluster_category"]
    df_long = df.melt(id_vars="cluster_category", value_vars=value_cols, var_name="dataset", value_name="percent")

    # Enforce phenotype-first ordering; keep any unexpected categories at the end.
    extra = [c for c in sorted(df_long["cluster_category"].dropna().unique()) if c not in PHENOTYPE_ORDER]
    ordered_clusters = PHENOTYPE_ORDER + extra
    df_long["cluster_category"] = pd.Categorical(
        df_long["cluster_category"],
        categories=ordered_clusters,
        ordered=True,
    )

    # Plot: bars colored by phenotype; hatch distinguishes datasets.
    pivot = (
        df_long
        .pivot_table(index="cluster_category", columns="dataset", values="percent", aggfunc="mean", fill_value=0)
        .reindex(ordered_clusters)
    )
    datasets = list(pivot.columns)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(datasets), 1)

    for i, ds in enumerate(datasets):
        heights = pivot[ds].to_numpy()
        colors = [PHENOTYPE_COLORS.get(str(cat), "#333333") for cat in pivot.index]
        ax.bar(
            x + (i - (len(datasets) - 1) / 2) * width,
            heights,
            width=width,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            hatch='//' if i == 1 else None,
            alpha=0.65 if i == 1 else 1.0,
            label=ds,
        )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Percent (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in pivot.index], rotation=30, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_tmem67_plot_dataframe(
    assignments_csv: Path,
    experiments: list[str],
    metric_col: str,
    time_col: str,
    pair_filter: str | None,
) -> pd.DataFrame:
    df_assign = _read_csv(assignments_csv)
    if "embryo_id" not in df_assign.columns:
        raise ValueError(f"`embryo_id` column missing from: {assignments_csv}")
    if "cluster_category" not in df_assign.columns:
        raise ValueError(f"`cluster_category` column missing from: {assignments_csv}")

    dfs = []
    for exp_id in experiments:
        df_exp = load_experiment_dataframe(exp_id, format_version="df03")
        df_exp["experiment_id"] = exp_id
        dfs.append(df_exp)
    df_tmem67 = pd.concat(dfs, ignore_index=True)

    if pair_filter is not None:
        df_tmem67 = df_tmem67.copy()
        df_tmem67.loc[df_tmem67["pair"].isna(), "pair"] = "tmem67_spawn"
        df_tmem67 = df_tmem67[df_tmem67["pair"] == pair_filter]

    keep_cols = {"embryo_id", time_col, metric_col, "genotype", "pair", "experiment_id"}
    missing = keep_cols - set(df_tmem67.columns)
    if missing:
        raise ValueError(f"TMEM67 dataframe missing required columns: {sorted(missing)}")

    df_tmem67 = df_tmem67[list(keep_cols)].copy()

    # Inner join ensures we only plot embryos that have assignments.
    df_plot = df_tmem67.merge(
        df_assign[["embryo_id", "cluster_category"] + ([c for c in ["cluster"] if c in df_assign.columns])],
        on="embryo_id",
        how="inner",
        validate="many_to_one",
    )

    # Faceting expects sortable group labels; avoid mixed NaN/str issues.
    if "cluster_category" in df_plot.columns:
        df_plot["cluster_category"] = df_plot["cluster_category"].fillna("Unassigned").astype(str)
        # Enforce ordering for cluster categories
        df_plot["cluster_category"] = pd.Categorical(
            df_plot["cluster_category"],
            categories=PHENOTYPE_ORDER + ["Unassigned"],
            ordered=True
        )
    if "genotype" in df_plot.columns:
        df_plot["genotype"] = df_plot["genotype"].fillna("Unknown").astype(str)
        # Normalize known label typos for cleaner facets.
        df_plot["genotype"] = df_plot["genotype"].replace(
            {
                "tmem7_heterozygote": "tmem67_heterozygous",
                "tmem67_heterozygote": "tmem67_heterozygous",
            }
        )

    return df_plot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mcolon/20260104_tmem67_cluster_projection_to_cep290/output"),
        help="Directory containing projection outputs (CSV/NPY).",
    )
    p.add_argument(
        "--assignments",
        choices=["nn", "knn"],
        default="nn",
        help="Which projection assignments to plot trajectories for.",
    )
    p.add_argument("--backend", choices=["plotly", "matplotlib", "both"], default="plotly")
    args = p.parse_args()

    output_dir: Path = args.output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Frequency comparison bar plot
    freq_csv = output_dir / "cluster_frequency_comparison_spawn.csv"
    if freq_csv.exists():
        plot_cluster_frequency_comparison(
            freq_csv,
            plots_dir / "cluster_frequency_comparison_spawn.png",
            title="Cluster frequency comparison (spawn)",
        )

    # 2) Faceted trajectories: rows=genotype, cols=assigned group
    assignments_csv = output_dir / ("tmem67_nn_projection.csv" if args.assignments == "nn" else "tmem67_knn_projection.csv")

    df_plot = build_tmem67_plot_dataframe(
        assignments_csv=assignments_csv,
        experiments=TMEM67_EXPERIMENTS,
        metric_col=METRIC_COL,
        time_col=TIME_COL,
        pair_filter=PAIR_FILTER,
    )

    # 2a) Cluster composition (grouped bars) across genotype/pair/experiment_id
    # Uses the shared plotting utility to match notebook usage.
    # Prefer category labels (matches CEP290 "cluster_categories" semantics).
    if "cluster_category" in df_plot.columns:
        plot_col_by = "cluster_category"
    elif "cluster_categories" in df_plot.columns:
        plot_col_by = "cluster_categories"
    elif "cluster" in df_plot.columns:
        plot_col_by = "cluster"
    else:
        raise ValueError("No cluster column found (expected one of: cluster_category, cluster_categories, cluster).")
    fig_grouped = plot_proportion_grid(
        df_plot[~df_plot[plot_col_by].isna()],
        col_by=plot_col_by,
        row_by=["genotype", "pair", "experiment_id"],
        count_by="embryo_id",
        bar_mode="grouped",
        title=f"TMEM67 projected cluster composition ({args.assignments.upper()})",
        output_path=plots_dir / f"tmem67_cluster_composition_grouped_{args.assignments}.png",
    )
    plt.close(fig_grouped)

    out_path = plots_dir / f"tmem67_trajectories_row-genotype_col-assigned_{args.assignments}.html"
    plot_trajectories_faceted(
        df_plot,
        x_col=TIME_COL,
        y_col=METRIC_COL,
        line_by="embryo_id",
        row_by="genotype",
        col_by="cluster_category",
        color_palette=PHENOTYPE_PALETTE,
        backend=args.backend,
        output_path=(
            out_path if args.backend in {"plotly", "both"} else out_path.with_suffix(".png")
        ),
        title=f"TMEM67 trajectories faceted by genotype × assigned group ({args.assignments.upper()})",
        y_label=METRIC_COL,
        show_individual=True,
        show_error_band=False,
        trend_statistic="median",
        trend_smooth_sigma=1.5,
    )


if __name__ == "__main__":
    main()
