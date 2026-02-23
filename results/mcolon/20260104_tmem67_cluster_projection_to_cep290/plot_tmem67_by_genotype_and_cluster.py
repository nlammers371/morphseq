"""
Plot TMEM67 projection results by genotype and cluster assignment.

Generates two types of visualizations:
1. Proportion grid bar plots showing cluster distribution by genotype
2. Multi-metric trajectory plots faceted by cluster with genotype color grouping

Usage:
    python plot_tmem67_by_genotype_and_cluster.py [--assignments nn|knn] [--backend plotly|matplotlib|both]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Add project root to path
import sys
sys.path.insert(0, "/net/trapnell/vol1/home/mdcolon/proj/morphseq")

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe
from src.analyze.trajectory_analysis.facetted_plotting import (
    plot_proportion_faceted,
    plot_multimetric_trajectories,
)
import matplotlib.pyplot as plt
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

# Configuration matching run_projection.py
TMEM67_EXPERIMENTS = ["20250711", "20251205"]
CEP290_DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction/final_data")
TIME_COL = "predicted_stage_hpf"
METRIC_COL = "baseline_deviation_normalized"
METRIC_COL_2 = "total_length_um"


def load_and_prepare_data(
    assignments_csv: Path,
    experiments: list[str],
) -> pd.DataFrame:
    """
    Load projection assignments and merge with full trajectory data.

    Parameters
    ----------
    assignments_csv : Path
        Path to projection CSV (tmem67_nn_projection.csv or tmem67_knn_projection.csv)
    experiments : list[str]
        List of experiment IDs to load

    Returns
    -------
    pd.DataFrame
        Merged dataframe with trajectories and cluster assignments
    """
    # Load projection assignments
    df_assign = pd.read_csv(assignments_csv, low_memory=False)
    print(f"Loaded {len(df_assign)} projection assignments")

    # Validate required columns
    required_cols = {"embryo_id", "cluster_category"}
    if not required_cols.issubset(df_assign.columns):
        raise ValueError(f"Projection CSV missing required columns: {required_cols - set(df_assign.columns)}")

    # Load TMEM67 trajectory data
    dfs = []
    for exp_id in experiments:
        print(f"Loading experiment {exp_id}...")
        df_exp = load_experiment_dataframe(exp_id, format_version="df03")
        df_exp["experiment_id"] = exp_id
        dfs.append(df_exp)

    df_tmem67 = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_tmem67)} trajectory rows from {df_tmem67['embryo_id'].nunique()} embryos")

    # Fill NaN pairs with spawn
    df_tmem67.loc[df_tmem67["pair"].isna(), "pair"] = "tmem67_spawn"

    # Keep only required columns
    keep_cols = {
        "embryo_id", TIME_COL, METRIC_COL, METRIC_COL_2,
        "genotype", "pair", "experiment_id"
    }
    available_cols = keep_cols & set(df_tmem67.columns)
    df_tmem67 = df_tmem67[list(available_cols)].copy()

    # Merge with cluster assignments
    df_merged = df_tmem67.merge(
        df_assign[["embryo_id", "cluster_category", "cluster"]],
        on="embryo_id",
        how="inner",
        validate="many_to_one",
    )
    print(f"Merged: {len(df_merged)} trajectory rows from {df_merged['embryo_id'].nunique()} embryos with assignments")

    # Clean up categorical columns
    df_merged["cluster_category"] = df_merged["cluster_category"].fillna("Unassigned").astype(str)
    df_merged["genotype"] = df_merged["genotype"].fillna("Unknown").astype(str)

    # Normalize genotype labels
    df_merged["genotype"] = df_merged["genotype"].replace({
        "tmem7_heterozygote": "tmem67_heterozygous",
        "tmem67_heterozygote": "tmem67_heterozygous",
    })

    # Filter out unassigned
    df_merged = df_merged[df_merged["cluster_category"] != "Unassigned"].copy()
    print(f"After filtering unassigned: {df_merged['embryo_id'].nunique()} embryos")

    # Enforce ordering
    df_merged["cluster_category"] = pd.Categorical(
        df_merged["cluster_category"],
        categories=PHENOTYPE_ORDER,
        ordered=True
    )

    return df_merged


def load_cep290_data() -> pd.DataFrame:
    """
    Load CEP290 data with cluster labels.

    Returns
    -------
    pd.DataFrame
        CEP290 embryo data with cluster assignments
    """
    # Load CEP290 data with labels
    df_cep290 = pd.read_csv(CEP290_DATA_DIR / "embryo_data_with_labels.csv", low_memory=False)
    df_labels = pd.read_csv(CEP290_DATA_DIR / "embryo_cluster_labels.csv", low_memory=False)

    print(f"Loaded CEP290 data: {df_cep290['embryo_id'].nunique()} embryos")

    # Get unique embryo data with labels
    df_unique = df_cep290[['embryo_id', 'genotype', 'pair']].drop_duplicates(subset='embryo_id')

    # Merge with cluster labels
    df_unique = df_unique.merge(
        df_labels[['embryo_id', 'cluster_categories']],
        on='embryo_id',
        how='inner'
    )

    # Rename for consistency
    df_unique = df_unique.rename(columns={'cluster_categories': 'cluster_category'})

    # Filter to valid cluster assignments
    df_unique = df_unique[df_unique['cluster_category'].notna()].copy()

    # Normalize genotype labels
    df_unique['genotype'] = df_unique['genotype'].replace({
        'cep290_heterozygote': 'cep290_heterozygous',
    })

    # Enforce ordering
    df_unique['cluster_category'] = pd.Categorical(
        df_unique['cluster_category'],
        categories=PHENOTYPE_ORDER,
        ordered=True
    )

    print(f"CEP290 embryos with valid clusters: {len(df_unique)}")

    return df_unique


def main():
    parser = argparse.ArgumentParser(
        description="Plot TMEM67 projection results by genotype and cluster"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mcolon/20260104_tmem67_cluster_projection_to_cep290/output"),
        help="Directory containing projection outputs",
    )
    parser.add_argument(
        "--assignments",
        choices=["nn", "knn"],
        default="nn",
        help="Which projection method to use (nearest neighbor or k-NN)",
    )
    parser.add_argument(
        "--backend",
        choices=["plotly", "matplotlib", "both"],
        default="both",
        help="Plotting backend for trajectory plots",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine assignment CSV
    assignments_csv = output_dir / (
        "tmem67_nn_projection.csv" if args.assignments == "nn"
        else "tmem67_knn_projection.csv"
    )

    print("="*80)
    print("TMEM67 & CEP290 Cluster Distribution Visualization")
    print("="*80)

    # Load and prepare data
    print("\n[1/5] Loading TMEM67 data...")
    df_tmem67_full = load_and_prepare_data(assignments_csv, TMEM67_EXPERIMENTS)
    df_tmem67_unique = df_tmem67_full.drop_duplicates(subset="embryo_id")

    print("\n[2/5] Loading CEP290 data...")
    df_cep290_unique = load_cep290_data()

    # =========================================================================
    # Plot 1: TMEM67 Proportion Faceted (both pairs)
    # =========================================================================
    print("\n[3/5] Creating TMEM67 proportion plot (all pairs)...")

    fig_tmem67 = plot_proportion_faceted(
        df_tmem67_unique,
        color_by_grouping='cluster_category',
        color_palette=PHENOTYPE_PALETTE,
        col_by='genotype',
        row_by='pair',
        count_by='embryo_id',
        bar_mode='grouped',
        normalize=True,
        title=f'TMEM67 Cluster Distribution by Genotype ({args.assignments.upper()})',
        output_path=plots_dir / f'tmem67_cluster_by_genotype_{args.assignments}.png',
    )
    print(f"  Saved: {plots_dir / f'tmem67_cluster_by_genotype_{args.assignments}.png'}")

    # =========================================================================
    # Plot 2: CEP290 spawn only
    # =========================================================================
    print("\n[4/5] Creating CEP290 proportion plots...")

    df_cep290_spawn = df_cep290_unique[df_cep290_unique['pair'] == 'cep290_spawn'].copy()
    fig_cep290_spawn = plot_proportion_faceted(
        df_cep290_spawn,
        color_by_grouping='cluster_category',
        color_palette=PHENOTYPE_PALETTE,
        col_by='genotype',
        row_by=None,  # Single row
        count_by='embryo_id',
        bar_mode='grouped',
        normalize=True,
        title='CEP290 Spawn - Cluster Distribution by Genotype',
        output_path=plots_dir / 'cep290_spawn_cluster_by_genotype.png',
    )
    print(f"  Saved: {plots_dir / 'cep290_spawn_cluster_by_genotype.png'}")

    # =========================================================================
    # Plot 3: CEP290 all pairs
    # =========================================================================
    fig_cep290_all = plot_proportion_faceted(
        df_cep290_unique,
        color_by_grouping='cluster_category',
        color_palette=PHENOTYPE_PALETTE,
        col_by='genotype',
        row_by='pair',  # All CEP290 pairs
        count_by='embryo_id',
        bar_mode='grouped',
        normalize=True,
        title='CEP290 All Pairs - Cluster Distribution by Genotype',
        output_path=plots_dir / 'cep290_all_pairs_cluster_by_genotype.png',
    )
    print(f"  Saved: {plots_dir / 'cep290_all_pairs_cluster_by_genotype.png'}")

    # =========================================================================
    # Plot 4: Multi-Metric Trajectories (Faceted by Cluster)
    # =========================================================================
    print("\n[5/5] Creating TMEM67 trajectory plots...")

    # Separate by pair
    for pair_name in ["tmem67_spawn", "tmem67_pair_1"]:
        df_pair = df_tmem67_full[df_tmem67_full["pair"] == pair_name].copy()

        if len(df_pair) == 0:
            print(f"  Skipping {pair_name} (no data)")
            continue

        print(f"  Processing {pair_name} ({df_pair['embryo_id'].nunique()} embryos)...")

        # Check if both metrics are available
        metrics = [METRIC_COL]
        if METRIC_COL_2 in df_pair.columns and df_pair[METRIC_COL_2].notna().any():
            metrics.append(METRIC_COL_2)

        metric_labels = {
            METRIC_COL: 'Curvature (normalized)',
            METRIC_COL_2: 'Body Length (μm)',
        }

        output_path = plots_dir / f'tmem67_{pair_name}_trajectories_by_cluster_{args.assignments}.html'

        fig_traj = plot_multimetric_trajectories(
            df_pair,
            metrics=metrics,
            col_by='cluster_category',
            color_by_grouping='genotype',
            x_col=TIME_COL,
            metric_labels=metric_labels,
            title=f'TMEM67 {pair_name} - Trajectories by Cluster ({args.assignments.upper()})',
            x_label='Time (hpf)',
            backend=args.backend,
            bin_width=2.0,
            show_individual=False,
            show_error_band=True,
            error_band_alpha=0.15,
            trend_statistic='median',
            trend_smooth_sigma=1.5,
            output_path=output_path if args.backend in {"plotly", "both"} else output_path.with_suffix(".png"),
        )

        if args.backend in {"plotly", "both"}:
            print(f"    Saved: {output_path}")
        if args.backend in {"matplotlib", "both"}:
            print(f"    Saved: {output_path.with_suffix('.png')}")

        # Create homozygous-only plot for spawn
        if pair_name == "tmem67_spawn":
            df_hom = df_pair[df_pair["genotype"] == "tmem67_homozygous"].copy()

            if len(df_hom) > 0:
                print(f"  Processing {pair_name} HOMOZYGOUS ONLY ({df_hom['embryo_id'].nunique()} embryos)...")

                # Create dummy column for col_by (workaround for col_by=None)
                df_hom['_dummy_col'] = 'All Homozygous'

                output_path_hom = plots_dir / f'tmem67_{pair_name}_homozygous_trajectories_by_cluster_{args.assignments}.html'

                fig_traj_hom = plot_multimetric_trajectories(
                    df_hom,
                    metrics=metrics,
                    col_by='_dummy_col',  # Dummy column so all data goes in one facet
                    color_by_grouping='cluster_category',  # Color by cluster instead of genotype
                    color_palette=PHENOTYPE_PALETTE,  # Use standardized phenotype colors
                    x_col=TIME_COL,
                    metric_labels=metric_labels,
                    title=f'TMEM67 {pair_name} Homozygous - Trajectories by Cluster ({args.assignments.upper()})',
                    x_label='Time (hpf)',
                    backend=args.backend,
                    bin_width=2.0,
                    show_individual=False,
                    show_error_band=True,
                    error_band_alpha=0.15,
                    trend_statistic='median',
                    trend_smooth_sigma=1.5,
                    output_path=output_path_hom if args.backend in {"plotly", "both"} else output_path_hom.with_suffix(".png"),
                )

                if args.backend in {"plotly", "both"}:
                    print(f"    Saved: {output_path_hom}")
                if args.backend in {"matplotlib", "both"}:
                    print(f"    Saved: {output_path_hom.with_suffix('.png')}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
