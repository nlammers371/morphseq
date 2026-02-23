"""
Side-by-side comparison of cluster distributions for homozygous-only embryos.

Compares CEP290 spawn homozygous vs TMEM67 spawn homozygous cluster proportions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, "/net/trapnell/vol1/home/mdcolon/proj/morphseq")

CEP290_DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction/final_data")

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

# Backward compatibility aliases
CLUSTER_COLORS = PHENOTYPE_COLORS
CLUSTER_ORDER = PHENOTYPE_ORDER


def load_cep290_homozygous() -> pd.DataFrame:
    """Load CEP290 spawn homozygous embryos with cluster labels."""
    df_cep290 = pd.read_csv(CEP290_DATA_DIR / "embryo_data_with_labels.csv", low_memory=False)
    df_labels = pd.read_csv(CEP290_DATA_DIR / "embryo_cluster_labels.csv", low_memory=False)

    # Get unique embryo data
    df_unique = df_cep290[['embryo_id', 'genotype', 'pair']].drop_duplicates(subset='embryo_id')

    # Merge with labels
    df_unique = df_unique.merge(
        df_labels[['embryo_id', 'cluster_categories']],
        on='embryo_id',
        how='inner'
    )

    # Filter to spawn homozygous only
    df_hom = df_unique[
        (df_unique['pair'] == 'cep290_spawn') &
        (df_unique['genotype'] == 'cep290_homozygous') &
        (df_unique['cluster_categories'].notna())
    ].copy()

    # Enforce ordering
    df_hom['cluster_categories'] = pd.Categorical(
        df_hom['cluster_categories'],
        categories=PHENOTYPE_ORDER,
        ordered=True
    )

    return df_hom


def load_tmem67_homozygous(assignments_csv: Path) -> pd.DataFrame:
    """Load TMEM67 spawn homozygous embryos with cluster assignments."""
    df_assign = pd.read_csv(assignments_csv, low_memory=False)

    # Filter to spawn homozygous only
    df_hom = df_assign[
        (df_assign['pair'] == 'tmem67_spawn') &
        (df_assign['genotype'] == 'tmem67_homozygous') &
        (df_assign['cluster_category'].notna())
    ].copy()

    # Enforce ordering
    df_hom['cluster_category'] = pd.Categorical(
        df_hom['cluster_category'],
        categories=PHENOTYPE_ORDER,
        ordered=True
    )

    return df_hom


def plot_side_by_side_comparison(
    df_cep290: pd.DataFrame,
    df_tmem67: pd.DataFrame,
    output_path: Path,
) -> plt.Figure:
    """
    Create side-by-side bar plot comparing cluster distributions.

    Parameters
    ----------
    df_cep290 : pd.DataFrame
        CEP290 homozygous data with 'cluster_categories' column
    df_tmem67 : pd.DataFrame
        TMEM67 homozygous data with 'cluster_category' column
    output_path : Path
        Where to save the plot

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    # Count cluster frequencies
    cep290_counts = df_cep290['cluster_categories'].value_counts()
    tmem67_counts = df_tmem67['cluster_category'].value_counts()

    # Convert to proportions
    cep290_props = (cep290_counts / cep290_counts.sum() * 100).to_dict()
    tmem67_props = (tmem67_counts / tmem67_counts.sum() * 100).to_dict()

    # Ensure all clusters are present
    for cluster in CLUSTER_ORDER:
        if cluster not in cep290_props:
            cep290_props[cluster] = 0.0
        if cluster not in tmem67_props:
            tmem67_props[cluster] = 0.0

    # Prepare data for plotting
    cep290_values = [cep290_props[c] for c in CLUSTER_ORDER]
    tmem67_values = [tmem67_props[c] for c in CLUSTER_ORDER]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(CLUSTER_ORDER))
    width = 0.35

    # Plot bars (color by cluster category; use hatch to distinguish datasets)
    cluster_colors = [CLUSTER_COLORS.get(c, '#333333') for c in CLUSTER_ORDER]
    bars1 = ax.bar(
        x - width / 2,
        cep290_values,
        width,
        color=cluster_colors,
        edgecolor='black',
        linewidth=1.0,
    )
    bars2 = ax.bar(
        x + width / 2,
        tmem67_values,
        width,
        color=cluster_colors,
        edgecolor='black',
        linewidth=1.0,
        hatch='//',
        alpha=0.65,
    )

    # Custom legend so colors remain phenotype-meaningful
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor='white', edgecolor='black', label=f'CEP290 spawn hom (n={len(df_cep290)})'),
            Patch(facecolor='white', edgecolor='black', hatch='//', label=f'TMEM67 spawn hom (n={len(df_tmem67)})'),
        ],
        loc='upper right',
        frameon=True,
        fontsize=10,
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

    # Customize plot
    ax.set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster Category', fontsize=12, fontweight='bold')
    ax.set_title('Homozygous Spawn: CEP290 vs TMEM67 Cluster Distribution',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(CLUSTER_ORDER, rotation=30, ha='right')
    ax.set_ylim(0, max(max(cep290_values), max(tmem67_values)) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("CLUSTER DISTRIBUTION SUMMARY")
    print("="*60)
    print(f"\nCEP290 spawn homozygous (n={len(df_cep290)}):")
    for cluster in CLUSTER_ORDER:
        print(f"  {cluster:20s}: {cep290_props[cluster]:5.1f}%")

    print(f"\nTMEM67 spawn homozygous (n={len(df_tmem67)}):")
    for cluster in CLUSTER_ORDER:
        print(f"  {cluster:20s}: {tmem67_props[cluster]:5.1f}%")

    print(f"\nDifference (TMEM67 - CEP290):")
    for cluster in CLUSTER_ORDER:
        diff = tmem67_props[cluster] - cep290_props[cluster]
        print(f"  {cluster:20s}: {diff:+5.1f}%")
    print("="*60)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare homozygous cluster distributions"
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
        help="Which projection method to use",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    plots_dir = output_dir / "plots"

    # Determine assignment CSV
    assignments_csv = output_dir / (
        "tmem67_nn_projection.csv" if args.assignments == "nn"
        else "tmem67_knn_projection.csv"
    )

    print("="*60)
    print("HOMOZYGOUS SPAWN COMPARISON: CEP290 vs TMEM67")
    print("="*60)

    print("\nLoading CEP290 homozygous spawn data...")
    df_cep290 = load_cep290_homozygous()

    print("Loading TMEM67 homozygous spawn data...")
    df_tmem67 = load_tmem67_homozygous(assignments_csv)

    print("\nCreating comparison plot...")
    output_path = plots_dir / f'homozygous_spawn_comparison_{args.assignments}.png'
    fig = plot_side_by_side_comparison(df_cep290, df_tmem67, output_path)

    plt.close(fig)

    print("\nDONE")


if __name__ == "__main__":
    main()
