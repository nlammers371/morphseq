"""
B9D2 Phenotype Distribution by Pair and Genotype

Creates faceted plots using the new plot_proportion_faceted and plot_multimetric_trajectories APIs:

1. Phenotype distribution: row_by=genotype, col_by=pair, color_by_grouping=phenotype
2. Multi-metric trajectories: col_by=phenotype, color_by_grouping=pair

Usage:
    python b9d2_phenotype_distribution_by_pair.py

Output:
    - faceted_plots/phenotype_distribution_by_pair.png
    - faceted_plots/phenotype_multimetric_trajectories.html

Author: Generated via Claude Code
Date: 2025-12-29
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe
from src.analyze.trajectory_analysis.facetted_plotting import (
    plot_proportion_faceted,
    plot_multimetric_trajectories,
)
from src.analyze.trajectory_analysis.genotype_styling import extract_genotype_suffix
from src.analyze.trajectory_analysis.plot_config import (
    PHENOTYPE_COLORS,
    PHENOTYPE_ORDER,
    GENOTYPE_SUFFIX_ORDER,
)

# Import shared functions from b9d2_phenotype_comparison
sys.path.insert(0, str(Path(__file__).parent))
from b9d2_phenotype_comparison import (
    parse_phenotype_file,
    load_all_phenotypes,
    load_experiment_data,
    extract_wildtype_embryos,
)

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_IDS = ['20251121', '20251125']

# Phenotype file paths
PHENOTYPE_DIR = Path(__file__).parent.parent / '20251219_b9d2_phenotype_extraction' / 'phenotype_lists'
CE_FILE = PHENOTYPE_DIR / 'b9d2-CE-phenotype.txt'
HTA_FILE = PHENOTYPE_DIR / 'b9d2-HTA-embryos.txt'
BA_RESCUE_FILE = PHENOTYPE_DIR / 'b9d2-curved-rescue.txt'

# Output directories
OUTPUT_DIR = Path(__file__).parent / 'faceted_plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
EMBRYO_ID_COL = 'embryo_id'
GENOTYPE_COL = 'genotype'
TIME_COL = 'predicted_stage_hpf'

# Pair colors will be auto-assigned from standard palette (no need to hardcode)


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_phenotype_dataframe(
    df: pd.DataFrame,
    phenotypes: Dict[str, List[str]],
    wildtype_ids: List[str],
    experiment_ids: List[str]
) -> pd.DataFrame:
    """
    Prepare dataframe with phenotype class assignments.

    Adds columns:
    - phenotype: CE, HTA, BA_rescue, non_penetrant
    - genotype_suffix: wildtype, heterozygous, homozygous
    - pair: extracted from genotype string

    Parameters
    ----------
    df : pd.DataFrame
        Full experiment data
    phenotypes : Dict[str, List[str]]
        Dictionary of phenotype -> embryo IDs
    wildtype_ids : List[str]
        Wildtype embryo IDs
    experiment_ids : List[str]
        Which experiments to include

    Returns
    -------
    df_prepared : pd.DataFrame
        Dataframe with phenotype, genotype_suffix, pair columns
    """
    # Filter to relevant experiments
    df_filtered = df[df['experiment_id'].isin(experiment_ids)].copy()

    # Create phenotype class mapping
    all_phenotype_ids = set()
    phenotype_map = {}

    for phenotype_name, embryo_ids in phenotypes.items():
        for eid in embryo_ids:
            phenotype_map[eid] = phenotype_name
            all_phenotype_ids.add(eid)

    # Assign phenotype column
    # Only 4 phenotypes: CE, HTA, BA_rescue, non_penetrant
    # Wildtype embryos are NOT a phenotype - they are controls (genotype_suffix='wildtype')
    def get_phenotype(row):
        eid = row[EMBRYO_ID_COL]
        genotype = row[GENOTYPE_COL] if GENOTYPE_COL in row else ''

        # If in a known phenotype list, use that
        if eid in phenotype_map:
            return phenotype_map[eid]

        # Wildtype genotype = non_penetrant (they are controls, not a separate phenotype)
        if 'wildtype' in str(genotype).lower():
            return 'non_penetrant'

        # Mutant embryos not in phenotype lists = non_penetrant
        if eid not in all_phenotype_ids:
            return 'non_penetrant'

        return None

    df_filtered['phenotype'] = df_filtered.apply(get_phenotype, axis=1)

    # Extract genotype suffix
    df_filtered['genotype_suffix'] = df_filtered[GENOTYPE_COL].apply(
        lambda x: extract_genotype_suffix(str(x)) if pd.notna(x) else 'unknown'
    )

    # Use the 'pair' column directly if it exists
    # If pair is NA but genotype contains 'b9d2', call it 'b9d2_spawn'
    if 'pair' in df_filtered.columns:
        df_filtered['pair'] = df_filtered['pair'].fillna('b9d2_spawn')
    else:
        df_filtered['pair'] = 'b9d2_spawn'

    # Filter to embryos with phenotype assignment
    df_prepared = df_filtered.dropna(subset=['phenotype'])

    return df_prepared


# =============================================================================
# Plot 1: Phenotype Distribution (row_by=genotype, col_by=pair)
# =============================================================================

def create_phenotype_distribution_plot(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create phenotype distribution plot using plot_proportion_faceted.

    Grid structure:
    - Rows: genotype_suffix (wildtype, heterozygous, homozygous)
    - Columns: pair (pair_2, pair_7, pair_8, etc.)
    - Bar colors: phenotype (CE, HTA, BA_rescue, non_penetrant)

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataframe with phenotype, genotype_suffix, pair columns
    output_path : Path
        Where to save the figure

    Returns
    -------
    fig : plt.Figure
    """
    # Get one row per embryo for counting
    df_unique = df.drop_duplicates(subset=[EMBRYO_ID_COL])

    # Make a copy for plotting
    df_plot = df_unique.copy()

    # Filter out unknown pairs
    df_plot = df_plot[df_plot['pair'] != 'unknown']

    # Define ordering - take pair values at face value, sorted
    pair_order = sorted(df_plot['pair'].unique())
    genotype_order = ['wildtype', 'heterozygous', 'homozygous']
    phenotype_order = ['CE', 'HTA', 'BA_rescue', 'non_penetrant']

    fig = plot_proportion_faceted(
        df_plot,
        color_by_grouping='phenotype',
        row_by='genotype_suffix',
        col_by='pair',
        count_by=EMBRYO_ID_COL,
        facet_order={
            'row_by': genotype_order,
            'col_by': pair_order,
        },
        color_order=phenotype_order,
        color_palette=PHENOTYPE_COLORS,
        normalize=True,
        bar_mode='grouped',
        title='Phenotype Distribution by Pair and Genotype (Exp 20251121 & 20251125)',
        output_path=output_path,
        show_counts=True,
    )

    return fig


# =============================================================================
# Plot 2: Multi-Metric Trajectories (col_by=phenotype, color_by=pair)
# =============================================================================

def create_multimetric_plot(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Create multi-metric trajectory plot using plot_multimetric_trajectories.

    Grid structure:
    - Rows: metrics (baseline_deviation_normalized, total_length_um)
    - Columns: phenotype (CE, HTA, BA_rescue, non_penetrant, wildtype)
    - Trend line colors: pair

    Parameters
    ----------
    df : pd.DataFrame
        Full trajectory data with phenotype column
    output_path : Path
        Where to save the figure

    Returns
    -------
    fig : plotly or matplotlib figure
    """
    # Filter out unknown pairs
    df_plot = df[df['pair'] != 'unknown'].copy()

    # Metrics to compare
    metrics = ['baseline_deviation_normalized', 'total_length_um']
    metric_labels = {
        'baseline_deviation_normalized': 'Curvature (normalized)',
        'total_length_um': 'Length (μm)',
    }

    # Phenotype order (only 4 phenotypes)
    phenotype_order = ['CE', 'HTA', 'BA_rescue', 'non_penetrant']

    fig = plot_multimetric_trajectories(
        df_plot,
        metrics=metrics,
        col_by='phenotype',
        x_col=TIME_COL,
        line_by=EMBRYO_ID_COL,
        color_by_grouping='pair',
        metric_labels=metric_labels,
        col_order=phenotype_order,
        backend='plotly',
        output_path=output_path,
        title='Trajectories by Phenotype, Colored by Pair',
        trend_statistic='median',
    )

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Run faceted phenotype distribution plots."""

    print("\n" + "=" * 70)
    print("B9D2 Phenotype Distribution by Pair and Genotype")
    print("Using new plot_proportion_faceted API")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    phenotypes = load_all_phenotypes()
    df = load_experiment_data()
    wildtype_ids = extract_wildtype_embryos(df, phenotypes)

    # Prepare phenotype dataframe
    print("Preparing phenotype assignments...")
    df_prepared = prepare_phenotype_dataframe(df, phenotypes, wildtype_ids, EXPERIMENT_IDS)

    print(f"\nData Summary:")
    print(f"  Total rows: {len(df_prepared)}")
    print(f"  Unique embryos: {df_prepared['embryo_id'].nunique()}")
    print(f"\n  Phenotypes:")
    for pheno in sorted(df_prepared['phenotype'].unique()):
        count = df_prepared[df_prepared['phenotype'] == pheno]['embryo_id'].nunique()
        print(f"    {pheno}: {count}")
    print(f"\n  Pairs:")
    for pair in sorted(df_prepared['pair'].unique()):
        count = df_prepared[df_prepared['pair'] == pair]['embryo_id'].nunique()
        print(f"    {pair}: {count}")
    print(f"\n  Genotype suffixes:")
    for suffix in sorted(df_prepared['genotype_suffix'].unique()):
        count = df_prepared[df_prepared['genotype_suffix'] == suffix]['embryo_id'].nunique()
        print(f"    {suffix}: {count}")

    # Plot 1: Phenotype Distribution
    print("\n" + "-" * 50)
    print("Creating phenotype distribution plot...")
    output_fig1 = OUTPUT_DIR / 'phenotype_distribution_by_pair.png'
    fig1 = create_phenotype_distribution_plot(df_prepared, output_fig1)
    plt.close(fig1)
    print(f"Saved: {output_fig1}")

    # Plot 2: Multi-Metric Trajectories
    print("\n" + "-" * 50)
    print("Creating multi-metric trajectory plot...")
    output_fig2 = OUTPUT_DIR / 'phenotype_multimetric_trajectories.html'
    fig2 = create_multimetric_plot(df_prepared, output_fig2)
    print(f"Saved: {output_fig2}")

    print("\n" + "=" * 70)
    print("Complete! Figures saved to:")
    print(f"  {output_fig1}")
    print(f"  {output_fig2}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
