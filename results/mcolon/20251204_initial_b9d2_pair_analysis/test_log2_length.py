#!/usr/bin/env python3
"""
Quick test: visualize log2(total_length_um) for 20251119 b9d2 data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.data_loading import _load_df03_format
from src.analyze.trajectory_analysis.pair_analysis import (
    plot_genotypes_overlaid,
)

# Configuration
EXPERIMENT_ID = '20251119'
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'
GENOTYPE_COL = 'genotype'

# Genotype configuration
GENOTYPE_ORDER = ['b9d2_wildtype', 'b9d2_heterozygous', 'b9d2_homozygous']
GENOTYPE_COLORS = {
    'b9d2_wildtype': '#2E7D32',
    'b9d2_heterozygous': '#FFA500',
    'b9d2_homozygous': '#D32F2F',
}

# Output
OUTPUT_DIR = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251204_initial_b9d2_pair_analysis')
OUTPUT_PATH = OUTPUT_DIR / 'test_log2_total_length.png'

# Smoothing
SMOOTH_METHOD = 'gaussian'
SMOOTH_PARAMS = {'sigma': 1.5}


def main():
    print("=" * 60)
    print("Testing log2(total_length_um) for 20251119")
    print("=" * 60)

    # Load data
    print(f"\nLoading data for {EXPERIMENT_ID}...")
    df = _load_df03_format(EXPERIMENT_ID)

    # Handle column collision (total_length_um_x from curvature, _y from df03)
    if 'total_length_um_y' in df.columns:
        df['total_length_um'] = df['total_length_um_y']
        print("  Using total_length_um_y (from df03)")

    # Filter for valid embryos
    df = df[df['use_embryo_flag'] == 1].copy()

    # Drop rows with missing values
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, 'total_length_um', PAIR_COL, GENOTYPE_COL])

    # Apply log2 transform
    df['log2_total_length_um'] = np.log2(df['total_length_um'])
    print(f"\nCreated log2_total_length_um column")
    print(f"  Original range: {df['total_length_um'].min():.1f} - {df['total_length_um'].max():.1f} µm")
    print(f"  Log2 range: {df['log2_total_length_um'].min():.2f} - {df['log2_total_length_um'].max():.2f}")

    # Get pairs
    pairs = sorted(df[PAIR_COL].unique())
    print(f"\nPairs: {pairs}")

    # Create plot
    print(f"\nGenerating plot...")
    plot_genotypes_overlaid(
        df, pairs,
        group_col=PAIR_COL,
        genotype_col=GENOTYPE_COL,
        time_col=TIME_COL,
        metric_col='log2_total_length_um',
        embryo_id_col=EMBRYO_ID_COL,
        genotype_order=GENOTYPE_ORDER,
        genotype_colors=GENOTYPE_COLORS,
        output_path=OUTPUT_PATH,
        title='Log2(Total Length) Trajectories by Pair - All Genotypes Compared',
        smooth_method=SMOOTH_METHOD,
        smooth_params=SMOOTH_PARAMS,
    )

    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
