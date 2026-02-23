"""
B9D2 Prediction vs Divergence Visualization

Creates comprehensive 3-panel figure showing:
1. F1-score vs time (when prediction becomes possible)
2. Mean difference between groups over time (when divergence occurs)
3. Raw individual trajectories with group means (underlying biology)

This answers: "Are we predicting divergence BEFORE it happens, or just detecting it?"

Usage:
    python b9d2_prediction_vs_divergence_plot.py

Output:
    - output/figures/prediction_vs_divergence_comprehensive.png

Author: Generated via Claude Code
Date: 2025-12-10
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_IDS = ['20251119', '20251125']
PAIRS = ['b9d2_pair_7', 'b9d2_pair_8']

# Cluster configuration (from Phase 1)
SELECTED_K = 3
PENETRANT_CLUSTER = 0      # Lower length, more severe
NON_PENETRANT_CLUSTER = 2  # Higher length, rescued

# Prediction threshold (from Phase 3 results)
PREDICTION_THRESHOLD_HPF = 20  # Earliest time with F1 >= 0.7

OUTPUT_DIR = Path(__file__).parent / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Trajectory parameters
TIME_COL = 'predicted_stage_hpf'
METRIC_COL = 'total_length_um'
EMBRYO_ID_COL = 'embryo_id'
GENOTYPE_COL = 'genotype'
PAIR_COL = 'pair'

MIN_TIMEPOINTS = 5
GRID_STEP = 0.5  # HPF for interpolation

# Colors
PENETRANT_COLOR = '#d62728'     # Red
NON_PENETRANT_COLOR = '#1f77b4' # Blue


# =============================================================================
# Data Loading
# =============================================================================

def load_trajectory_data():
    """
    Load and interpolate total_length_um trajectories.

    Returns
    -------
    df_interpolated : pd.DataFrame
        Long-format interpolated data with columns:
        embryo_id, hpf, metric_value, genotype, pair, experiment_id
    common_grid : ndarray
        Time grid (hpf)
    """
    print("Loading trajectory data...")

    # Load data from experiments
    dfs = []
    for exp_id in EXPERIMENT_IDS:
        print(f"  Loading {exp_id}...")
        df = load_experiment_dataframe(exp_id, format_version='df03')
        df['experiment_id'] = exp_id
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Handle merge suffixes
    if 'total_length_um_x' in df.columns and 'total_length_um' not in df.columns:
        df['total_length_um'] = df['total_length_um_x']
    elif 'total_length_um_y' in df.columns and 'total_length_um' not in df.columns:
        df['total_length_um'] = df['total_length_um_y']

    # Filter for valid embryos
    if 'use_embryo_flag' in df.columns:
        df = df[df['use_embryo_flag'] == 1].copy()

    # Filter for target pairs
    df = df[df[PAIR_COL].isin(PAIRS)].copy()

    # Drop missing values
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, METRIC_COL, GENOTYPE_COL])

    print(f"  Loaded {len(df)} rows, {df[EMBRYO_ID_COL].nunique()} unique embryos")

    # Extract and interpolate trajectories
    print("  Interpolating trajectories...")

    # Filter embryos with enough timepoints
    embryo_counts = df.groupby(EMBRYO_ID_COL).size()
    valid_embryos = embryo_counts[embryo_counts >= MIN_TIMEPOINTS].index.tolist()
    df_filtered = df[df[EMBRYO_ID_COL].isin(valid_embryos)]

    # Create common time grid
    time_min = np.floor(df_filtered[TIME_COL].min() / GRID_STEP) * GRID_STEP
    time_max = np.ceil(df_filtered[TIME_COL].max() / GRID_STEP) * GRID_STEP
    common_grid = np.arange(time_min, time_max + GRID_STEP, GRID_STEP)

    # Interpolate each embryo
    interpolated_records = []

    for embryo_id in valid_embryos:
        embryo_data = df_filtered[df_filtered[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)

        if len(embryo_data) < 2:
            continue

        # Get metadata
        genotype = embryo_data[GENOTYPE_COL].iloc[0]
        pair = embryo_data[PAIR_COL].iloc[0]
        exp_id = embryo_data['experiment_id'].iloc[0]

        # Interpolate
        interp_values = np.interp(
            common_grid,
            embryo_data[TIME_COL].values,
            embryo_data[METRIC_COL].values
        )

        for t, v in zip(common_grid, interp_values):
            interpolated_records.append({
                'embryo_id': embryo_id,
                'hpf': t,
                'metric_value': v,
                'genotype': genotype,
                'pair': pair,
                'experiment_id': exp_id,
            })

    df_interpolated = pd.DataFrame(interpolated_records)

    print(f"  Interpolated {df_interpolated[EMBRYO_ID_COL].nunique()} embryos")
    print(f"  Time range: {time_min:.1f} - {time_max:.1f} hpf")

    return df_interpolated, common_grid


def load_cluster_assignments():
    """
    Load cluster assignments from Phase 1.

    Returns
    -------
    cluster_assignments : dict
        embryo_id -> cluster_id
    """
    assignments_file = OUTPUT_DIR / f'cluster_assignments_k{SELECTED_K}.csv'

    if not assignments_file.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found at {assignments_file}\n"
            f"Please run b9d2_trajectory_clustering.py first!"
        )

    df_assign = pd.read_csv(assignments_file)
    cluster_assignments = dict(zip(df_assign['embryo_id'], df_assign['cluster_id']))

    print(f"Loaded {len(cluster_assignments)} cluster assignments")

    return cluster_assignments


def load_classification_results():
    """
    Load F1-scores from Phase 3.

    Returns
    -------
    results_df : pd.DataFrame
        Classification results with F1-scores per time bin
    """
    results_file = OUTPUT_DIR / 'classification_results.csv'

    if not results_file.exists():
        raise FileNotFoundError(
            f"Classification results not found at {results_file}\n"
            f"Please run b9d2_trajectory_classifier.py first!"
        )

    results_df = pd.read_csv(results_file)

    print(f"Loaded classification results: {len(results_df)} time bins")

    return results_df


# =============================================================================
# Analysis
# =============================================================================

def compute_group_divergence(df_interpolated, cluster_assignments):
    """
    Compute mean difference between penetrant and non-penetrant groups over time.

    Returns
    -------
    divergence_df : pd.DataFrame
        Columns: hpf, penetrant_mean, penetrant_sem, non_penetrant_mean,
                 non_penetrant_sem, abs_difference, n_penetrant, n_non_penetrant
    """
    print("Computing group divergence over time...")

    # Add cluster labels
    df = df_interpolated.copy()
    df['cluster_id'] = df['embryo_id'].map(cluster_assignments)

    # Filter to valid clusters only
    df = df[df['cluster_id'].isin([PENETRANT_CLUSTER, NON_PENETRANT_CLUSTER])]

    # Label groups
    df['group'] = df['cluster_id'].map({
        PENETRANT_CLUSTER: 'penetrant',
        NON_PENETRANT_CLUSTER: 'non_penetrant'
    })

    # Compute stats per timepoint
    divergence_records = []

    for hpf in sorted(df['hpf'].unique()):
        df_t = df[df['hpf'] == hpf]

        pen_values = df_t[df_t['group'] == 'penetrant']['metric_value'].values
        non_pen_values = df_t[df_t['group'] == 'non_penetrant']['metric_value'].values

        if len(pen_values) > 0 and len(non_pen_values) > 0:
            pen_mean = np.mean(pen_values)
            pen_sem = stats.sem(pen_values) if len(pen_values) > 1 else 0

            non_pen_mean = np.mean(non_pen_values)
            non_pen_sem = stats.sem(non_pen_values) if len(non_pen_values) > 1 else 0

            abs_diff = abs(non_pen_mean - pen_mean)

            divergence_records.append({
                'hpf': hpf,
                'penetrant_mean': pen_mean,
                'penetrant_sem': pen_sem,
                'non_penetrant_mean': non_pen_mean,
                'non_penetrant_sem': non_pen_sem,
                'abs_difference': abs_diff,
                'n_penetrant': len(pen_values),
                'n_non_penetrant': len(non_pen_values),
            })

    divergence_df = pd.DataFrame(divergence_records)

    print(f"  Computed divergence for {len(divergence_df)} timepoints")

    return divergence_df


# =============================================================================
# Plotting
# =============================================================================

def create_comprehensive_figure(results_df, divergence_df, df_interpolated, cluster_assignments):
    """
    Create 3-panel comprehensive figure.
    """
    print("Creating comprehensive figure...")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=False)

    # Filter valid F1 scores
    df_f1 = results_df[results_df['f1_mean'].notna()].copy()

    # Get overall time range
    time_min = min(df_f1['time_bin'].min(), divergence_df['hpf'].min())
    time_max = max(df_f1['time_bin'].max(), divergence_df['hpf'].max())

    # =========================================================================
    # Panel 1: F1-Score vs Time
    # =========================================================================
    ax1 = axes[0]

    ax1.errorbar(df_f1['time_bin'], df_f1['f1_mean'],
                yerr=df_f1['f1_std'], marker='o', linewidth=2.5,
                markersize=8, capsize=4, color='#1f77b4', label='F1-score')

    ax1.axhline(y=0.7, color='green', linestyle=':', alpha=0.7, linewidth=2,
               label='Prediction threshold (F1=0.7)')
    ax1.axvline(x=PREDICTION_THRESHOLD_HPF, color='purple', linestyle='--',
               alpha=0.6, linewidth=2, label=f'Earliest prediction ({PREDICTION_THRESHOLD_HPF} hpf)')

    ax1.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax1.set_title('(A) Phenotype Prediction Performance\n'
                 'When can we predict penetrant vs non-penetrant from VAE embeddings?',
                 fontsize=14, fontweight='bold', loc='left')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(time_min, time_max)

    # =========================================================================
    # Panel 2: Mean Difference Between Groups
    # =========================================================================
    ax2 = axes[1]

    # Plot absolute difference with error bands
    ax2.plot(divergence_df['hpf'], divergence_df['abs_difference'],
            linewidth=3, color='black', label='|Mean difference|', zorder=100)

    # Add shaded error region (propagated SEM)
    combined_sem = np.sqrt(divergence_df['penetrant_sem']**2 +
                           divergence_df['non_penetrant_sem']**2)
    ax2.fill_between(divergence_df['hpf'],
                     divergence_df['abs_difference'] - combined_sem,
                     divergence_df['abs_difference'] + combined_sem,
                     alpha=0.3, color='gray', label='± SEM')

    # Mark prediction threshold
    ax2.axvline(x=PREDICTION_THRESHOLD_HPF, color='purple', linestyle='--',
               alpha=0.6, linewidth=2, label=f'Prediction threshold ({PREDICTION_THRESHOLD_HPF} hpf)')

    ax2.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Absolute Difference (µm)', fontsize=14, fontweight='bold')
    ax2.set_title('(B) Phenotypic Divergence Over Time\n'
                 'When do penetrant and non-penetrant groups actually diverge?',
                 fontsize=14, fontweight='bold', loc='left')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(time_min, time_max)

    # =========================================================================
    # Panel 3: Raw Trajectories with Group Means
    # =========================================================================
    ax3 = axes[2]

    # Add cluster labels
    df_plot = df_interpolated.copy()
    df_plot['cluster_id'] = df_plot['embryo_id'].map(cluster_assignments)
    df_plot = df_plot[df_plot['cluster_id'].isin([PENETRANT_CLUSTER, NON_PENETRANT_CLUSTER])]

    # Plot individual trajectories
    penetrant_embryos = df_plot[df_plot['cluster_id'] == PENETRANT_CLUSTER]['embryo_id'].unique()
    non_penetrant_embryos = df_plot[df_plot['cluster_id'] == NON_PENETRANT_CLUSTER]['embryo_id'].unique()

    for embryo_id in penetrant_embryos:
        embryo_data = df_plot[df_plot['embryo_id'] == embryo_id]
        ax3.plot(embryo_data['hpf'], embryo_data['metric_value'],
                alpha=0.25, linewidth=0.8, color=PENETRANT_COLOR)

    for embryo_id in non_penetrant_embryos:
        embryo_data = df_plot[df_plot['embryo_id'] == embryo_id]
        ax3.plot(embryo_data['hpf'], embryo_data['metric_value'],
                alpha=0.25, linewidth=0.8, color=NON_PENETRANT_COLOR)

    # Plot group means (bold)
    ax3.plot(divergence_df['hpf'], divergence_df['penetrant_mean'],
            linewidth=4, color=PENETRANT_COLOR,
            label=f'Penetrant mean (n={len(penetrant_embryos)})', zorder=100)
    ax3.plot(divergence_df['hpf'], divergence_df['non_penetrant_mean'],
            linewidth=4, color=NON_PENETRANT_COLOR,
            label=f'Non-penetrant mean (n={len(non_penetrant_embryos)})', zorder=100)

    # Mark prediction threshold
    ax3.axvline(x=PREDICTION_THRESHOLD_HPF, color='purple', linestyle='--',
               alpha=0.6, linewidth=2, label=f'Prediction threshold ({PREDICTION_THRESHOLD_HPF} hpf)')

    ax3.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Total Length (µm)', fontsize=14, fontweight='bold')
    ax3.set_title('(C) Individual Embryo Trajectories and Group Means\n'
                 'Biological data underlying the prediction and divergence',
                 fontsize=14, fontweight='bold', loc='left')
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(time_min, time_max)

    # Overall title
    fig.suptitle('B9D2 Phenotypic Prediction vs Divergence Analysis\n'
                f'Penetrant (Cluster {PENETRANT_CLUSTER}) vs Non-Penetrant (Cluster {NON_PENETRANT_CLUSTER})',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / 'prediction_vs_divergence_comprehensive.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("B9D2 PREDICTION VS DIVERGENCE VISUALIZATION")
    print("=" * 80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Pairs: {PAIRS}")
    print(f"Penetrant cluster: {PENETRANT_CLUSTER}")
    print(f"Non-penetrant cluster: {NON_PENETRANT_CLUSTER}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD_HPF} hpf")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading trajectory data...")
    df_interpolated, common_grid = load_trajectory_data()

    print("\n[2/5] Loading cluster assignments...")
    cluster_assignments = load_cluster_assignments()

    print("\n[3/5] Loading classification results...")
    results_df = load_classification_results()

    print("\n[4/5] Computing group divergence...")
    divergence_df = compute_group_divergence(df_interpolated, cluster_assignments)

    print("\n[5/5] Creating comprehensive figure...")
    create_comprehensive_figure(results_df, divergence_df, df_interpolated, cluster_assignments)

    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {FIGURES_DIR / 'prediction_vs_divergence_comprehensive.png'}")
    print("\nThis figure shows:")
    print("  (A) When prediction becomes possible (F1-score)")
    print("  (B) When phenotypic divergence actually occurs (mean difference)")
    print("  (C) The underlying biological data (individual trajectories)")
    print("\nKey question: Does prediction occur BEFORE divergence?")
    print("=" * 80)


if __name__ == '__main__':
    main()
