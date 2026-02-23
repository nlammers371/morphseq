#!/usr/bin/env python3
"""
Visualize curvature trajectories with ranking-based coloring.

This script creates trajectory plots where embryos are colored by their ranking
based on average curvature values in a specified time window. This helps understand:
- How predictive is curvature at one timepoint for curvature at another?
- Do embryos maintain their relative ranking over time?
- Are there genotype-specific patterns in temporal consistency?

Uses the reusable trajectory_visualization module.

Output
------
- Trajectory plots colored by rank (one per metric)
- Trajectory plots showing full developmental time range
- Embryo ranking tables (CSV)
"""

from pathlib import Path
import pandas as pd

# Import data loading from this directory
from load_data import get_analysis_dataframe, get_genotype_short_name

# Import trajectory visualization
from trajectory_visualization import plot_ranked_trajectories


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent
FIGURE_DIR = RESULTS_DIR / 'outputs' / 'figures' / '03_trajectory_rankings'
TABLE_DIR = RESULTS_DIR / 'outputs' / 'tables' / '03_trajectory_rankings'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
PRIMARY_METRICS = ['normalized_baseline_deviation']

# Time windows for analysis
RANKING_WINDOWS = [
    (44, 50),   # Very early development
    (60, 72),   # Mid development (critical window)
    (100, 130), # Late development / extended timepoints
]

# Display full trajectory
FULL_DISPLAY_WINDOW = None  # Show all available data


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CREATING TRAJECTORY VISUALIZATIONS WITH RANKING-BASED COLORING")
    print("="*80)

    # Load data
    print("\nSTEP 1: LOADING DATA")
    df, metadata = get_analysis_dataframe()

    print(f"  Loaded {len(df)} timepoints from {df['embryo_id'].nunique()} embryos")
    print(f"  Time range: {df['predicted_stage_hpf'].min():.1f} - {df['predicted_stage_hpf'].max():.1f} hpf")

    # Create trajectory plots for each metric and ranking window
    print("\nSTEP 2: CREATING TRAJECTORY PLOTS")

    for metric in PRIMARY_METRICS:
        print(f"\n  Processing metric: {metric}")

        for ranking_window in RANKING_WINDOWS:
            min_t, max_t = ranking_window
            print(f"    Ranking window: {min_t}-{max_t} hpf")

            # Create plot
            try:
                fig, rankings = plot_ranked_trajectories(
                    df,
                    metric=metric,
                    time_column='predicted_stage_hpf',
                    genotype_column='genotype',
                    embryo_id_column='embryo_id',
                    ranking_window=ranking_window,
                    display_window=FULL_DISPLAY_WINDOW,
                    smooth_window=5,
                    cmap='viridis',
                    reverse_cmap=False,
                    figsize=(18, 8),
                    save_path=FIGURE_DIR / f'trajectories_{metric}_ranked_{min_t}_{max_t}.png'
                )

                # Save rankings table
                rankings_path = TABLE_DIR / f'rankings_{metric}_{min_t}_{max_t}.csv'
                rankings.to_csv(rankings_path, index=False)
                print(f"      Saved rankings: {rankings_path}")

                # Print summary statistics
                print(f"      Embryos ranked: {len(rankings)}")
                print(f"      Metric range: {rankings['avg_metric'].min():.3f} - {rankings['avg_metric'].max():.3f}")

            except Exception as e:
                print(f"      ERROR: {e}")
                continue

    print("\n" + "="*80)
    print("TRAJECTORY VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables:  {TABLE_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
