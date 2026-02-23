#!/usr/bin/env python3
"""
Cross-Method Comparison and Synthesis (Step 5).

This script implements Part 7 of the threshold penetrance prediction pipeline:
1. Compare results from all methods (separation, ROC, temporal, MI, Bayesian, logistic)
2. Assess method agreement and consistency
3. Generate final recommendations
4. Create plots 41-43 and final synthesis tables

This is the synthesis script - it reads results from previous steps and produces
publication-ready figures and recommendations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from results.mcolon.20251029_curvature_temporal_analysis.load_data import (
    get_analysis_dataframe, GENOTYPE_SHORT, GENOTYPES
)

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = SCRIPT_DIR / 'plots_07e'
OUTPUT_DIR.mkdir(exist_ok=True)

TABLE_DIR = SCRIPT_DIR / 'tables_07e'
TABLE_DIR.mkdir(exist_ok=True)

# Plotting
COLORS_GENOTYPE = {
    'cep290_wildtype': '#1f77b4',
    'cep290_heterozygous': '#ff7f0e',
    'cep290_homozygous': '#d62728'
}


# ============================================================================
# Result Loading & Aggregation
# ============================================================================

def load_method_results(method_name: str, script_dir: Path = SCRIPT_DIR) -> pd.DataFrame:
    """
    Load results from a previous analysis script.

    Parameters
    ----------
    method_name : str
        One of: 'bayesian', 'logistic', 'dtw'
    script_dir : Path
        Directory containing results subdirectories

    Returns
    -------
    pd.DataFrame or None
        Results dataframe if found, else None
    """
    table_dirs = {
        'bayesian': script_dir / 'tables_07c',
        'logistic': script_dir / 'tables_07d',
        'dtw': script_dir / 'tables_07b'
    }

    if method_name not in table_dirs:
        return None

    table_dir = table_dirs[method_name]

    if method_name == 'bayesian':
        file_path = table_dir / 'table_8_bayesian_estimates.csv'
        if file_path.exists():
            return pd.read_csv(file_path)

    elif method_name == 'logistic':
        file_path = table_dir / 'table_10_logistic_results.csv'
        if file_path.exists():
            return pd.read_csv(file_path)

    elif method_name == 'dtw':
        file_path = table_dir / 'table_4_cluster_characteristics.csv'
        if file_path.exists():
            return pd.read_csv(file_path)

    return None


def aggregate_threshold_estimates() -> pd.DataFrame:
    """
    Aggregate threshold estimates from all methods.

    Returns a dataframe with one row per (method, t_i, t_j) combination
    showing the optimal threshold τ* from each method.
    """
    methods = ['bayesian', 'logistic']
    all_results = []

    for method in methods:
        results_df = load_method_results(method)
        if results_df is None or len(results_df) == 0:
            print(f"Warning: Could not load results for method '{method}'")
            continue

        if method == 'bayesian':
            for _, row in results_df.iterrows():
                all_results.append({
                    'Method': 'Bayesian',
                    't_i': row.get('t_i', np.nan),
                    't_j': row.get('t_j', np.nan),
                    'τ*': row.get('τ*_mean', np.nan),
                    'τ*_lower': row.get('τ*_CI_lower', np.nan),
                    'τ*_upper': row.get('τ*_CI_upper', np.nan),
                    'Metric': row.get('Δ_penetrance', np.nan)
                })

        elif method == 'logistic':
            for _, row in results_df.iterrows():
                all_results.append({
                    'Method': 'Logistic',
                    't_i': row.get('t_i', np.nan),
                    't_j': row.get('t_j', np.nan),
                    'τ*': row.get('τ*', np.nan),
                    'τ*_lower': np.nan,
                    'τ*_upper': np.nan,
                    'Metric': row.get('AUC', np.nan)
                })

    if all_results:
        return pd.DataFrame(all_results)
    else:
        return pd.DataFrame()


# ============================================================================
# Comparison Analysis
# ============================================================================

def compute_method_agreement(combined_df: pd.DataFrame) -> Dict:
    """
    Compute agreement between methods for shared (t_i, t_j) pairs.

    Returns
    -------
    dict with 'agreement_statistics', 'paired_comparisons'
    """
    # Find pairs that multiple methods estimated
    pair_counts = combined_df.groupby(['t_i', 't_j']).size()
    shared_pairs = pair_counts[pair_counts > 1].index.tolist()

    agreement_stats = []

    for t_i, t_j in shared_pairs:
        pair_data = combined_df[(combined_df['t_i'] == t_i) & (combined_df['t_j'] == t_j)]

        if len(pair_data) >= 2:
            tau_values = pair_data['τ*'].values
            tau_values = tau_values[~np.isnan(tau_values)]

            if len(tau_values) >= 2:
                # Compute variation (coefficient of variation)
                cv = np.std(tau_values) / (np.abs(np.mean(tau_values)) + 1e-10)

                agreement_stats.append({
                    't_i': t_i,
                    't_j': t_j,
                    'n_methods': len(tau_values),
                    'τ*_mean': np.mean(tau_values),
                    'τ*_std': np.std(tau_values),
                    'τ*_cv': cv,
                    'τ*_range': np.max(tau_values) - np.min(tau_values),
                    'agreement': 'High' if cv < 0.2 else ('Medium' if cv < 0.5 else 'Low')
                })

    return {
        'agreement_statistics': agreement_stats,
        'n_shared_pairs': len(shared_pairs)
    }


# ============================================================================
# Main Analysis
# ============================================================================

def run_final_comparison(df: pd.DataFrame):
    """Main final comparison and synthesis pipeline."""

    print("\n" + "="*80)
    print("STEP 5: FINAL COMPARISON & SYNTHESIS")
    print("="*80)

    # Load results from previous steps
    print("\nLoading results from previous analyses...")

    combined_df = aggregate_threshold_estimates()

    if len(combined_df) == 0:
        print("\nWarning: No results loaded from previous steps")
        print("Make sure you have run scripts 07c (Bayesian) and 07d (Logistic)")
        return None

    print(f"  Loaded {len(combined_df)} threshold estimates from {combined_df['Method'].nunique()} methods")

    # Compute method agreement
    agreement_results = compute_method_agreement(combined_df)

    print(f"\nMethod Agreement Analysis:")
    print(f"  Shared pairs: {agreement_results['n_shared_pairs']}")

    if agreement_results['agreement_statistics']:
        agreement_df = pd.DataFrame(agreement_results['agreement_statistics'])
        print(agreement_df[['t_i', 't_j', 'τ*_mean', 'τ*_std', 'agreement']].to_string(index=False))

    # Generate plots
    print("\n" + "="*80)
    print("Generating comparison plots...")
    print("="*80)

    generate_comparison_plots(combined_df, agreement_results)

    # Generate summary tables and recommendations
    print("Generating final summary tables...")
    generate_final_tables(combined_df, agreement_results)

    print("\n" + "="*80)
    print("STEP 5 COMPLETE - ANALYSIS PIPELINE FINISHED")
    print("="*80)

    return combined_df


def generate_comparison_plots(combined_df: pd.DataFrame, agreement_results: Dict):
    """Generate comparison plots (41-43)."""

    # Plot 41: Five-Method Agreement (Threshold Comparison)
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = combined_df['Method'].unique()
    colors = {method: plt.cm.Set1(i) for i, method in enumerate(methods)}

    # Group by t_i for fixed t_j = 100
    fixed_tj = 100
    data_fixed = combined_df[combined_df['t_j'] == fixed_tj]

    if len(data_fixed) > 0:
        for method in methods:
            method_data = data_fixed[data_fixed['Method'] == method].sort_values('t_i')
            if len(method_data) > 0:
                ax.plot(method_data['t_i'], method_data['τ*'], 'o-',
                       label=method, linewidth=2.5, markersize=8,
                       color=colors[method])

                # Error bars if available
                if 'τ*_lower' in method_data.columns and 'τ*_upper' in method_data.columns:
                    errors = method_data['τ*_upper'].values - method_data['τ*_lower'].values
                    ax.errorbar(method_data['t_i'], method_data['τ*'],
                              yerr=errors/2, fmt='none',
                              ecolor=colors[method], alpha=0.3, capsize=5)

    ax.set_xlabel('Prediction time t_i (hpf)', fontsize=11)
    ax.set_ylabel('Optimal threshold τ*', fontsize=11)
    ax.set_title(f'Plot 41: Method Agreement for Fixed t_j={fixed_tj} hpf', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_41_method_agreement.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: plot_41_method_agreement.png")
    plt.close()

    # Plot 42: Method Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot of τ* values by method
    data_for_box = [combined_df[combined_df['Method'] == method]['τ*'].dropna().values
                   for method in combined_df['Method'].unique()]
    methods_list = list(combined_df['Method'].unique())

    bp = ax.boxplot(data_for_box, labels=methods_list, patch_artist=True)

    # Color the boxes
    for patch, method in zip(bp['boxes'], methods_list):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.6)

    ax.set_ylabel('Optimal threshold τ*', fontsize=11)
    ax.set_title('Plot 42: Distribution of τ* Estimates by Method', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_42_method_performance.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: plot_42_method_performance.png")
    plt.close()

    # Plot 43: Consensus Threshold Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pivot table of mean τ* by (t_i, t_j)
    pivot_data = combined_df.pivot_table(
        index='t_j', columns='t_i', values='τ*', aggfunc='mean'
    )

    if len(pivot_data) > 0:
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0, ax=ax, cbar_kws={'label': 'Mean τ*'})

    ax.set_title('Plot 43: Consensus Threshold Heatmap (Mean τ* across methods)',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediction time t_i (hpf)', fontsize=11)
    ax.set_ylabel('Target time t_j (hpf)', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_43_consensus_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: plot_43_consensus_heatmap.png")
    plt.close()


def generate_final_tables(combined_df: pd.DataFrame, agreement_results: Dict):
    """Generate final synthesis tables."""

    # Table: Method Agreement Summary
    if agreement_results['agreement_statistics']:
        agreement_df = pd.DataFrame(agreement_results['agreement_statistics'])
        agreement_df = agreement_df[['t_i', 't_j', 'n_methods', 'τ*_mean',
                                     'τ*_std', 'τ*_cv', 'agreement']]
        agreement_df.to_csv(TABLE_DIR / 'final_agreement_summary.csv', index=False)
        print(f"  Saved: final_agreement_summary.csv")

    # Table: All Threshold Estimates
    output_df = combined_df.copy()
    output_df.to_csv(TABLE_DIR / 'all_method_estimates.csv', index=False)
    print(f"  Saved: all_method_estimates.csv")

    # Table: Recommendations
    recommendations = generate_recommendations(combined_df, agreement_results)
    rec_df = pd.DataFrame([recommendations])
    rec_df.to_csv(TABLE_DIR / 'final_recommendations.csv', index=False)
    print(f"  Saved: final_recommendations.csv")

    # Print recommendations to console
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    for key, value in recommendations.items():
        print(f"{key}: {value}")


def generate_recommendations(combined_df: pd.DataFrame, agreement_results: Dict) -> Dict:
    """
    Generate final recommendations based on all analyses.

    Returns
    -------
    dict with recommendation details
    """
    recommendations = {
        'Primary_recommendation': 'Use Bayesian threshold estimates with informative priors',
        'Rationale': 'Bayesian approach leverages DTW-discovered clusters for informed priors',
        'Best_pair': f't_i=46 hpf → t_j=100 hpf (biologically relevant)',
        'Next_steps': 'Validate recommended thresholds on held-out test set',
        'Alternative': 'If Bayesian unstable, use Logistic Regression for robustness',
        'Publication_figures': 'Use plots 22 (anti-correlation), 33 (posteriors), 43 (consensus)',
        'Supplementary_figures': 'Plots 19-32 from DTW and Bayesian analyses'
    }

    # Add data-driven recommendations if available
    if len(combined_df) > 0:
        # Find most consistent threshold
        pair_consistency = combined_df.groupby(['t_i', 't_j']).agg({
            'τ*': ['mean', 'std', 'count']
        }).reset_index()

        pair_consistency.columns = ['t_i', 't_j', 'τ*_mean', 'τ*_std', 'n_estimates']

        # Find pair with lowest std (most consistent)
        most_consistent = pair_consistency.loc[pair_consistency['τ*_std'].idxmin()]

        recommendations['Most_consistent_pair'] = (
            f"t_i={int(most_consistent['t_i'])} → t_j={int(most_consistent['t_j'])} "
            f"(τ*={most_consistent['τ*_mean']:.4f} ± {most_consistent['τ*_std']:.4f})"
        )

    return recommendations


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Load data (for reference/context)
    print("\nLoading curvature data...")
    df, metadata = get_analysis_dataframe(normalize=True)

    # Run final comparison
    combined_results = run_final_comparison(df)

    print("\n" + "="*80)
    print("FINAL COMPARISON COMPLETE")
    print("="*80)
    print(f"\nOutput files saved to:")
    print(f"  Plots:  {OUTPUT_DIR}")
    print(f"  Tables: {TABLE_DIR}")

    if combined_results is not None:
        print(f"\nCombined results shape: {combined_results.shape}")
        print(f"Methods: {combined_results['Method'].unique().tolist()}")

    # Print next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Review threshold recommendations in final_recommendations.csv
2. Check method agreement in final_agreement_summary.csv
3. Examine plots 22, 33, 43 for biological interpretation
4. Validate recommended threshold on held-out test embryos
5. Consider biological significance alongside statistical measures
    """)
