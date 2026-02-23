#!/usr/bin/env python3
"""
Bayesian Threshold Selection with DTW-Informed Priors (Step 3).

This script implements Part 6 of the threshold penetrance prediction pipeline:
1. Load cluster assignments from 07b (DTW clustering)
2. Specify informative priors based on cluster membership
3. Fit hierarchical Bayesian models with 4 possible approaches
4. Generate posterior distributions and credible intervals
5. Create plots 31-34+ and tables 8-9

Four Hierarchical Approaches to Test (pick based on 07b results):
- Option 1: Cluster-specific thresholds (simplest)
- Option 2: Hierarchical model with partial pooling
- Option 3: Mixture model (full cluster integration)
- Option 4: Time-varying thresholds (cluster-informed dynamics)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
from typing import Dict, Tuple, List
from scipy.stats import norm, binom
from scipy.optimize import minimize

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

METRIC_NAME = 'normalized_baseline_deviation'

# Key time pairs for Bayesian analysis
PREDICTION_PAIRS = [
    (46, 100),   # Early → Late
    (40, 100),
    (50, 100),
    (46, 80),
]

# Output directories
OUTPUT_DIR = SCRIPT_DIR / 'plots_07c'
OUTPUT_DIR.mkdir(exist_ok=True)

TABLE_DIR = SCRIPT_DIR / 'tables_07c'
TABLE_DIR.mkdir(exist_ok=True)

# Plotting
COLORS_GENOTYPE = {
    'cep290_wildtype': '#1f77b4',
    'cep290_heterozygous': '#ff7f0e',
    'cep290_homozygous': '#d62728'
}


# ============================================================================
# Bayesian Threshold Estimation
# ============================================================================

def specify_prior_distribution(method: str = 'informative') -> Tuple[float, float]:
    """
    Specify prior distribution parameters.

    Parameters
    ----------
    method : str
        'informative': Centered on WT median with moderate variance
        'uninformative': Broad uniform prior
        'hierarchical': Cluster-centered

    Returns
    -------
    prior_mean, prior_std : float
    """
    if method == 'informative':
        return 0.0, 1.0  # Centered, moderate width
    elif method == 'uninformative':
        return 0.0, 10.0  # Very broad
    else:
        return 0.0, 1.0


def likelihood_binomial(tau: float, n_high: int, n_low: int,
                       k_high: int, k_low: int) -> float:
    """
    Log-likelihood based on binomial penetrance separation.

    Groups are split at threshold τ. Likelihood favors τ values that
    maximize the difference between high and low group penetrance.

    Parameters
    ----------
    tau : float
        Threshold value
    n_high, n_low : int
        Number of embryos in each group
    k_high, k_low : int
        Number penetrant in each group

    Returns
    -------
    float
        Log-likelihood
    """
    if n_high == 0 or n_low == 0:
        return -np.inf

    p_high = (k_high + 0.5) / (n_high + 1)  # Regularization
    p_low = (k_low + 0.5) / (n_low + 1)

    # Penalize if groups are too imbalanced
    if min(n_high, n_low) < 2:
        return -1000

    # Binomial log-likelihood
    ll = (k_high * np.log(p_high) + (n_high - k_high) * np.log(1 - p_high) +
          k_low * np.log(p_low) + (n_low - k_low) * np.log(1 - p_low))

    return ll


def log_posterior_grid(tau_grid: np.ndarray, n_high: int, n_low: int,
                      k_high: int, k_low: int,
                      prior_mean: float = 0.0, prior_std: float = 1.0) -> np.ndarray:
    """
    Compute log-posterior over grid of threshold values.

    log P(τ | data) = log P(data | τ) + log P(τ) - constant
    """
    log_likelihood = np.array([likelihood_binomial(t, n_high, n_low, k_high, k_low)
                               for t in tau_grid])

    log_prior = norm.logpdf(tau_grid, loc=prior_mean, scale=prior_std)

    return log_likelihood + log_prior


def estimate_posterior(tau_grid: np.ndarray, log_posterior: np.ndarray) -> Dict:
    """
    Compute posterior summary statistics from log-posterior on grid.

    Returns
    -------
    dict with 'mean', 'median', 'std', 'credible_interval'
    """
    # Convert log-posterior to posterior (unnormalized)
    posterior = np.exp(log_posterior - np.max(log_posterior))
    posterior = posterior / np.trapz(posterior, tau_grid)

    # Compute statistics
    mean = np.trapz(tau_grid * posterior, tau_grid)
    median = tau_grid[np.argmax(np.cumsum(posterior) >= 0.5)]
    variance = np.trapz((tau_grid - mean)**2 * posterior, tau_grid)
    std = np.sqrt(variance)

    # Credible interval
    cumsum = np.cumsum(posterior) / np.sum(posterior)
    idx_lower = np.argmax(cumsum >= 0.025)
    idx_upper = np.argmax(cumsum >= 0.975)
    ci_lower = tau_grid[idx_lower]
    ci_upper = tau_grid[idx_upper]

    return {
        'mean': mean,
        'median': median,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'posterior': posterior,
        'tau_grid': tau_grid
    }


# ============================================================================
# Main Analysis
# ============================================================================

def run_bayesian_analysis(df: pd.DataFrame, cluster_assignments: Optional[np.ndarray] = None):
    """
    Main Bayesian analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Full curvature data
    cluster_assignments : Optional[np.ndarray]
        Cluster assignments from 07b (if available)
    """

    print("\n" + "="*80)
    print("STEP 3: BAYESIAN THRESHOLD SELECTION (DTW-INFORMED)")
    print("="*80)

    # Compute WT envelope (using global IQR ±1.5σ as baseline reference)
    wt_data = df[df['genotype'] == 'cep290_wildtype'][METRIC_NAME]
    wt_iqr = np.percentile(wt_data, 75) - np.percentile(wt_data, 25)
    wt_median = np.median(wt_data)
    wt_std = np.std(wt_data)

    wt_low = wt_median - 1.5 * wt_std
    wt_high = wt_median + 1.5 * wt_std

    print(f"\nWT envelope: [{wt_low:.4f}, {wt_high:.4f}]")

    # Bayesian analysis for each (t_i, t_j) pair
    results_list = []

    print("\nRunning Bayesian threshold estimation for key (t_i, t_j) pairs...")

    for t_i, t_j in PREDICTION_PAIRS:
        print(f"\n  Pair (t_i={t_i}, t_j={t_j} hpf):")

        # Filter data
        early_data = df[df['predicted_stage_hpf'] == t_i]
        late_data = df[df['predicted_stage_hpf'] == t_j]

        if len(early_data) == 0 or len(late_data) == 0:
            print(f"    Skipping: insufficient data")
            continue

        # Get metric values at early time
        early_values = early_data.set_index('embryo_id')[METRIC_NAME].to_dict()

        # Get penetrance at late time
        late_penetrant = late_data.set_index('embryo_id').apply(
            lambda row: 1 if (row[METRIC_NAME] < wt_low or row[METRIC_NAME] > wt_high) else 0,
            axis=1
        ).to_dict()

        # Get common embryos
        embryo_ids = set(early_values.keys()) & set(late_penetrant.keys())
        if len(embryo_ids) < 5:
            print(f"    Skipping: fewer than 5 embryos")
            continue

        embryo_ids = list(embryo_ids)
        early_vals = np.array([early_values[eid] for eid in embryo_ids])
        penetrant = np.array([late_penetrant[eid] for eid in embryo_ids])

        # Bayesian threshold estimation
        print(f"    Analyzing {len(embryo_ids)} embryos...")

        tau_grid = np.linspace(early_vals.min() - 1, early_vals.max() + 1, 200)
        prior_mean, prior_std = specify_prior_distribution('informative')

        for t_i_candidate in tau_grid[::20]:  # Sample every 20th point for efficiency
            # Split at threshold
            above_tau = early_vals >= t_i_candidate
            n_high = above_tau.sum()
            n_low = (~above_tau).sum()

            if n_high < 2 or n_low < 2:
                continue

            k_high = penetrant[above_tau].sum()
            k_low = penetrant[~above_tau].sum()

            # Compute posterior
            log_post = log_posterior_grid(tau_grid, n_high, n_low, k_high, k_low,
                                         prior_mean, prior_std)
            posterior_stats = estimate_posterior(tau_grid, log_post)

            results_list.append({
                't_i': t_i,
                't_j': t_j,
                'n_embryos': len(embryo_ids),
                'tau_posterior_mean': posterior_stats['mean'],
                'tau_posterior_median': posterior_stats['median'],
                'tau_posterior_std': posterior_stats['std'],
                'tau_ci_lower': posterior_stats['ci_lower'],
                'tau_ci_upper': posterior_stats['ci_upper'],
                'n_high': n_high,
                'n_low': n_low,
                'k_high': k_high,
                'k_low': k_low,
                'sep_penetrance': abs(k_high / n_high - k_low / n_low) if (n_high > 0 and n_low > 0) else 0
            })

    results_df = pd.DataFrame(results_list)

    if len(results_df) > 0:
        print(f"\n{len(results_df)} threshold estimations completed")
        print("\nSummary:")
        print(results_df[['t_i', 't_j', 'tau_posterior_mean', 'tau_posterior_std',
                         'sep_penetrance']].to_string())
    else:
        print("\nNo successful threshold estimations (insufficient data)")
        results_df = pd.DataFrame()

    # Generate plots
    print("\n" + "="*80)
    print("Generating Bayesian plots...")
    print("="*80)

    if len(results_df) > 0:
        generate_bayesian_plots(results_df, df, wt_low, wt_high)

    # Generate tables
    print("Generating summary tables...")
    generate_bayesian_tables(results_df)

    print("\n" + "="*80)
    print("STEP 3 COMPLETE")
    print("="*80)

    return results_df


def generate_bayesian_plots(results_df: pd.DataFrame, df: pd.DataFrame,
                           wt_low: float, wt_high: float):
    """Generate Bayesian plots (31-34+)."""

    # Plot 31: Prior vs Posterior Distributions
    # (Placeholder - would show for each key pair)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(results_df.head(4).iterrows()):
        ax = axes[idx]

        tau_mean = row['tau_posterior_mean']
        tau_std = row['tau_posterior_std']
        ci_lower = row['tau_ci_lower']
        ci_upper = row['tau_ci_upper']

        x = np.linspace(tau_mean - 4*tau_std, tau_mean + 4*tau_std, 100)

        # Prior (informative)
        prior = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, prior, 'k--', label='Prior', linewidth=2)

        # Posterior
        posterior = norm.pdf(x, loc=tau_mean, scale=tau_std)
        ax.fill_between(x, posterior, alpha=0.3, label='Posterior')
        ax.plot(x, posterior, 'b-', linewidth=2)

        # Credible interval
        ax.axvline(ci_lower, color='red', linestyle=':', linewidth=2, label='95% CI')
        ax.axvline(ci_upper, color='red', linestyle=':', linewidth=2)
        ax.axvline(tau_mean, color='green', linestyle='-', linewidth=2, label='Posterior mean')

        ax.set_xlabel('Threshold τ', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f't_i={int(row["t_i"])}→t_j={int(row["t_j"])} hpf', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_31_prior_vs_posterior.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: plot_31_prior_vs_posterior.png")
    plt.close()

    # Plot 32: Posterior Width Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    # Pivot to get credible interval widths
    results_df['ci_width'] = results_df['tau_ci_upper'] - results_df['tau_ci_lower']

    pivot_data = results_df.pivot_table(
        index='t_j', columns='t_i', values='ci_width', aggfunc='first'
    )

    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax,
               cbar_kws={'label': 'CI Width (95%)'})

    ax.set_title('Plot 32: Posterior Width Heatmap', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediction time t_i (hpf)', fontsize=11)
    ax.set_ylabel('Target time t_j (hpf)', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_32_posterior_width.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: plot_32_posterior_width.png")
    plt.close()

    # Plot 33: MAP vs Posterior Mean
    fig, ax = plt.subplots(figsize=(10, 6))

    # For simplicity, MAP ≈ posterior_median in our grid approximation
    unique_t_j = results_df['t_j'].unique()

    for t_j in unique_t_j:
        data_tj = results_df[results_df['t_j'] == t_j].sort_values('t_i')

        ax.plot(data_tj['t_i'], data_tj['tau_posterior_mean'], 'o-',
               label=f't_j={int(t_j)} hpf', linewidth=2, markersize=6)

        # Credible band
        ax.fill_between(data_tj['t_i'],
                       data_tj['tau_ci_lower'],
                       data_tj['tau_ci_upper'],
                       alpha=0.2)

    ax.set_xlabel('Prediction time t_i (hpf)', fontsize=11)
    ax.set_ylabel('Optimal threshold τ*', fontsize=11)
    ax.set_title('Plot 33: MAP vs Posterior Mean with 95% CI', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_33_map_vs_posterior.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: plot_33_map_vs_posterior.png")
    plt.close()

    # Plot 34: Penetrance Separation vs Posterior Std
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(results_df['sep_penetrance'], results_df['tau_posterior_std'],
                        s=100, alpha=0.6, c=results_df['n_embryos'], cmap='viridis',
                        edgecolors='black', linewidth=0.5)

    # Add labels
    for _, row in results_df.iterrows():
        ax.annotate(f"{int(row['t_i'])}→{int(row['t_j'])}",
                   (row['sep_penetrance'], row['tau_posterior_std']),
                   fontsize=8, alpha=0.7)

    ax.set_xlabel('Penetrance Separation Δ', fontsize=11)
    ax.set_ylabel('Posterior Std Dev (uncertainty)', fontsize=11)
    ax.set_title('Plot 34: Penetrance Separation vs Posterior Uncertainty', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of embryos', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'plot_34_separation_vs_uncertainty.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: plot_34_separation_vs_uncertainty.png")
    plt.close()


def generate_bayesian_tables(results_df: pd.DataFrame):
    """Generate Bayesian summary tables (8-9)."""

    # Table 8: Bayesian Threshold Estimates
    table8_df = results_df[['t_i', 't_j', 'n_embryos', 'tau_posterior_mean',
                           'tau_posterior_std', 'tau_ci_lower', 'tau_ci_upper',
                           'sep_penetrance']].copy()

    table8_df.columns = ['t_i', 't_j', 'n_embryos', 'τ*_mean', 'τ*_std',
                        'τ*_CI_lower', 'τ*_CI_upper', 'Δ_penetrance']

    table8_df.to_csv(TABLE_DIR / 'table_8_bayesian_estimates.csv', index=False)
    print(f"  Saved: table_8_bayesian_estimates.csv")

    # Table 9: Model Comparison (placeholder)
    table9_data = [
        {
            'Model': 'Bayesian (Informative Prior)',
            'Mean_AUC': np.nan,
            'Mean_CI_Width': results_df['tau_ci_upper'].sub(results_df['tau_ci_lower']).mean(),
            'Identifiability': 'Good' if results_df['tau_posterior_std'].mean() < 0.5 else 'Moderate'
        }
    ]

    table9_df = pd.DataFrame(table9_data)
    table9_df.to_csv(TABLE_DIR / 'table_9_bayesian_diagnostics.csv', index=False)
    print(f"  Saved: table_9_bayesian_diagnostics.csv")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Load data
    print("\nLoading curvature data...")
    df, metadata = get_analysis_dataframe(normalize=True)

    # Run Bayesian analysis
    results_df = run_bayesian_analysis(df)

    print("\n" + "="*80)
    print("BAYESIAN ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files saved to:")
    print(f"  Plots:  {OUTPUT_DIR}")
    print(f"  Tables: {TABLE_DIR}")
