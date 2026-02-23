"""
CEP290 Variance Decomposition Analysis

Decomposes variance in curvature trajectories into:
- Genetic (Pair) component
- Individual (Embryo) component
- Noise/Residual component

Uses Hierarchical Bayesian Model with HSGP (Bambi/PyMC).

Usage:
    python cep290_variance_decomposition.py

Author: Generated via Claude Code
Date: 2025-12-10
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

# Direct import to avoid sklearn dependency in other modules
from pathlib import Path as _Path
def _load_qc_staged(experiment_id: str) -> pd.DataFrame:
    """Load build04 output (qc_staged) CSV files."""
    project_root = _Path(__file__).resolve().parents[3]
    qc_staged_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'
    qc_staged_path = qc_staged_dir / f'qc_staged_{experiment_id}.csv'
    if not qc_staged_path.exists():
        raise FileNotFoundError(f"QC staged data not found: {qc_staged_path}")
    print(f"  Loading qc_staged from: {qc_staged_path}")
    df = pd.read_csv(qc_staged_path)
    print(f"    Loaded: {len(df)} rows")
    return df

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_IDS = ['20251106', '20251112', '20251113']
PAIRS = ['cep290_pair_1', 'cep290_pair_2', 'cep290_pair_3']

OUTPUT_DIR = Path(__file__).parent / 'output'
MODEL_CACHE_DIR = OUTPUT_DIR / 'model_cache'

# Column names
TIME_COL = 'predicted_stage_hpf'
METRIC_COL = 'baseline_deviation_normalized'  # curvature
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'
GENOTYPE_COL = 'genotype'

MIN_TIMEPOINTS = 5

# =============================================================================
# Data Loading
# =============================================================================

def load_and_prepare_data():
    """
    Load data from multiple experiments and filter for CEP290 pairs.
    """
    print("Loading data from experiments...")

    dfs = []
    for exp_id in EXPERIMENT_IDS:
        print(f"  Loading {exp_id}...")
        try:
            df = _load_qc_staged(exp_id)
            df['experiment_id'] = exp_id
            dfs.append(df)
        except FileNotFoundError as e:
            print(f"    Warning: {e}")
            continue

    if not dfs:
        raise ValueError("No data loaded!")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(df)} rows")

    # Filter for valid embryos
    if 'use_embryo_flag' in df.columns:
        df = df[df['use_embryo_flag'] == 1].copy()
        print(f"  After use_embryo_flag filter: {len(df)} rows")

    # Filter for CEP290 pairs
    df = df[df[PAIR_COL].isin(PAIRS)].copy()
    print(f"  After pair filter: {len(df)} rows")

    # Drop rows with missing values
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, METRIC_COL])
    print(f"  After dropna: {len(df)} rows")

    # Create unique embryo ID (pair + embryo)
    df['unique_embryo_id'] = df[PAIR_COL].astype(str) + "_" + df[EMBRYO_ID_COL].astype(str)

    # Filter embryos with enough timepoints
    embryo_counts = df.groupby('unique_embryo_id').size()
    valid_embryos = embryo_counts[embryo_counts >= MIN_TIMEPOINTS].index
    df = df[df['unique_embryo_id'].isin(valid_embryos)].copy()
    print(f"  After min timepoints filter: {len(df)} rows")

    # Summary
    print(f"\nData summary:")
    print(f"  Unique embryos: {df['unique_embryo_id'].nunique()}")
    print(f"  Pairs: {df[PAIR_COL].unique().tolist()}")
    print(f"  Time range: {df[TIME_COL].min():.1f} - {df[TIME_COL].max():.1f} hpf")

    # Per-pair summary
    for pair in PAIRS:
        pair_df = df[df[PAIR_COL] == pair]
        n_embryos = pair_df['unique_embryo_id'].nunique()
        t_range = f"{pair_df[TIME_COL].min():.1f}-{pair_df[TIME_COL].max():.1f}"
        print(f"    {pair}: {n_embryos} embryos, {t_range} hpf")

    return df


# =============================================================================
# Model Fitting
# =============================================================================

def fit_variance_model(df, use_cached=True):
    """
    Fit Hierarchical Bayesian Model with HSGP using Bambi.

    Model: curvature ~ 0 + hsgp(time, by=pair_id) + (1|unique_embryo_id)

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    use_cached : bool
        If True, load cached model if available
    """
    import bambi as bmb
    import arviz as az
    import pickle

    print("\n" + "="*60)
    print("FITTING VARIANCE DECOMPOSITION MODEL")
    print("="*60)

    # Prepare data for Bambi
    model_df = df[[TIME_COL, METRIC_COL, PAIR_COL, 'unique_embryo_id']].copy()
    model_df = model_df.rename(columns={
        TIME_COL: 'time',
        METRIC_COL: 'curvature',
        PAIR_COL: 'pair_id'
    })

    print(f"\nModel data: {len(model_df)} observations")
    print(f"  Pairs: {model_df['pair_id'].nunique()}")
    print(f"  Embryos: {model_df['unique_embryo_id'].nunique()}")

    # Check for cached results
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MODEL_CACHE_DIR / 'idata.nc'
    model_df_cache = MODEL_CACHE_DIR / 'model_df.pkl'

    if use_cached and cache_path.exists() and model_df_cache.exists():
        print(f"\n✓ Found cached model at: {cache_path}")
        print("  Loading cached results (set use_cached=False to refit)...")
        idata = az.from_netcdf(cache_path)
        with open(model_df_cache, 'rb') as f:
            model_df = pickle.load(f)

        # Recreate model (lightweight, just for structure)
        model_formula = "curvature ~ 0 + hsgp(time, m=20, c=1.5, by=pair_id) + (1|unique_embryo_id)"
        model = bmb.Model(model_formula, data=model_df, dropna=True)

        print("  Loaded successfully!")
        print(az.summary(idata, var_names=['~hsgp']))
        return model, idata, model_df

    # Define model
    # hsgp(time, m=20, c=1.5, by=pair_id): Separate smooth curve for each pair (Genetics)
    #   m=20: number of basis functions for the HSGP approximation
    #   c=1.5: proportion factor extending boundary beyond data range
    # (1|unique_embryo_id): Random intercept for each embryo (Individual)
    # 0 +: Remove global intercept
    model_formula = "curvature ~ 0 + hsgp(time, m=20, c=1.5, by=pair_id) + (1|unique_embryo_id)"

    print(f"\nModel formula: {model_formula}")

    model = bmb.Model(model_formula, data=model_df, dropna=True)
    print("\nModel created. Starting MCMC sampling...")
    print("  Note: Increased target_accept=0.95 to reduce divergences")

    # Fit model with better sampling parameters
    idata = model.fit(
        draws=1000,
        tune=2000,              # Increased tune steps
        chains=4,
        target_accept=0.95,      # Increase to reduce divergences (default 0.8)
        max_treedepth=12,        # Increase if still hitting max depth
        random_seed=42,
        progressbar=True
    )

    print("\nSampling complete!")
    print(az.summary(idata, var_names=['~hsgp']))

    # Save results
    print(f"\nSaving model cache to: {MODEL_CACHE_DIR}")
    idata.to_netcdf(cache_path)
    with open(model_df_cache, 'wb') as f:
        pickle.dump(model_df, f)
    print("  ✓ Model saved!")

    return model, idata, model_df


# =============================================================================
# Variance Decomposition
# =============================================================================

def calculate_variance_components(model, idata, model_df):
    """
    Calculate variance explained by genetics, individual, and noise.
    """
    print("\n" + "="*60)
    print("VARIANCE DECOMPOSITION")
    print("="*60)

    # Get predictions
    preds = model.predict(idata, inplace=False)

    # Full prediction (genetic + individual effects)
    # The variable is called 'mu' not 'curvature_mean'
    full_prediction = preds.posterior["mu"].mean(dim=["chain", "draw"]).values

    # Extract embryo random intercepts
    random_intercepts = idata.posterior["1|unique_embryo_id"].mean(dim=["chain", "draw"])

    # Map intercepts back to observations
    embryo_ids = model_df['unique_embryo_id'].values
    unique_embryos = model_df['unique_embryo_id'].unique()

    embryo_offsets = np.zeros(len(model_df))
    for i, uid in enumerate(embryo_ids):
        try:
            embryo_offsets[i] = random_intercepts.sel(unique_embryo_id__factor_dim=uid).values
        except KeyError:
            # Try alternative dimension name
            embryo_offsets[i] = random_intercepts.sel(unique_embryo_id_dim=uid).values

    # Calculate components
    # Genetic signal = Full prediction - Embryo offset
    genetic_signal = full_prediction - embryo_offsets

    # Residuals = Raw data - Full prediction
    residuals = model_df['curvature'].values - full_prediction

    # Calculate variances
    var_genetics = np.var(genetic_signal)
    var_individual = np.var(embryo_offsets)
    var_noise = np.var(residuals)

    total_var = var_genetics + var_individual + var_noise

    # Results
    results = {
        'var_genetics': var_genetics,
        'var_individual': var_individual,
        'var_noise': var_noise,
        'total_var': total_var,
        'pct_genetics': var_genetics / total_var,
        'pct_individual': var_individual / total_var,
        'pct_noise': var_noise / total_var,
    }

    print(f"\nVariance Components:")
    print(f"  Genetics (Pair):    {var_genetics:.4f} ({results['pct_genetics']:.1%})")
    print(f"  Individual (Embryo): {var_individual:.4f} ({results['pct_individual']:.1%})")
    print(f"  Noise (Residual):    {var_noise:.4f} ({results['pct_noise']:.1%})")
    print(f"  Total:               {total_var:.4f}")

    return results, genetic_signal, embryo_offsets, residuals


# =============================================================================
# Visualization
# =============================================================================

def plot_trajectories_by_pair(df, output_path):
    """
    Plot raw trajectories colored by pair.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'cep290_pair_1': '#1f77b4', 'cep290_pair_2': '#ff7f0e', 'cep290_pair_3': '#2ca02c'}

    for pair in PAIRS:
        pair_df = df[df[PAIR_COL] == pair]
        for embryo_id in pair_df['unique_embryo_id'].unique():
            embryo_data = pair_df[pair_df['unique_embryo_id'] == embryo_id].sort_values(TIME_COL)
            ax.plot(embryo_data[TIME_COL], embryo_data[METRIC_COL],
                   alpha=0.3, linewidth=0.8, color=colors[pair])

        # Mean trajectory
        mean_traj = pair_df.groupby(TIME_COL)[METRIC_COL].mean()
        ax.plot(mean_traj.index, mean_traj.values,
               linewidth=3, color=colors[pair], label=pair)

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('Curvature (baseline_deviation_normalized)', fontsize=12)
    ax.set_title('CEP290 Curvature Trajectories by Pair', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_variance_pie(results, output_path):
    """
    Plot variance decomposition as pie chart.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = ['Genetics (Pair)', 'Individual (Embryo)', 'Noise']
    sizes = [results['pct_genetics'], results['pct_individual'], results['pct_noise']]
    colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12}
    )

    ax.set_title('Variance Decomposition\nCEP290 Curvature Trajectories',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_variance_bar(results, output_path):
    """
    Plot variance decomposition as bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Genetics\n(Pair)', 'Individual\n(Embryo)', 'Noise\n(Residual)']
    percentages = [results['pct_genetics'] * 100,
                   results['pct_individual'] * 100,
                   results['pct_noise'] * 100]
    colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']

    bars = ax.bar(categories, percentages, color=colors, edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title('Variance Decomposition\nCEP290 Curvature Trajectories',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(percentages) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*80)
    print("CEP290 VARIANCE DECOMPOSITION ANALYSIS")
    print("="*80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Pairs: {PAIRS}")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data()

    # Plot raw trajectories
    plot_trajectories_by_pair(df, OUTPUT_DIR / 'trajectories_by_pair.png')

    # Fit model (will use cached results if available)
    # Set use_cached=False to force refit
    model, idata, model_df = fit_variance_model(df, use_cached=True)

    # Calculate variance components
    results, genetic_signal, embryo_offsets, residuals = calculate_variance_components(
        model, idata, model_df
    )

    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(OUTPUT_DIR / 'variance_decomposition_results.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'variance_decomposition_results.csv'}")

    # Plot variance decomposition
    plot_variance_pie(results, OUTPUT_DIR / 'variance_pie.png')
    plot_variance_bar(results, OUTPUT_DIR / 'variance_bar.png')

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  Genetics (Pair):     {results['pct_genetics']:.1%}")
    print(f"  Individual (Embryo): {results['pct_individual']:.1%}")
    print(f"  Noise:               {results['pct_noise']:.1%}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
