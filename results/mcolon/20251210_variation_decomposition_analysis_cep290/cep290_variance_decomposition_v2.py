"""
CEP290 Variance Decomposition Analysis - V2 (Fast Test Version)

Faster version for testing with:
- Reduced draws (500 instead of 1000)
- Parameter-tagged cache names
- Posterior draw visualizations

Decomposes variance in curvature trajectories into:
- Genetic (Pair) component
- Individual (Embryo) component
- Noise/Residual component

Uses Hierarchical Bayesian Model with HSGP (Bambi/PyMC).

Usage:
    python cep290_variance_decomposition_v2.py

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

# Direct implementation of _load_df03_format to avoid sklearn dependency
def _load_df03_format(experiment_id: str) -> pd.DataFrame:
    """
    Load df03 format CSVs - merges curvature from body_axis with metadata from build06.
    This includes pair information for all experiments.
    """
    search_paths = [
        (project_root / 'morphseq_playground' / 'metadata' / 'body_axis' / 'summary',
         project_root / 'morphseq_playground' / 'metadata' / 'build06_output'),
        (project_root / 'morphseq_playground' / 'metadata' / 'body_axis' / 'summary',
         project_root / 'morphseq_playground' / 'metadata' / 'body_axis' / 'summary'),
    ]

    curv_path = None
    meta_path = None

    curv_patterns = [
        f'curvature_metrics_{experiment_id}.csv',
        f'{experiment_id}_curvature.csv',
    ]

    meta_patterns = [
        f'df03_final_output_with_latents_{experiment_id}.csv',
        f'{experiment_id}_metadata.csv',
    ]

    for curv_dir, meta_dir in search_paths:
        if curv_path is None:
            for pattern in curv_patterns:
                candidate = curv_dir / pattern
                if candidate.exists():
                    curv_path = candidate
                    break

        if meta_path is None:
            for pattern in meta_patterns:
                candidate = meta_dir / pattern
                if candidate.exists():
                    meta_path = candidate
                    break

        if curv_path and meta_path:
            break

    if not curv_path:
        raise FileNotFoundError(f"Curvature data not found for {experiment_id}")

    if not meta_path:
        raise FileNotFoundError(f"Metadata not found for {experiment_id}")

    print(f"  Loading curvature from: {curv_path}")
    print(f"  Loading metadata from: {meta_path}")

    df_curv = pd.read_csv(curv_path)
    df_meta = pd.read_csv(meta_path)

    print(f"    Curvature: {len(df_curv)} rows")
    print(f"    Metadata: {len(df_meta)} rows")

    # Merge on snip_id
    if 'snip_id' in df_curv.columns and 'snip_id' in df_meta.columns:
        df_merged = df_curv.merge(df_meta, on='snip_id', how='inner')
        print(f"    Merged on 'snip_id': {len(df_merged)} rows")
    elif 'embryo_id' in df_curv.columns and 'embryo_id' in df_meta.columns:
        df_merged = df_curv.merge(df_meta, on='embryo_id', how='inner')
        print(f"    Merged on 'embryo_id': {len(df_merged)} rows")
    else:
        raise ValueError(f"Could not find common merge key")

    if len(df_merged) == 0:
        raise ValueError("Merge resulted in empty dataframe")

    return df_merged

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_IDS = ['20251106', '20251112', '20251113']
PAIRS = ['cep290_pair_1', 'cep290_pair_2', 'cep290_pair_3']

OUTPUT_DIR = Path(__file__).parent / 'output_v2'
MODEL_CACHE_DIR = OUTPUT_DIR / 'model_cache'

# Column names
TIME_COL = 'predicted_stage_hpf'
METRIC_COL = 'baseline_deviation_normalized'  # curvature
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'
GENOTYPE_COL = 'genotype'

MIN_TIMEPOINTS = 5

# MCMC Parameters (FAST for testing)
MCMC_PARAMS = {
    'draws': 200,           # Very reduced for quick test
    'tune': 200,            # Reduced tuning (was stuck here)
    'chains': 2,            # Only 2 chains for speed
    'target_accept': 0.85,  # Lower to allow bigger steps
    'max_treedepth': 10,    # Default
    'random_seed': 42,
}

# NOTE: If rerunning after code changes, delete old cache or change parameters
# to force new cache creation with predictions included

# HSGP Parameters
HSGP_PARAMS = {
    'm': 10,      # Even fewer basis functions
    'c': 1.5,
}

# =============================================================================
# Data Loading
# =============================================================================

def load_and_prepare_data():
    """
    Load data from multiple experiments and filter for CEP290 pairs.
    Uses df03_format which merges build06 metadata (with pair info) with curvature.
    """
    print("Loading data from experiments...")

    dfs = []
    for exp_id in EXPERIMENT_IDS:
        print(f"  Loading {exp_id}...")
        try:
            df = _load_df03_format(exp_id)
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

def get_cache_name():
    """Generate cache name based on parameters."""
    param_str = f"m{HSGP_PARAMS['m']}_c{HSGP_PARAMS['c']}_draws{MCMC_PARAMS['draws']}_tune{MCMC_PARAMS['tune']}_ta{int(MCMC_PARAMS['target_accept']*100)}"
    return param_str


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
    print("FITTING VARIANCE DECOMPOSITION MODEL (V2 - FAST)")
    print("="*60)
    print(f"MCMC Params: {MCMC_PARAMS}")
    print(f"HSGP Params: {HSGP_PARAMS}")

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

    # Check for cached results with parameter-specific name
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = get_cache_name()
    cache_path = MODEL_CACHE_DIR / f'idata_{cache_name}.nc'
    model_df_cache = MODEL_CACHE_DIR / f'model_df_{cache_name}.pkl'

    if use_cached and cache_path.exists() and model_df_cache.exists():
        print(f"\n✓ Found cached model: {cache_name}")
        print(f"  Path: {cache_path}")
        print("  Loading cached results (set use_cached=False to refit)...")
        idata = az.from_netcdf(cache_path)
        with open(model_df_cache, 'rb') as f:
            model_df = pickle.load(f)

        # Recreate model (lightweight, just for structure)
        model_formula = f"curvature ~ 0 + hsgp(time, m={HSGP_PARAMS['m']}, c={HSGP_PARAMS['c']}, by=pair_id) + (1|unique_embryo_id)"
        model = bmb.Model(model_formula, data=model_df, dropna=True)

        print("  Loaded successfully!")
        print(az.summary(idata, var_names=['~hsgp']))
        return model, idata, model_df

    # Define model
    model_formula = f"curvature ~ 0 + hsgp(time, m={HSGP_PARAMS['m']}, c={HSGP_PARAMS['c']}, by=pair_id) + (1|unique_embryo_id)"

    print(f"\nModel formula: {model_formula}")

    model = bmb.Model(model_formula, data=model_df, dropna=True)
    print("\nModel created. Starting MCMC sampling...")

    # Fit model
    idata = model.fit(
        draws=MCMC_PARAMS['draws'],
        tune=MCMC_PARAMS['tune'],
        chains=MCMC_PARAMS['chains'],
        target_accept=MCMC_PARAMS['target_accept'],
        max_treedepth=MCMC_PARAMS['max_treedepth'],
        random_seed=MCMC_PARAMS['random_seed'],
        progressbar=True
    )

    print("\nSampling complete!")
    print(az.summary(idata, var_names=['~hsgp']))

    # Generate predictions (adds 'mu' to posterior_predictive)
    print("\nGenerating predictions for variance decomposition...")
    model.predict(idata, kind="mean", inplace=True)
    print("  ✓ Predictions added to idata")

    # Save results with parameter-specific name
    print(f"\nSaving model cache: {cache_name}")
    print(f"  Path: {MODEL_CACHE_DIR}")
    idata.to_netcdf(cache_path)
    with open(model_df_cache, 'wb') as f:
        pickle.dump(model_df, f)
    print("  ✓ Model saved with predictions!")

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

    return results, genetic_signal, embryo_offsets, residuals, full_prediction


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def plot_posterior_draws(model, idata, model_df, output_path, n_draws=50):
    """
    Plot posterior draws of pair-specific curves.

    Shows uncertainty in the fitted curves by plotting multiple samples
    from the posterior distribution using the HSGP weights directly.
    """
    print(f"\nGenerating posterior draws plot ({n_draws} draws)...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    colors = {'cep290_pair_1': '#1f77b4', 'cep290_pair_2': '#ff7f0e', 'cep290_pair_3': '#2ca02c'}

    for idx, pair in enumerate(PAIRS):
        ax = axes[idx]

        # Get data for this pair
        pair_mask = model_df['pair_id'] == pair
        pair_data = model_df[pair_mask].sort_values('time')

        # Plot raw data points
        for embryo_id in pair_data['unique_embryo_id'].unique():
            embryo_data = pair_data[pair_data['unique_embryo_id'] == embryo_id]
            ax.plot(embryo_data['time'], embryo_data['curvature'],
                   alpha=0.2, linewidth=0.5, color=colors[pair])

        # Get predictions on the original data (not new grid)
        # This avoids the "new groups" error
        preds = model.predict(idata, inplace=False)
        mu_posterior = preds.posterior['mu']

        # Sample random draws
        n_chains = mu_posterior.shape[0]
        n_total_draws = mu_posterior.shape[1]
        draw_indices = np.random.choice(n_total_draws, min(n_draws, n_total_draws), replace=False)

        # Get mean trajectory per timepoint for this pair
        pair_indices = np.where(pair_mask)[0]
        pair_times = model_df.loc[pair_mask, 'time'].values

        # For plotting, we'll aggregate predictions at each unique time
        unique_times = np.sort(pair_data['time'].unique())

        for draw_idx in draw_indices:
            # Get predictions for this draw
            mu_draw = mu_posterior.isel(draw=draw_idx).mean(dim='chain').values
            mu_pair = mu_draw[pair_indices]

            # Aggregate by time (mean across embryos at each timepoint)
            time_means = []
            for t in unique_times:
                t_mask = pair_times == t
                time_means.append(np.mean(mu_pair[t_mask]))

            ax.plot(unique_times, time_means, alpha=0.15, linewidth=1, color='black')

        # Plot mean prediction
        mu_mean = mu_posterior.mean(dim=['chain', 'draw']).values
        mu_pair_mean = mu_mean[pair_indices]

        time_means_final = []
        for t in unique_times:
            t_mask = pair_times == t
            time_means_final.append(np.mean(mu_pair_mean[t_mask]))

        ax.plot(unique_times, time_means_final, linewidth=3, color='black', label='Posterior Mean')

        ax.set_title(f'{pair}\n{pair_data["unique_embryo_id"].nunique()} embryos',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hpf)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Curvature', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f'Posterior Draws from HSGP Model\n{n_draws} samples from posterior',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_component_decomposition(model_df, genetic_signal, embryo_offsets, residuals, full_prediction, output_path):
    """
    Plot the decomposition of signal into components for a sample of embryos.
    """
    print("\nGenerating component decomposition plot...")

    # Select 3 random embryos per pair
    np.random.seed(42)
    sample_embryos = []
    for pair in PAIRS:
        pair_embryos = model_df[model_df['pair_id'] == pair]['unique_embryo_id'].unique()
        sample_embryos.extend(np.random.choice(pair_embryos, min(3, len(pair_embryos)), replace=False))

    fig, axes = plt.subplots(len(sample_embryos), 1, figsize=(12, 3*len(sample_embryos)))

    if len(sample_embryos) == 1:
        axes = [axes]

    for idx, embryo_id in enumerate(sample_embryos):
        ax = axes[idx]

        # Get data for this embryo
        mask = model_df['unique_embryo_id'] == embryo_id
        embryo_data = model_df[mask].sort_values('time')
        times = embryo_data['time'].values

        # Extract components
        genetic = genetic_signal[mask]
        individual = embryo_offsets[mask]
        noise = residuals[mask]
        observed = embryo_data['curvature'].values
        predicted = full_prediction[mask]

        # Plot
        ax.plot(times, observed, 'o', alpha=0.5, label='Observed', markersize=3)
        ax.plot(times, predicted, linewidth=2, label='Full Model', color='black')
        ax.plot(times, genetic, linewidth=2, label='Genetic (Pair)', linestyle='--')
        ax.axhline(individual[0], linewidth=2, label=f'Individual Offset ({individual[0]:.3f})',
                  linestyle=':', color='orange')

        pair = embryo_data['pair_id'].iloc[0]
        ax.set_title(f'{embryo_id} ({pair})', fontsize=10)
        ax.set_ylabel('Curvature', fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (hpf)', fontsize=10)

    plt.suptitle('Variance Component Decomposition\nSample Embryos', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*80)
    print("CEP290 VARIANCE DECOMPOSITION ANALYSIS - V2 (FAST)")
    print("="*80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Pairs: {PAIRS}")
    print(f"MCMC: {MCMC_PARAMS}")
    print(f"HSGP: {HSGP_PARAMS}")
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
    results, genetic_signal, embryo_offsets, residuals, full_prediction = calculate_variance_components(
        model, idata, model_df
    )

    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(OUTPUT_DIR / 'variance_decomposition_results.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'variance_decomposition_results.csv'}")

    # Plot variance decomposition
    plot_variance_pie(results, OUTPUT_DIR / 'variance_pie.png')
    plot_variance_bar(results, OUTPUT_DIR / 'variance_bar.png')

    # Plot posterior draws
    plot_posterior_draws(model, idata, model_df, OUTPUT_DIR / 'posterior_draws.png', n_draws=50)

    # Plot component decomposition
    plot_component_decomposition(model_df, genetic_signal, embryo_offsets, residuals,
                                full_prediction, OUTPUT_DIR / 'component_decomposition.png')

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  Genetics (Pair):     {results['pct_genetics']:.1%}")
    print(f"  Individual (Embryo): {results['pct_individual']:.1%}")
    print(f"  Noise:               {results['pct_noise']:.1%}")
    print(f"\nCache name: {get_cache_name()}")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
