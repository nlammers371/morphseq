"""
CEP290 Variance Decomposition - Plot Results from Cached Model

Loads cached Bambi model results and generates visualizations.
No MCMC fitting required - just loads and plots.

Usage:
    python cep290_plot_results.py [cache_name]

    cache_name: Optional. Defaults to most recent cache.
                Example: m10_c1.5_draws200_tune200_ta85

Author: Generated via Claude Code
Date: 2025-12-10
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import arviz as az

# =============================================================================
# Configuration
# =============================================================================

PAIRS = ['cep290_pair_1', 'cep290_pair_2', 'cep290_pair_3']
PAIR_COLORS = {
    'cep290_pair_1': '#1f77b4',
    'cep290_pair_2': '#ff7f0e',
    'cep290_pair_3': '#2ca02c'
}

OUTPUT_DIR = Path(__file__).parent / 'output_v2'
MODEL_CACHE_DIR = OUTPUT_DIR / 'model_cache'

# =============================================================================
# Load Cached Results
# =============================================================================

def find_latest_cache():
    """Find the most recent cache files."""
    nc_files = list(MODEL_CACHE_DIR.glob('idata_*.nc'))
    if not nc_files:
        raise FileNotFoundError(f"No cached models found in {MODEL_CACHE_DIR}")

    # Sort by modification time
    latest = max(nc_files, key=lambda p: p.stat().st_mtime)
    cache_name = latest.stem.replace('idata_', '')
    return cache_name


def load_cached_results(cache_name=None):
    """
    Load cached idata and model_df.

    Parameters
    ----------
    cache_name : str, optional
        Cache identifier (e.g., 'm10_c1.5_draws200_tune200_ta85')
        If None, uses most recent cache.

    Returns
    -------
    idata : arviz.InferenceData
        Posterior samples
    model_df : pd.DataFrame
        Model input data
    """
    if cache_name is None:
        cache_name = find_latest_cache()

    cache_path = MODEL_CACHE_DIR / f'idata_{cache_name}.nc'
    model_df_path = MODEL_CACHE_DIR / f'model_df_{cache_name}.pkl'

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    if not model_df_path.exists():
        raise FileNotFoundError(f"Model data not found: {model_df_path}")

    print(f"Loading cached model: {cache_name}")
    print(f"  idata: {cache_path}")
    print(f"  model_df: {model_df_path}")

    idata = az.from_netcdf(cache_path)
    with open(model_df_path, 'rb') as f:
        model_df = pickle.load(f)

    print(f"  Loaded {len(model_df)} observations, {model_df['unique_embryo_id'].nunique()} embryos")

    return idata, model_df, cache_name


# =============================================================================
# Variance Decomposition
# =============================================================================

def calculate_variance_components(idata, model_df):
    """
    Calculate variance explained by genetics, individual, and noise.

    Note: We need to reconstruct mu from the model components since
    it's not stored in the cached posterior.
    """
    print("\nCalculating variance components...")

    # Check if mu is in posterior_predictive (from model.predict())
    if 'mu' in idata.posterior_predictive:
        print("  Using mu from posterior_predictive...")
        mu_mean = idata.posterior_predictive['mu'].mean(dim=['chain', 'draw']).values
    else:
        # Reconstruct from HSGP weights and random intercepts
        print("  Reconstructing mu from model components...")

        # Get HSGP contribution (genetic)
        hsgp_key = [k for k in idata.posterior.data_vars if k.startswith('hsgp') and not ('sigma' in k or 'ell' in k or 'weights' in k or 'by' in k)][0]
        genetic_contribution = idata.posterior[hsgp_key].mean(dim=['chain', 'draw']).values

        # Get random intercepts (individual)
        random_intercepts = idata.posterior["1|unique_embryo_id"].mean(dim=["chain", "draw"])

        # Map intercepts to observations
        embryo_ids = model_df['unique_embryo_id'].values
        embryo_offsets = np.zeros(len(model_df))

        for i, uid in enumerate(embryo_ids):
            try:
                embryo_offsets[i] = random_intercepts.sel(unique_embryo_id__factor_dim=uid).values
            except KeyError:
                try:
                    embryo_offsets[i] = random_intercepts.sel(unique_embryo_id_dim=uid).values
                except KeyError:
                    dims = list(random_intercepts.dims)
                    embryo_offsets[i] = random_intercepts.sel({dims[0]: uid}).values

        # Reconstruct mu = genetic + individual
        if len(genetic_contribution.shape) == 1:
            mu_mean = genetic_contribution + embryo_offsets
        else:
            # Need to match dimensions
            mu_mean = genetic_contribution.flatten() + embryo_offsets

        print(f"    Reconstructed mu: shape={mu_mean.shape}")
        return calculate_variance_from_components(idata, model_df, mu_mean)

    # Original path if mu exists
    mu_mean = idata.posterior_predictive['mu'].mean(dim=['chain', 'draw']).values

    # Extract embryo random intercepts
    random_intercepts = idata.posterior["1|unique_embryo_id"].mean(dim=["chain", "draw"])

    # Map intercepts to observations
    embryo_ids = model_df['unique_embryo_id'].values
    embryo_offsets = np.zeros(len(model_df))

    for i, uid in enumerate(embryo_ids):
        try:
            embryo_offsets[i] = random_intercepts.sel(unique_embryo_id__factor_dim=uid).values
        except KeyError:
            try:
                embryo_offsets[i] = random_intercepts.sel(unique_embryo_id_dim=uid).values
            except KeyError:
                # Find the dimension name dynamically
                dims = list(random_intercepts.dims)
                embryo_offsets[i] = random_intercepts.sel({dims[0]: uid}).values

    # Calculate components
    genetic_signal = mu_mean - embryo_offsets
    residuals = model_df['curvature'].values - mu_mean

    # Variances
    var_genetics = np.var(genetic_signal)
    var_individual = np.var(embryo_offsets)
    var_noise = np.var(residuals)
    total_var = var_genetics + var_individual + var_noise

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
    print(f"  Genetics (Pair):     {var_genetics:.4f} ({results['pct_genetics']:.1%})")
    print(f"  Individual (Embryo): {var_individual:.4f} ({results['pct_individual']:.1%})")
    print(f"  Noise (Residual):    {var_noise:.4f} ({results['pct_noise']:.1%})")

    return results, genetic_signal, embryo_offsets, residuals, mu_mean


def calculate_variance_from_components(idata, model_df, mu_mean):
    """
    Helper function when we reconstruct mu from components.
    """
    # Extract embryo random intercepts
    random_intercepts = idata.posterior["1|unique_embryo_id"].mean(dim=["chain", "draw"])

    embryo_ids = model_df['unique_embryo_id'].values
    embryo_offsets = np.zeros(len(model_df))

    for i, uid in enumerate(embryo_ids):
        try:
            embryo_offsets[i] = random_intercepts.sel(unique_embryo_id__factor_dim=uid).values
        except KeyError:
            try:
                embryo_offsets[i] = random_intercepts.sel(unique_embryo_id_dim=uid).values
            except KeyError:
                dims = list(random_intercepts.dims)
                embryo_offsets[i] = random_intercepts.sel({dims[0]: uid}).values

    genetic_signal = mu_mean - embryo_offsets
    residuals = model_df['curvature'].values - mu_mean

    var_genetics = np.var(genetic_signal)
    var_individual = np.var(embryo_offsets)
    var_noise = np.var(residuals)
    total_var = var_genetics + var_individual + var_noise

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
    print(f"  Genetics (Pair):     {var_genetics:.4f} ({results['pct_genetics']:.1%})")
    print(f"  Individual (Embryo): {var_individual:.4f} ({results['pct_individual']:.1%})")
    print(f"  Noise (Residual):    {var_noise:.4f} ({results['pct_noise']:.1%})")

    return results, genetic_signal, embryo_offsets, residuals, mu_mean


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_variance_bar(results, output_path):
    """Plot variance decomposition as bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Genetics\n(Pair)', 'Individual\n(Embryo)', 'Noise\n(Residual)']
    percentages = [results['pct_genetics'] * 100,
                   results['pct_individual'] * 100,
                   results['pct_noise'] * 100]
    colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']

    bars = ax.bar(categories, percentages, color=colors, edgecolor='black', linewidth=1.5)

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title('Variance Decomposition\nCEP290 Curvature Trajectories', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(percentages) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_variance_pie(results, output_path):
    """Plot variance decomposition as pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = ['Genetics (Pair)', 'Individual (Embryo)', 'Noise']
    sizes = [results['pct_genetics'], results['pct_individual'], results['pct_noise']]
    colors = ['#1f77b4', '#ff7f0e', '#7f7f7f']

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12}
    )

    ax.set_title('Variance Decomposition\nCEP290 Curvature Trajectories', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_posterior_draws(idata, model_df, output_path, n_draws=50):
    """
    Plot posterior draws showing uncertainty in pair-specific curves.
    """
    print(f"\nGenerating posterior draws plot ({n_draws} draws)...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Check where mu is stored
    if 'mu' in idata.posterior_predictive:
        mu_posterior = idata.posterior_predictive['mu']
    elif hasattr(idata, 'predictions') and 'mu' in idata.predictions:
        mu_posterior = idata.predictions['mu']
    else:
        print("  WARNING: Cannot find mu in idata - skipping posterior draws plot")
        plt.close(fig)
        return

    n_total_draws = mu_posterior.shape[1]
    draw_indices = np.random.choice(n_total_draws, min(n_draws, n_total_draws), replace=False)

    for idx, pair in enumerate(PAIRS):
        ax = axes[idx]

        # Get data for this pair
        pair_mask = model_df['pair_id'] == pair
        pair_data = model_df[pair_mask].copy()
        pair_indices = np.where(pair_mask)[0]
        pair_times = model_df.loc[pair_mask, 'time'].values

        # Plot raw data
        for embryo_id in pair_data['unique_embryo_id'].unique():
            embryo_data = pair_data[pair_data['unique_embryo_id'] == embryo_id]
            ax.plot(embryo_data['time'], embryo_data['curvature'],
                   alpha=0.2, linewidth=0.5, color=PAIR_COLORS[pair])

        # Get unique times for aggregation
        unique_times = np.sort(pair_data['time'].unique())

        # Plot posterior draws
        for draw_idx in draw_indices:
            mu_draw = mu_posterior.isel(draw=draw_idx).mean(dim='chain').values
            mu_pair = mu_draw[pair_indices]

            # Aggregate by time
            time_means = []
            for t in unique_times:
                t_mask = pair_times == t
                time_means.append(np.mean(mu_pair[t_mask]))

            ax.plot(unique_times, time_means, alpha=0.15, linewidth=1, color='black')

        # Plot posterior mean
        mu_mean = mu_posterior.mean(dim=['chain', 'draw']).values
        mu_pair_mean = mu_mean[pair_indices]

        time_means_final = []
        for t in unique_times:
            t_mask = pair_times == t
            time_means_final.append(np.mean(mu_pair_mean[t_mask]))

        ax.plot(unique_times, time_means_final, linewidth=3, color='black', label='Posterior Mean')

        ax.set_title(f'{pair}\n{pair_data["unique_embryo_id"].nunique()} embryos', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hpf)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Curvature', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f'Posterior Draws from HSGP Model\n{n_draws} samples from posterior', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_component_decomposition(model_df, genetic_signal, embryo_offsets, residuals, full_prediction, output_path):
    """Plot decomposition of signal into components for sample embryos."""
    print("\nGenerating component decomposition plot...")

    np.random.seed(42)
    sample_embryos = []
    for pair in PAIRS:
        pair_embryos = model_df[model_df['pair_id'] == pair]['unique_embryo_id'].unique()
        sample_embryos.extend(np.random.choice(pair_embryos, min(2, len(pair_embryos)), replace=False))

    fig, axes = plt.subplots(len(sample_embryos), 1, figsize=(12, 2.5*len(sample_embryos)))

    if len(sample_embryos) == 1:
        axes = [axes]

    for idx, embryo_id in enumerate(sample_embryos):
        ax = axes[idx]

        mask = model_df['unique_embryo_id'] == embryo_id
        embryo_data = model_df[mask].sort_values('time')
        times = embryo_data['time'].values

        # Extract components
        genetic = genetic_signal[mask]
        individual = embryo_offsets[mask]
        observed = embryo_data['curvature'].values
        predicted = full_prediction[mask]

        # Plot
        ax.plot(times, observed, 'o', alpha=0.5, label='Observed', markersize=3)
        ax.plot(times, predicted, linewidth=2, label='Full Model', color='black')
        ax.plot(times, genetic, linewidth=2, label='Genetic (Pair)', linestyle='--')
        ax.axhline(individual[0], linewidth=2, label=f'Embryo Offset ({individual[0]:.3f})',
                  linestyle=':', color='orange')

        pair = embryo_data['pair_id'].iloc[0]
        ax.set_title(f'{embryo_id}', fontsize=10)
        ax.set_ylabel('Curvature', fontsize=9)
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (hpf)', fontsize=10)

    plt.suptitle('Variance Component Decomposition - Sample Embryos', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_trajectories_by_pair(model_df, output_path):
    """Plot raw trajectories colored by pair."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for pair in PAIRS:
        pair_df = model_df[model_df['pair_id'] == pair]
        for embryo_id in pair_df['unique_embryo_id'].unique():
            embryo_data = pair_df[pair_df['unique_embryo_id'] == embryo_id].sort_values('time')
            ax.plot(embryo_data['time'], embryo_data['curvature'],
                   alpha=0.3, linewidth=0.8, color=PAIR_COLORS[pair])

        # Mean trajectory
        mean_traj = pair_df.groupby('time')['curvature'].mean()
        ax.plot(mean_traj.index, mean_traj.values, linewidth=3, color=PAIR_COLORS[pair], label=pair)

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('Curvature', fontsize=12)
    ax.set_title('CEP290 Curvature Trajectories by Pair', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main(cache_name=None):
    print("="*60)
    print("CEP290 VARIANCE DECOMPOSITION - PLOT RESULTS")
    print("="*60)

    # Load cached results
    idata, model_df, cache_name = load_cached_results(cache_name)

    # Calculate variance components
    results, genetic_signal, embryo_offsets, residuals, full_prediction = calculate_variance_components(idata, model_df)

    # Generate all plots
    print("\nGenerating plots...")

    plot_trajectories_by_pair(model_df, OUTPUT_DIR / 'trajectories_by_pair.png')
    plot_variance_bar(results, OUTPUT_DIR / 'variance_bar.png')
    plot_variance_pie(results, OUTPUT_DIR / 'variance_pie.png')
    plot_posterior_draws(idata, model_df, OUTPUT_DIR / 'posterior_draws.png', n_draws=50)
    plot_component_decomposition(model_df, genetic_signal, embryo_offsets, residuals,
                                 full_prediction, OUTPUT_DIR / 'component_decomposition.png')

    # Save results CSV
    results_df = pd.DataFrame([results])
    results_df['cache_name'] = cache_name
    results_df.to_csv(OUTPUT_DIR / 'variance_decomposition_results.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'variance_decomposition_results.csv'}")

    # Summary
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Cache: {cache_name}")
    print(f"Results:")
    print(f"  Genetics (Pair):     {results['pct_genetics']:.1%}")
    print(f"  Individual (Embryo): {results['pct_individual']:.1%}")
    print(f"  Noise:               {results['pct_noise']:.1%}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == '__main__':
    cache_name = sys.argv[1] if len(sys.argv) > 1 else None
    main(cache_name)
