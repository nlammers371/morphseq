"""
Earliest Predictive Timepoint Analysis

Predict future curvature (baseline_deviation_normalized) from embeddings at each
time bin to identify the earliest timepoint at which phenotypic outcomes become
predictable.

For each (genotype, start_time, target_time) combination, train a Ridge regression
model to predict curvature at target_time from embeddings at start_time. Generate
R² matrices showing prediction accuracy across all time pairs.

Author: Generated for experiment 20250512
Date: 2024-12-09
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

from analyze.trajectory_analysis.data_loading import load_experiment_dataframe
from analyze.difference_detection.horizon_plots import plot_horizon_grid


# =============================================================================
# Binning Utility (copied from analyze.utils.binning to avoid import issues)
# =============================================================================

def bin_embryos_by_time(df, time_col='predicted_stage_hpf', z_cols=None, bin_width=2.0, suffix='_binned'):
    """
    Bin VAE embeddings by predicted time and embryo.

    Copied from analyze.utils.binning.bin_embryos_by_time() to avoid import issues.
    """
    df = df.copy()

    # Detect latent columns
    if z_cols is None:
        z_cols = [c for c in df.columns if 'z_mu_n_' in c or 'z_mu_b' in c]
        if not z_cols:
            raise ValueError("No latent columns found. Please specify z_cols explicitly.")

    # Create time bins
    df['time_bin'] = (np.floor(df[time_col] / bin_width) * bin_width).astype(int)

    # Average latent vectors per embryo × time_bin
    agg = df.groupby(['embryo_id', 'time_bin'], as_index=False)[z_cols].mean()

    # Rename averaged latent columns
    agg.rename(columns={c: f'{c}{suffix}' for c in z_cols}, inplace=True)

    # Merge back non-latent metadata (take first unique per embryo)
    meta_cols = [c for c in df.columns if c not in z_cols + [time_col, 'time_bin']]
    meta_df = df[meta_cols].drop_duplicates(subset=['embryo_id'])

    # Merge metadata back in
    out = agg.merge(meta_df, on='embryo_id', how='left')

    # Ensure sorting
    out = out.sort_values(['embryo_id', 'time_bin']).reset_index(drop=True)

    return out


# =============================================================================
# Data Loading and Binning
# =============================================================================

def bin_data_for_prediction(df, bin_width=2.0, target_col='baseline_deviation_normalized'):
    """
    Bin embeddings and curvature by time for prediction analysis.

    Uses existing bin_embryos_by_time() to bin embeddings, then computes
    binned curvature (mean per time bin per embryo).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with embeddings (z_mu_n_*) and curvature column
    bin_width : float, default=2.0
        Width of time bins in hours
    target_col : str, default='baseline_deviation_normalized'
        Name of curvature column to bin

    Returns
    -------
    pd.DataFrame
        One row per (embryo_id, time_bin) with binned embeddings and curvature.
        Columns: embryo_id, time_bin, z_mu_n_*_binned, {target_col}_binned, genotype
    """
    # Bin embeddings using existing utility
    df_binned = bin_embryos_by_time(
        df,
        time_col='predicted_stage_hpf',
        bin_width=bin_width,
        suffix='_binned'
    )

    # Compute binned curvature (mean per time bin)
    df['time_bin'] = (np.floor(df['predicted_stage_hpf'] / bin_width) * bin_width).astype(int)

    curvature_binned = (
        df.groupby(['embryo_id', 'time_bin'], as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: f'{target_col}_binned'})
    )

    # Merge binned curvature with binned embeddings
    df_binned = df_binned.merge(curvature_binned, on=['embryo_id', 'time_bin'], how='left')

    return df_binned


# =============================================================================
# Time Pair Generation
# =============================================================================

def get_forward_time_pairs(time_bins):
    """
    Generate all forward time pairs where target > start.

    Parameters
    ----------
    time_bins : array-like
        List of unique time bins (e.g., [10, 12, 14, 16, ...])

    Returns
    -------
    list of tuple
        List of (start_time, target_time) pairs where target > start.
        Example: [(10, 12), (10, 14), (10, 16), ..., (12, 14), (12, 16), ...]
    """
    time_bins = sorted(time_bins)
    pairs = []

    for i, start_time in enumerate(time_bins):
        for target_time in time_bins[i+1:]:  # Only future times
            pairs.append((start_time, target_time))

    return pairs


# =============================================================================
# Ridge Regression for Single Time Pair
# =============================================================================

def fit_single_time_pair(df_binned, start_time, target_time,
                         latent_suffix='_binned',
                         target_col='baseline_deviation_normalized_binned',
                         n_folds=5,
                         alpha_grid=(0.01, 0.1, 1.0, 10.0, 50.0, 100.0),
                         min_samples=10,
                         eval_by_genotype=False):
    """
    Train Ridge regression predicting curvature at target_time from embeddings at start_time.

    Uses GroupKFold cross-validation by embryo_id to evaluate prediction accuracy.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned data with one row per (embryo_id, time_bin).
        Can contain multiple genotypes for training.
    start_time : float
        Time bin for predictor embeddings
    target_time : float
        Time bin for target curvature (must be > start_time)
    latent_suffix : str, default='_binned'
        Suffix for binned latent column names (e.g., z_mu_n_00_binned)
    target_col : str, default='baseline_deviation_normalized_binned'
        Name of binned curvature column to predict
    n_folds : int, default=5
        Number of cross-validation folds (adapted if fewer embryos)
    alpha_grid : tuple, default=(0.01, 0.1, 1.0, 10.0, 50.0, 100.0)
        Ridge regularization parameters to try
    min_samples : int, default=10
        Minimum number of samples required to run regression
    eval_by_genotype : bool, default=False
        If True, compute separate metrics for each genotype in test set

    Returns
    -------
    dict
        Results with keys: r2_mean, r2_std, mae_mean, rmse_mean,
        n_embryos, n_folds, alpha_median.
        If eval_by_genotype=True, also includes genotype-specific MSE metrics.
        Returns NaN values if insufficient samples.
    """
    # Extract data at start and target times
    df_start = df_binned[df_binned['time_bin'] == start_time].copy()
    df_target = df_binned[df_binned['time_bin'] == target_time][['embryo_id', target_col]].copy()
    df_target.rename(columns={target_col: 'y_target'}, inplace=True)

    # Inner join: only embryos present at BOTH timepoints
    df_joined = df_start.merge(df_target, on='embryo_id', how='inner')

    # Check if we have enough samples
    if len(df_joined) < min_samples:
        return {
            'r2_mean': np.nan,
            'r2_std': np.nan,
            'mae_mean': np.nan,
            'rmse_mean': np.nan,
            'n_embryos': len(df_joined),
            'n_folds': 0,
            'alpha_median': np.nan,
        }

    # Prepare features (binned embeddings) and target
    latent_cols = [c for c in df_joined.columns if c.startswith('z_mu_n_') and c.endswith(latent_suffix)]
    X = df_joined[latent_cols].to_numpy(dtype=float)
    y = df_joined['y_target'].to_numpy(dtype=float)
    groups = df_joined['embryo_id'].to_numpy()

    # Check for valid number of unique embryos
    n_unique_embryos = len(np.unique(groups))
    if n_unique_embryos < 2:
        return {
            'r2_mean': np.nan,
            'r2_std': np.nan,
            'mae_mean': np.nan,
            'rmse_mean': np.nan,
            'n_embryos': n_unique_embryos,
            'n_folds': 0,
            'alpha_median': np.nan,
        }

    # Adapt n_folds if necessary
    n_folds_actual = min(n_folds, n_unique_embryos)

    # GroupKFold cross-validation
    gkf = GroupKFold(n_splits=n_folds_actual)
    fold_metrics = []

    # For genotype-specific evaluation
    genotype_mses = {} if eval_by_genotype else None

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build Ridge regression pipeline (following curvature_regression.py pattern)
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=alpha_grid, cv=4))  # Inner CV for alpha selection
        ])

        # Fit and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Compute overall metrics
        fold_metrics.append({
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mse': mean_squared_error(y_test, y_pred, squared=True),
            'alpha': pipeline.named_steps['ridge'].alpha_,
        })

        # Compute genotype-specific MSE and R² if requested
        if eval_by_genotype and 'genotype' in df_joined.columns:
            test_genotypes = df_joined.iloc[test_idx]['genotype'].values
            for genotype in np.unique(test_genotypes):
                geno_mask = test_genotypes == genotype
                # Only compute metrics if at least 2 samples for valid R² score
                if geno_mask.sum() >= 2:
                    geno_mse = mean_squared_error(y_test[geno_mask], y_pred[geno_mask], squared=True)
                    geno_r2 = r2_score(y_test[geno_mask], y_pred[geno_mask])

                    if genotype not in genotype_mses:
                        genotype_mses[genotype] = []
                    genotype_mses[genotype].append(geno_mse)

                    # Also track R² for each genotype
                    r2_key = f'r2_{genotype}'
                    if r2_key not in genotype_mses:  # Reuse dict structure
                        genotype_mses[r2_key] = []
                    genotype_mses[r2_key].append(geno_r2)

    # Aggregate metrics across folds
    df_folds = pd.DataFrame(fold_metrics)

    result = {
        'r2_mean': df_folds['r2'].mean(),
        'r2_std': df_folds['r2'].std(ddof=0),
        'mae_mean': df_folds['mae'].mean(),
        'rmse_mean': df_folds['rmse'].mean(),
        'mse_mean': df_folds['mse'].mean(),
        'n_embryos': n_unique_embryos,
        'n_folds': n_folds_actual,
        'alpha_median': df_folds['alpha'].median(),
    }

    # Add genotype-specific MSE and R² if computed
    if eval_by_genotype and genotype_mses:
        for key, value_list in genotype_mses.items():
            if key.startswith('r2_'):
                # R² values
                result[key] = np.mean(value_list)
            else:
                # MSE values
                result[f'mse_{key}'] = np.mean(value_list)
                result[f'n_{key}'] = len(value_list)

    return result


# =============================================================================
# Run All Time Pairs
# =============================================================================

def run_all_time_pairs(df_binned, n_folds=5, target_col='baseline_deviation_normalized_binned',
                       train_genotypes=None, eval_by_genotype=False):
    """
    Run fit_single_time_pair for all time pairs.

    Can train on multiple genotypes together and evaluate metrics separately by genotype.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned data with embeddings and curvature
    n_folds : int, default=5
        Number of cross-validation folds
    target_col : str, default='baseline_deviation_normalized_binned'
        Name of binned curvature column
    train_genotypes : list of str, optional
        Genotypes to include in training. If None, trains separately on each genotype.
    eval_by_genotype : bool, default=False
        If True and train_genotypes is provided, compute separate MSE for each genotype.

    Returns
    -------
    pd.DataFrame
        Results with columns: start_time, target_time, r2_mean, r2_std,
        mae_mean, rmse_mean, mse_mean, n_embryos, n_folds, alpha_median.
        If eval_by_genotype=True, also includes mse_{genotype} columns.
    """
    # Get unique time bins
    time_bins = sorted(df_binned['time_bin'].unique())

    # Generate forward time pairs
    time_pairs = get_forward_time_pairs(time_bins)

    # Filter to training genotypes if specified
    if train_genotypes is not None:
        df_train = df_binned[df_binned['genotype'].isin(train_genotypes)].copy()
        print(f"Training on genotypes: {train_genotypes}")
        print(f"Evaluating by genotype: {eval_by_genotype}")
    else:
        df_train = df_binned.copy()
        print(f"Training separately on each genotype")

    print(f"Running predictions for {len(time_pairs)} time pairs...")
    print(f"Time bins: {time_bins}")
    print(f"Total samples: {len(df_train)}")

    # Run regression for each time pair
    results = []

    for pair_idx, (start_time, target_time) in enumerate(time_pairs):
        # Print progress
        if (pair_idx + 1) % 50 == 0 or (pair_idx + 1) == len(time_pairs):
            print(f"  Progress: {pair_idx + 1}/{len(time_pairs)} time pairs...")

        # Fit model for this time pair
        result = fit_single_time_pair(
            df_train,
            start_time,
            target_time,
            target_col=target_col,
            n_folds=n_folds,
            eval_by_genotype=eval_by_genotype
        )

        # Add identifiers
        result['start_time'] = start_time
        result['target_time'] = target_time

        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Reorder columns for readability
    base_cols = ['start_time', 'target_time', 'r2_mean', 'r2_std',
                 'mae_mean', 'rmse_mean', 'mse_mean', 'n_embryos', 'n_folds', 'alpha_median']

    # Add genotype-specific metrics if present
    genotype_cols = [
        c for c in results_df.columns
        if (c.startswith('mse_') or c.startswith('r2_') or c.startswith('n_')) and c not in base_cols
    ]
    col_order = base_cols + genotype_cols

    # Only keep columns that exist
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]

    print(f"\nCompleted! Generated {len(results_df)} prediction results.")
    print(f"R² range: [{results_df['r2_mean'].min():.3f}, {results_df['r2_mean'].max():.3f}]")
    if 'mse_mean' in results_df.columns:
        try:
            print(f"MSE range: [{results_df['mse_mean'].min():.6f}, {results_df['mse_mean'].max():.6f}]")
        except:
            pass

    return results_df


# =============================================================================
# Build R² Matrices
# =============================================================================

def build_r2_matrix(results_df, metric='r2_mean'):
    """
    Pivot results into 2D matrix (start × target).

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-format results from run_all_time_pairs()
    metric : str, default='r2_mean'
        Column name to pivot (e.g., 'r2_mean', 'mse_mean', 'mse_cep290_wildtype')

    Returns
    -------
    pd.DataFrame
        Matrix with start_times as index, target_times as columns, metric values.
        Upper triangle only (NaN for target <= start).
    """
    matrix = results_df.pivot_table(
        index='start_time',
        columns='target_time',
        values=metric,
        aggfunc='mean'  # In case of duplicates
    )

    # Ensure sorted
    matrix = matrix.sort_index().sort_index(axis=1)

    return matrix


# =============================================================================
# Visualization
# =============================================================================

def plot_r2_horizons(r2_matrices, save_path, title='Predictive R²',
                     clip_percentiles=(5, 95)):
    """
    Generate horizon plots using existing plot_horizon_grid() from horizon_plots.py.

    Parameters
    ----------
    r2_matrices : dict
        Dict mapping label -> R² DataFrame (start × target).
        Can be genotypes or metric types.
    save_path : str or Path
        Path to save figure
    title : str, default='Predictive R²'
        Figure title
    clip_percentiles : tuple, default=(5, 95)
        (low, high) percentiles to clip colorscale.
        Set to None to use full data range.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    # Convert to nested structure expected by plot_horizon_grid()
    # Structure: {row_label: {col_label: matrix}}
    nested_matrices = {
        label: {'value': matrix}
        for label, matrix in r2_matrices.items()
    }

    # Get ordered names
    label_order = sorted(r2_matrices.keys())

    # Generate plot
    fig = plot_horizon_grid(
        matrices=nested_matrices,
        row_labels=label_order,
        col_labels=['value'],
        metric='value',
        cmap='viridis',
        clip_percentiles=clip_percentiles,
        title=title,
        figsize=(10, 4 * len(label_order)),
        save_path=save_path,
        dpi=300,
    )

    return fig


def plot_predictions_from_start_time(results_df, start_times=[14, 30], save_path=None):
    """
    Plot R² and MSE for predictions starting from specific timepoints.

    For each start_time in start_times, shows how well the model predicts
    to all future target times. Useful for understanding predictability windows.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_all_time_pairs()
    start_times : list of float, default=[14, 30]
        Starting timepoints to visualize predictions from
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure with R² and MSE subplots
    """
    # Determine how many genotypes we have
    r2_cols = [c for c in results_df.columns if c.startswith('r2_cep290')]
    mse_cols = [c for c in results_df.columns if c.startswith('mse_cep290')]
    genotypes = sorted(set(c.replace('r2_', '').replace('mse_', '') for c in r2_cols + mse_cols))

    # Create figure with subplots
    n_starts = len(start_times)
    fig, axes = plt.subplots(2, n_starts, figsize=(6*n_starts, 10))

    # Ensure axes is 2D array
    if n_starts == 1:
        axes = axes.reshape(2, 1)

    # Color map for genotypes
    colors = plt.cm.Set2(np.linspace(0, 1, len(genotypes)))
    color_map = {geno: colors[i] for i, geno in enumerate(genotypes)}

    for col_idx, start_time in enumerate(start_times):
        df_start = results_df[results_df['start_time'] == start_time].copy()

        if df_start.empty:
            axes[0, col_idx].text(0.5, 0.5, f'No data for start_time={start_time}',
                                   ha='center', va='center', transform=axes[0, col_idx].transAxes)
            axes[1, col_idx].text(0.5, 0.5, f'No data for start_time={start_time}',
                                   ha='center', va='center', transform=axes[1, col_idx].transAxes)
            continue

        # Plot R² for each genotype
        ax_r2 = axes[0, col_idx]
        for r2_col in r2_cols:
            genotype = r2_col.replace('r2_', '')
            if r2_col in df_start.columns:
                valid = df_start[df_start[r2_col].notna()]
                if not valid.empty:
                    ax_r2.plot(valid['target_time'], valid[r2_col],
                              marker='o', label=genotype, linewidth=2, markersize=6,
                              color=color_map[genotype])

        ax_r2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax_r2.set_xlabel('Target Time (hpf)', fontsize=11)
        ax_r2.set_ylabel('R²', fontsize=11)
        ax_r2.set_title(f'R² from start={start_time}h', fontsize=12, fontweight='bold')
        ax_r2.legend(fontsize=9)
        ax_r2.grid(True, alpha=0.3)

        # Plot MSE for each genotype
        ax_mse = axes[1, col_idx]
        for mse_col in mse_cols:
            genotype = mse_col.replace('mse_', '')
            if mse_col in df_start.columns:
                valid = df_start[df_start[mse_col].notna()]
                if not valid.empty:
                    ax_mse.plot(valid['target_time'], valid[mse_col],
                               marker='s', label=genotype, linewidth=2, markersize=6,
                               color=color_map[genotype])

        ax_mse.set_xlabel('Target Time (hpf)', fontsize=11)
        ax_mse.set_ylabel('MSE', fontsize=11)
        ax_mse.set_title(f'MSE from start={start_time}h', fontsize=12, fontweight='bold')
        ax_mse.legend(fontsize=9)
        ax_mse.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_earliest_predictive(results_df, r2_threshold=0.3, save_path=None):
    """
    For each target time, find earliest start time with R² > threshold.

    Shows "earliest predictive timepoint" curve by genotype.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_all_time_pairs()
    r2_threshold : float, default=0.3
        Minimum R² to consider prediction valid
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for genotype in sorted(results_df['genotype'].unique()):
        df_geno = results_df[results_df['genotype'] == genotype]

        earliest = []
        for target_time in sorted(df_geno['target_time'].unique()):
            df_target = df_geno[df_geno['target_time'] == target_time]
            df_target = df_target[df_target['r2_mean'] >= r2_threshold]

            if not df_target.empty:
                earliest_start = df_target['start_time'].min()
                earliest.append((target_time, earliest_start))

        if earliest:
            targets, starts = zip(*earliest)
            ax.plot(targets, starts, marker='o', label=genotype.replace('_', ' '), linewidth=2, markersize=6)

    # Add identity line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Identity (no prediction)')

    ax.set_xlabel('Target Time (hpf)', fontsize=12)
    ax.set_ylabel('Earliest Predictive Start Time (hpf)', fontsize=12)
    ax.set_title(f'Earliest Predictive Timepoint (R² ≥ {r2_threshold})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main execution function for earliest predictive timepoint analysis.

    Trains on wildtype + heterozygous + homozygous together,
    then evaluates MSE and R² separately for each genotype to see prediction differences.
    Generates horizon plots and specific start-time prediction curves (14h, 30h).
    """
    # Configuration
    EXPERIMENT_ID = '20250512'
    BIN_WIDTH = 2.0
    N_FOLDS = 5
    TARGET_COL = 'baseline_deviation_normalized'
    TRAIN_GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']  # Include homozygous

    OUTPUT_DIR = Path(__file__).parent / 'output'
    FIGURES_DIR = OUTPUT_DIR / 'figures'

    print("="*80)
    print("Earliest Predictive Timepoint Analysis")
    print("="*80)
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Bin width: {BIN_WIDTH}h")
    print(f"CV folds: {N_FOLDS}")
    print(f"Target metric: {TARGET_COL}")
    print(f"Training genotypes: {TRAIN_GENOTYPES}")
    print("="*80 + "\n")

    # Step 1: Load data
    print("Step 1: Loading experiment data...")
    df = load_experiment_dataframe(EXPERIMENT_ID, format_version='df03')
    print(f"  Loaded {len(df)} rows")
    print(f"  Genotypes: {df['genotype'].unique()}")
    print(f"  Time range: {df['predicted_stage_hpf'].min():.1f} - {df['predicted_stage_hpf'].max():.1f} hpf\n")

    # Step 2: Bin data
    print("Step 2: Binning data...")
    df_binned = bin_data_for_prediction(df, bin_width=BIN_WIDTH, target_col=TARGET_COL)
    print(f"  Binned to {len(df_binned)} rows (embryo × time_bin)")
    print(f"  Time bins: {sorted(df_binned['time_bin'].unique())}")

    # Show genotype counts
    print(f"\n  Genotype counts:")
    for geno in sorted(df_binned['genotype'].unique()):
        n_embryos = df_binned[df_binned['genotype'] == geno]['embryo_id'].nunique()
        print(f"    {geno}: {n_embryos} embryos")
    print()

    # Step 3: Run predictions (train on wildtype + het, evaluate by genotype)
    print("Step 3: Running time-pair predictions...")
    print("  Training on wildtype + heterozygous together")
    print("  Evaluating MSE separately for each genotype\n")

    results_df = run_all_time_pairs(
        df_binned,
        n_folds=N_FOLDS,
        target_col=f'{TARGET_COL}_binned',
        train_genotypes=TRAIN_GENOTYPES,
        eval_by_genotype=True
    )

    # Step 4: Save results
    print("\nStep 4: Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT_DIR / 'results_all_pairs.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  Saved results to: {results_path}")

    # Step 5: Build matrices for R² and genotype-specific MSE
    print("\nStep 5: Building matrices...")

    # R² matrix (overall)
    r2_matrix = build_r2_matrix(results_df, metric='r2_mean')
    r2_matrix.to_csv(OUTPUT_DIR / 'r2_matrix_overall.csv')
    print(f"  Saved overall R² matrix")

    # R² matrices by genotype
    r2_matrices_by_genotype = {}
    r2_cols = [c for c in results_df.columns if c.startswith('r2_cep290')]
    for r2_col in r2_cols:
        genotype = r2_col.replace('r2_', '')
        r2_matrices_by_genotype[genotype] = build_r2_matrix(results_df, metric=r2_col)
        r2_matrices_by_genotype[genotype].to_csv(OUTPUT_DIR / f'r2_matrix_{genotype}.csv')
        print(f"  Saved R² matrix for {genotype}")

    # MSE matrices by genotype
    mse_matrices = {}
    mse_cols = [c for c in results_df.columns if c.startswith('mse_cep290')]
    for mse_col in mse_cols:
        genotype = mse_col.replace('mse_', '')
        mse_matrices[genotype] = build_r2_matrix(results_df, metric=mse_col)
        mse_matrices[genotype].to_csv(OUTPUT_DIR / f'mse_matrix_{genotype}.csv')
        print(f"  Saved MSE matrix for {genotype}")

    # Step 6: Generate visualizations
    print("\nStep 6: Generating visualizations...")

    # Overall R² horizon plot (clipped to reasonable range)
    horizon_r2_overall_path = FIGURES_DIR / 'r2_horizon_overall.png'
    plot_r2_horizons(
        {'Overall R²': r2_matrix},
        save_path=horizon_r2_overall_path,
        title='Overall Predictive R² (Wildtype + Het Model)',
        clip_percentiles=(5, 95)
    )
    print(f"  Saved overall R² horizon plot to: {horizon_r2_overall_path}")

    # R² horizon plots by genotype
    if r2_matrices_by_genotype:
        horizon_r2_by_geno_path = FIGURES_DIR / 'r2_horizon_by_genotype.png'
        plot_r2_horizons(
            r2_matrices_by_genotype,
            save_path=horizon_r2_by_geno_path,
            title='R² by Genotype (Wildtype + Het Model)',
            clip_percentiles=(5, 95)
        )
        print(f"  Saved R² by genotype horizon plot to: {horizon_r2_by_geno_path}")

    # MSE horizon plots by genotype
    if mse_matrices:
        horizon_mse_path = FIGURES_DIR / 'mse_horizon_by_genotype.png'
        plot_r2_horizons(
            mse_matrices,
            save_path=horizon_mse_path,
            title='MSE by Genotype (Wildtype + Het Model)',
            clip_percentiles=(5, 95)
        )
        print(f"  Saved MSE horizon plot to: {horizon_mse_path}")

    # Predictions from specific start times (14h and 30h)
    start_time_pred_path = FIGURES_DIR / 'predictions_from_start_times.png'
    plot_predictions_from_start_time(
        results_df,
        start_times=[14, 30],
        save_path=start_time_pred_path
    )
    print(f"  Saved predictions from start times (14h, 30h) plot to: {start_time_pred_path}")

    # Step 7: Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    df_valid = results_df[results_df['r2_mean'].notna()]

    print(f"\nOverall Model Performance:")
    print(f"  Valid predictions: {len(df_valid)}/{len(results_df)}")
    if len(df_valid) > 0:
        print(f"  R² range: [{df_valid['r2_mean'].min():.3f}, {df_valid['r2_mean'].max():.3f}]")
        print(f"  Mean R²: {df_valid['r2_mean'].mean():.3f} ± {df_valid['r2_mean'].std():.3f}")
        if 'mse_mean' in df_valid.columns:
            try:
                print(f"  Mean MSE: {df_valid['mse_mean'].mean():.6f}")
            except:
                pass

        # Best prediction
        best_idx = df_valid['r2_mean'].idxmax()
        best = df_valid.loc[best_idx]
        print(f"  Best R²: {best['start_time']:.0f}h → {best['target_time']:.0f}h (R² = {best['r2_mean']:.3f})")

    # Genotype-specific R² and MSE
    print(f"\nGenotype-Specific Metrics:")
    for r2_col in r2_cols:
        genotype = r2_col.replace('r2_', '')
        df_with_r2 = df_valid[df_valid[r2_col].notna()]
        if len(df_with_r2) > 0:
            mean_r2 = df_with_r2[r2_col].mean()
            print(f"  {genotype} R²: {mean_r2:.3f} (mean across {len(df_with_r2)} time pairs)")

    for mse_col in mse_cols:
        genotype = mse_col.replace('mse_', '')
        df_with_mse = df_valid[df_valid[mse_col].notna()]
        if len(df_with_mse) > 0:
            mean_mse = df_with_mse[mse_col].mean()
            print(f"  {genotype} MSE: {mean_mse:.6f} (mean across {len(df_with_mse)} time pairs)")

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
