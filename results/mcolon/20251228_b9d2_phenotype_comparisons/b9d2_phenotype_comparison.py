"""
B9D2 Phenotype Comparison Analysis

Compares b9d2 phenotypes (CE, HTA, BA-rescue) using difference detection to test
the hypothesis that HTA and BA-rescue are the same underlying phenotype that
diverges after 60 hpf.

Comparisons:
1. CE vs Wildtype
2. HTA vs Wildtype
3. BA-rescue vs Wildtype
4. HTA vs BA-rescue (key hypothesis test)
5. CE vs HTA

Usage:
    python b9d2_phenotype_comparison.py

Output:
    - output/classification_results/*.csv
    - output/figures/*_comprehensive.png

Author: Generated via Claude Code
Date: 2025-12-28
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Tuple, Optional

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe
# Note: We implement our own permutation test with progress tracking instead of using
# predictive_signal_test from src.analyze.difference_detection

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
OUTPUT_DIR = Path(__file__).parent / 'output_bin_width_4'
CLASSIFICATION_DIR = OUTPUT_DIR / 'classification_results'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Analysis parameters
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
GENOTYPE_COL = 'genotype'
BIN_WIDTH = 4.0  # hours (was 2.0, increased for speed)

# Metrics for visualization (phenotype-specific)
METRICS = {
    'CE': 'total_length_um',
    'HTA': 'baseline_deviation_normalized',
    'BA_rescue': 'baseline_deviation_normalized'
}

# Colors for plotting
COLOR_MAP = {
    'CE': '#d62728',           # Red
    'HTA': '#ff7f0e',          # Orange
    'BA_rescue': '#2ca02c',    # Green
    'wildtype': '#1f77b4',     # Blue
}

# Statistical parameters
N_SPLITS = 5      # was 5, reduced for speed
N_PERM = 100       # was 100, reduced for speed
RANDOM_STATE = 42

MIN_TIMEPOINTS = 3


# =============================================================================
# Phenotype File Parsing
# =============================================================================

def parse_phenotype_file(filepath: Path) -> List[str]:
    """
    Parse phenotype file to extract embryo IDs.

    Handles files with:
    - Simple list of embryo_ids (one per line)
    - Files with "b9d2_pair_X" headers (skip those lines)

    Parameters
    ----------
    filepath : Path
        Path to phenotype file

    Returns
    -------
    embryo_ids : List[str]
        List of embryo IDs
    """
    embryo_ids = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip pair headers
            if line.startswith('b9d2_pair'):
                continue

            # Skip comments
            if '#' in line:
                # Take only the part before the comment
                line = line.split('#')[0].strip()
                if not line:
                    continue

            embryo_ids.append(line)

    return embryo_ids


def load_all_phenotypes() -> Dict[str, List[str]]:
    """
    Load all phenotype embryo ID lists.

    Returns
    -------
    phenotypes : Dict[str, List[str]]
        Dictionary mapping phenotype name to list of embryo IDs
    """
    phenotypes = {
        'CE': parse_phenotype_file(CE_FILE),
        'HTA': parse_phenotype_file(HTA_FILE),
        'BA_rescue': parse_phenotype_file(BA_RESCUE_FILE),
    }

    print("Loaded phenotype lists:")
    for name, ids in phenotypes.items():
        print(f"  {name}: {len(ids)} embryos")

    return phenotypes


# =============================================================================
# Data Loading
# =============================================================================

def load_experiment_data() -> pd.DataFrame:
    """
    Load and combine data from both experiments.

    Returns
    -------
    df : pd.DataFrame
        Combined dataframe with all embryo data
    """
    print(f"\nLoading experiment data from {EXPERIMENT_IDS}...")

    dfs = []
    for exp_id in EXPERIMENT_IDS:
        print(f"  Loading {exp_id}...")
        df = load_experiment_dataframe(exp_id, format_version='df03')
        df['experiment_id'] = exp_id
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)

    # Handle column name variations for baseline deviation
    if 'baseline_deviation_normalized' not in df_combined.columns:
        if 'normalized_baseline_deviation' in df_combined.columns:
            df_combined['baseline_deviation_normalized'] = df_combined['normalized_baseline_deviation']
        elif 'baseline_deviation_um' in df_combined.columns and 'total_length_um' in df_combined.columns:
            # Normalize by total length if raw values exist
            df_combined['baseline_deviation_normalized'] = (
                df_combined['baseline_deviation_um'] / df_combined['total_length_um']
            )

    # Filter for valid embryos
    if 'use_embryo_flag' in df_combined.columns:
        df_combined = df_combined[df_combined['use_embryo_flag'] == 1].copy()

    print(f"  Loaded {len(df_combined)} rows, {df_combined[EMBRYO_ID_COL].nunique()} unique embryos")

    return df_combined


def extract_wildtype_embryos(df: pd.DataFrame, phenotype_dict: Dict[str, List[str]]) -> List[str]:
    """
    Extract wildtype embryo IDs from genotype column, excluding phenotype embryos.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe
    phenotype_dict : Dict[str, List[str]]
        Dictionary of phenotype embryo IDs

    Returns
    -------
    wildtype_ids : List[str]
        List of wildtype embryo IDs
    """
    # Get all embryos labeled as b9d2_wildtype
    wildtype_mask = df[GENOTYPE_COL] == 'b9d2_wildtype'
    wildtype_embryos = df[wildtype_mask][EMBRYO_ID_COL].unique().tolist()

    # Exclude any embryos that appear in phenotype lists
    all_phenotype_embryos = set()
    for phenotype_ids in phenotype_dict.values():
        all_phenotype_embryos.update(phenotype_ids)

    wildtype_ids = [eid for eid in wildtype_embryos if eid not in all_phenotype_embryos]

    print(f"\nWildtype embryos: {len(wildtype_ids)} (excluded {len(wildtype_embryos) - len(wildtype_ids)} phenotype embryos)")

    return wildtype_ids


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_comparison_data(
    df: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    group1_label: str,
    group2_label: str
) -> pd.DataFrame:
    """
    Prepare data for binary comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    group1_label : str
        Label for group 1
    group2_label : str
        Label for group 2

    Returns
    -------
    df_binned : pd.DataFrame
        Binned data ready for classification
    """
    # Filter to relevant embryos
    df_filtered = df[
        df[EMBRYO_ID_COL].isin(group1_ids + group2_ids)
    ].copy()

    # Create labels
    df_filtered['phenotype_label'] = df_filtered[EMBRYO_ID_COL].apply(
        lambda x: group1_label if x in group1_ids else group2_label
    )

    # Drop missing values in required columns
    required_cols = [EMBRYO_ID_COL, TIME_COL, 'phenotype_label']
    df_filtered = df_filtered.dropna(subset=required_cols)

    # Create time bins
    df_filtered['time_bin'] = (
        np.floor(df_filtered[TIME_COL] / BIN_WIDTH) * BIN_WIDTH
    ).astype(int)

    # Get VAE embedding columns (only biological features z_mu_b, NOT z_mu_n)
    z_cols = [c for c in df_filtered.columns if 'z_mu_b' in c]

    if len(z_cols) == 0:
        raise ValueError("No VAE embedding columns found (expected z_mu_b* columns)")

    # Bin embeddings: average per embryo x time_bin
    groupby_cols = [EMBRYO_ID_COL, 'time_bin', 'phenotype_label']
    df_binned = df_filtered.groupby(groupby_cols, as_index=False)[z_cols].mean()

    # Filter embryos with enough timepoints
    embryo_counts = df_binned.groupby(EMBRYO_ID_COL).size()
    valid_embryos = embryo_counts[embryo_counts >= MIN_TIMEPOINTS].index
    df_binned = df_binned[df_binned[EMBRYO_ID_COL].isin(valid_embryos)]

    print(f"  Prepared {len(df_binned)} binned samples from {df_binned[EMBRYO_ID_COL].nunique()} embryos")
    print(f"  Group 1 ({group1_label}): {df_binned[df_binned['phenotype_label']==group1_label][EMBRYO_ID_COL].nunique()} embryos")
    print(f"  Group 2 ({group2_label}): {df_binned[df_binned['phenotype_label']==group2_label][EMBRYO_ID_COL].nunique()} embryos")

    return df_binned


# =============================================================================
# Difference Detection
# =============================================================================

def run_difference_detection(
    df_binned: pd.DataFrame,
    comparison_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run AUROC-based permutation test for binary comparison with progress tracking.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned data with phenotype_label column
    comparison_name : str
        Name of comparison for logging

    Returns
    -------
    df_results : pd.DataFrame
        Classification results with AUROC and p-values per time bin
    df_embryo_probs : pd.DataFrame
        Per-embryo prediction probabilities
    """
    print(f"\n  Running difference detection for {comparison_name}...")

    # Get VAE embedding columns (only biological features z_mu_b, NOT z_mu_n)
    z_cols = [c for c in df_binned.columns if 'z_mu_b' in c]

    if len(z_cols) == 0:
        raise ValueError("No VAE embedding columns found in binned data")

    time_bins = sorted(df_binned['time_bin'].unique())
    n_time_bins = len(time_bins)
    print(f"  Using {len(z_cols)} VAE embedding columns")
    print(f"  Processing {n_time_bins} time bins ({N_PERM} permutations each)...")
    print(f"  Time bins: {time_bins[0]} to {time_bins[-1]} hpf")

    # Process each time bin with progress output
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(RANDOM_STATE)
    results = []
    embryo_predictions = []

    for i, t in enumerate(time_bins):
        # Progress indicator
        print(f"    [{i+1}/{n_time_bins}] Time bin {t} hpf...", end=' ', flush=True)

        sub = df_binned[df_binned['time_bin'] == t]
        X = sub[z_cols].values
        y = sub['phenotype_label'].values
        embryo_ids = sub['embryo_id'].values

        # Check class count
        classes = np.unique(y)
        if len(classes) != 2:
            print(f"skipped (need 2 classes, got {len(classes)})")
            continue

        # Check minimum samples per class
        class_counts = {c: np.sum(y == c) for c in classes}
        min_count = min(class_counts.values())
        if min_count < N_SPLITS:
            print(f"skipped (min class has {min_count} samples, need {N_SPLITS})")
            continue

        # Set up classifier
        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced' if True else None,
            random_state=RANDOM_STATE
        )

        # Cross-validated predictions for observed data
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        try:
            probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
            # Get probability of positive class (second unique class alphabetically)
            pos_class = classes[1]
            pos_idx = list(classes).index(pos_class)
            y_binary = (y == pos_class).astype(int)
            true_auroc = roc_auc_score(y_binary, probs[:, pos_idx])
        except Exception as e:
            print(f"error: {e}")
            continue

        # Permutation test
        null_aurocs = []
        for _ in range(N_PERM):
            y_perm = rng.permutation(y)
            y_perm_binary = (y_perm == pos_class).astype(int)
            try:
                probs_perm = cross_val_predict(clf, X, y_perm, cv=cv, method='predict_proba')
                null_aurocs.append(roc_auc_score(y_perm_binary, probs_perm[:, pos_idx]))
            except:
                continue

        # Compute p-value
        null_aurocs = np.array(null_aurocs)
        k = np.sum(null_aurocs >= true_auroc)
        pval = (k + 1) / (len(null_aurocs) + 1)

        results.append({
            'time_bin': t,
            'auroc_observed': true_auroc,
            'auroc_null_mean': np.mean(null_aurocs),
            'auroc_null_std': np.std(null_aurocs),
            'pval': pval,
            'n_samples': len(y)
        })

        # Store embryo predictions
        for eid, prob, true_label in zip(embryo_ids, probs[:, pos_idx], y):
            embryo_predictions.append({
                'embryo_id': eid,
                'time_bin': t,
                'true_label': true_label,
                'pred_proba_positive': prob,
                'positive_class': pos_class
            })

        print(f"AUROC={true_auroc:.3f}, p={pval:.3f}")

    df_results = pd.DataFrame(results)
    df_embryo_probs = pd.DataFrame(embryo_predictions) if embryo_predictions else None

    print(f"\n  Completed: {len(df_results)} time bins analyzed")

    return df_results, df_embryo_probs


# =============================================================================
# Divergence Computation
# =============================================================================

def compute_morphological_divergence(
    df: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    metric_col: str
) -> pd.DataFrame:
    """
    Compute mean morphological metric difference between groups over time.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    metric_col : str
        Morphological metric column name

    Returns
    -------
    divergence_df : pd.DataFrame
        Columns: hpf, group1_mean, group1_sem, group2_mean, group2_sem, abs_difference
    """
    # Filter to relevant embryos
    df_filtered = df[
        df[EMBRYO_ID_COL].isin(group1_ids + group2_ids)
    ].copy()

    # Add group labels
    df_filtered['group'] = df_filtered[EMBRYO_ID_COL].apply(
        lambda x: 'group1' if x in group1_ids else 'group2'
    )

    # Drop missing values
    df_filtered = df_filtered.dropna(subset=[TIME_COL, metric_col])

    # Interpolate trajectories to common grid (NO extrapolation - only within each embryo's range)
    grid_step = 0.5
    time_min = np.floor(df_filtered[TIME_COL].min() / grid_step) * grid_step
    time_max = np.ceil(df_filtered[TIME_COL].max() / grid_step) * grid_step
    common_grid = np.arange(time_min, time_max + grid_step, grid_step)

    # Interpolate each embryo
    interpolated_records = []
    for embryo_id in df_filtered[EMBRYO_ID_COL].unique():
        embryo_data = df_filtered[df_filtered[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)

        if len(embryo_data) < 2:
            continue

        group = embryo_data['group'].iloc[0]

        # Get this embryo's actual time range (no extrapolation!)
        embryo_time_min = embryo_data[TIME_COL].min()
        embryo_time_max = embryo_data[TIME_COL].max()

        # Interpolate only within embryo's time range
        interp_values = np.interp(
            common_grid,
            embryo_data[TIME_COL].values,
            embryo_data[metric_col].values
        )

        for t, v in zip(common_grid, interp_values):
            # Only keep values within embryo's actual time range
            if embryo_time_min <= t <= embryo_time_max:
                interpolated_records.append({
                    'embryo_id': embryo_id,
                    'hpf': t,
                    'metric_value': v,
                    'group': group
                })

    df_interp = pd.DataFrame(interpolated_records)

    # Compute stats per timepoint
    divergence_records = []
    for hpf in sorted(df_interp['hpf'].unique()):
        df_t = df_interp[df_interp['hpf'] == hpf]

        group1_values = df_t[df_t['group'] == 'group1']['metric_value'].values
        group2_values = df_t[df_t['group'] == 'group2']['metric_value'].values

        if len(group1_values) > 0 and len(group2_values) > 0:
            group1_mean = np.mean(group1_values)
            group1_sem = stats.sem(group1_values) if len(group1_values) > 1 else 0

            group2_mean = np.mean(group2_values)
            group2_sem = stats.sem(group2_values) if len(group2_values) > 1 else 0

            abs_diff = abs(group2_mean - group1_mean)

            divergence_records.append({
                'hpf': hpf,
                'group1_mean': group1_mean,
                'group1_sem': group1_sem,
                'group2_mean': group2_mean,
                'group2_sem': group2_sem,
                'abs_difference': abs_diff,
                'n_group1': len(group1_values),
                'n_group2': len(group2_values)
            })

    return pd.DataFrame(divergence_records)


# =============================================================================
# Visualization
# =============================================================================

def create_comprehensive_figure(
    df_results: pd.DataFrame,
    divergence_df: pd.DataFrame,
    df_raw: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    group1_label: str,
    group2_label: str,
    metric_col: str,
    metric_label: str,
    save_path: Path
):
    """
    Create 3-panel comprehensive figure.

    Parameters
    ----------
    df_results : pd.DataFrame
        Classification results
    divergence_df : pd.DataFrame
        Morphological divergence over time
    df_raw : pd.DataFrame
        Raw data for individual trajectories
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    group1_label : str
        Label for group 1
    group2_label : str
        Label for group 2
    metric_col : str
        Morphological metric column
    metric_label : str
        Label for metric axis
    save_path : Path
        Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Filter for valid results
    df_res = df_results[df_results['auroc_observed'].notna()].copy()

    # Get time range
    time_min = min(df_res['time_bin'].min(), divergence_df['hpf'].min())
    time_max = max(df_res['time_bin'].max(), divergence_df['hpf'].max())

    # Colors
    color1 = COLOR_MAP.get(group1_label, '#d62728')
    color2 = COLOR_MAP.get(group2_label, '#1f77b4')

    # =========================================================================
    # Panel 1: AUROC vs Time (colored by significance, with error bars)
    # =========================================================================
    ax1 = axes[0]

    # Significance threshold
    ALPHA = 0.05

    # Split data by significance
    df_sig = df_res[df_res['pval'] < ALPHA]
    df_nonsig = df_res[df_res['pval'] >= ALPHA]

    # Plot connecting line (all points)
    ax1.plot(
        df_res['time_bin'],
        df_res['auroc_observed'],
        linewidth=2,
        color='gray',
        alpha=0.5,
        zorder=1
    )

    # Plot error bars for all points (using null_std as uncertainty)
    if 'auroc_null_std' in df_res.columns:
        ax1.errorbar(
            df_res['time_bin'],
            df_res['auroc_observed'],
            yerr=df_res['auroc_null_std'],
            fmt='none',
            ecolor='gray',
            elinewidth=1.5,
            capsize=3,
            capthick=1.5,
            alpha=0.5,
            zorder=1
        )

    # Plot significant points (filled green circles)
    if len(df_sig) > 0:
        ax1.scatter(
            df_sig['time_bin'],
            df_sig['auroc_observed'],
            s=100,
            c='#2ca02c',  # Green
            marker='o',
            edgecolors='darkgreen',
            linewidths=1.5,
            label=f'Significant (p < {ALPHA})',
            zorder=3
        )

    # Plot non-significant points (open gray circles)
    if len(df_nonsig) > 0:
        ax1.scatter(
            df_nonsig['time_bin'],
            df_nonsig['auroc_observed'],
            s=100,
            c='white',
            marker='o',
            edgecolors='gray',
            linewidths=1.5,
            label=f'Not significant (p ≥ {ALPHA})',
            zorder=2
        )

    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=2, label='Chance (AUROC=0.5)')

    ax1.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('AUROC', fontsize=14, fontweight='bold')
    ax1.set_title(
        f'(A) Phenotype Prediction Performance\n'
        f'When can we predict {group1_label} vs {group2_label} from VAE embeddings?',
        fontsize=14, fontweight='bold', loc='left'
    )
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(time_min, time_max)

    # =========================================================================
    # Panel 2: Morphological Divergence
    # =========================================================================
    ax2 = axes[1]

    # Apply Gaussian smoothing to divergence line
    smooth_sigma = 1.5
    abs_diff_smoothed = gaussian_filter1d(
        divergence_df['abs_difference'].values,
        sigma=smooth_sigma
    )

    ax2.plot(
        divergence_df['hpf'],
        abs_diff_smoothed,
        linewidth=3,
        color='black',
        label='|Mean difference| (smoothed)',
        zorder=100
    )

    # Error bands (using smoothed values as center)
    combined_sem = np.sqrt(divergence_df['group1_sem']**2 + divergence_df['group2_sem']**2)
    ax2.fill_between(
        divergence_df['hpf'],
        abs_diff_smoothed - combined_sem,
        abs_diff_smoothed + combined_sem,
        alpha=0.3,
        color='gray',
        label='± SEM'
    )

    ax2.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax2.set_ylabel(f'Absolute Difference ({metric_label})', fontsize=14, fontweight='bold')
    ax2.set_title(
        f'(B) Phenotypic Divergence Over Time\n'
        f'When do {group1_label} and {group2_label} groups actually diverge?',
        fontsize=14, fontweight='bold', loc='left'
    )
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(time_min, time_max)

    # =========================================================================
    # Panel 3: Individual Trajectories
    # =========================================================================
    ax3 = axes[2]

    # Filter raw data
    df_plot = df_raw[
        df_raw[EMBRYO_ID_COL].isin(group1_ids + group2_ids)
    ].copy()
    df_plot = df_plot.dropna(subset=[TIME_COL, metric_col])

    # Smoothing parameter for Panel C (match Panel B)
    smooth_sigma_panel_c = 1.5

    # Plot individual trajectories with smoothing
    for embryo_id in group1_ids:
        embryo_data = df_plot[df_plot[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)
        if len(embryo_data) > 0:
            # Apply Gaussian smoothing to individual embryo trajectory
            metric_values_smoothed = gaussian_filter1d(
                embryo_data[metric_col].values,
                sigma=smooth_sigma_panel_c
            )
            ax3.plot(
                embryo_data[TIME_COL],
                metric_values_smoothed,
                alpha=0.25,
                linewidth=0.8,
                color=color1
            )

    for embryo_id in group2_ids:
        embryo_data = df_plot[df_plot[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)
        if len(embryo_data) > 0:
            # Apply Gaussian smoothing to individual embryo trajectory
            metric_values_smoothed = gaussian_filter1d(
                embryo_data[metric_col].values,
                sigma=smooth_sigma_panel_c
            )
            ax3.plot(
                embryo_data[TIME_COL],
                metric_values_smoothed,
                alpha=0.25,
                linewidth=0.8,
                color=color2
            )

    # Smooth group means
    group1_mean_smoothed = gaussian_filter1d(
        divergence_df['group1_mean'].values,
        sigma=smooth_sigma_panel_c
    )
    group2_mean_smoothed = gaussian_filter1d(
        divergence_df['group2_mean'].values,
        sigma=smooth_sigma_panel_c
    )

    # Plot group means
    ax3.plot(
        divergence_df['hpf'],
        group1_mean_smoothed,
        linewidth=4,
        color=color1,
        label=f'{group1_label} mean (n={len(group1_ids)})',
        zorder=100
    )
    ax3.plot(
        divergence_df['hpf'],
        group2_mean_smoothed,
        linewidth=4,
        color=color2,
        label=f'{group2_label} mean (n={len(group2_ids)})',
        zorder=100
    )

    ax3.set_xlabel('Developmental Time (hpf)', fontsize=14, fontweight='bold')
    ax3.set_ylabel(metric_label, fontsize=14, fontweight='bold')
    ax3.set_title(
        f'(C) Individual Embryo Trajectories and Group Means\n'
        f'Biological data underlying the prediction and divergence',
        fontsize=14, fontweight='bold', loc='left'
    )
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(time_min, time_max)

    # Set x-axis ticks to every 5 hpf for all panels
    tick_start = np.ceil(time_min / 5) * 5  # Round up to nearest 5
    tick_end = np.floor(time_max / 5) * 5 + 5  # Include endpoint
    tick_positions = np.arange(tick_start, tick_end, 5)
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(tick_positions)

    # Overall title
    fig.suptitle(
        f'B9D2 Phenotype Comparison: {group1_label} vs {group2_label}',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved figure: {save_path}")

    plt.close(fig)


# =============================================================================
# Main Execution
# =============================================================================

def run_single_comparison(
    df_raw: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    group1_label: str,
    group2_label: str,
    comparison_name: str
):
    """
    Run a single comparison between two groups.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Full raw dataframe
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    group1_label : str
        Label for group 1
    group2_label : str
        Label for group 2
    comparison_name : str
        Name for saving files
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: {comparison_name}")
    print(f"{'='*80}")
    print(f"Group 1 ({group1_label}): {len(group1_ids)} embryos")
    print(f"Group 2 ({group2_label}): {len(group2_ids)} embryos")

    # Step 1: Prepare data
    print("\n[1/4] Preparing comparison data...")
    df_binned = prepare_comparison_data(
        df_raw, group1_ids, group2_ids, group1_label, group2_label
    )

    # Step 2: Run difference detection
    print("\n[2/4] Running difference detection...")
    df_results, df_embryo_probs = run_difference_detection(df_binned, comparison_name)

    # Step 3: Compute morphological divergence
    print("\n[3/4] Computing morphological divergence...")

    # Determine which metric to use
    if group1_label in METRICS:
        metric_col = METRICS[group1_label]
    elif group2_label in METRICS:
        metric_col = METRICS[group2_label]
    else:
        metric_col = 'total_length_um'  # Default

    # Check if metric exists
    if metric_col not in df_raw.columns:
        print(f"  Warning: {metric_col} not found, using total_length_um")
        metric_col = 'total_length_um'

    divergence_df = compute_morphological_divergence(
        df_raw, group1_ids, group2_ids, metric_col
    )

    # Metric label for plotting
    metric_labels = {
        'total_length_um': 'Total Length (µm)',
        'baseline_deviation_normalized': 'Baseline Deviation (normalized)',
        'normalized_baseline_deviation': 'Baseline Deviation (normalized)'
    }
    metric_label = metric_labels.get(metric_col, metric_col)

    # Step 4: Create figure
    print("\n[4/4] Creating comprehensive figure...")
    figure_path = FIGURES_DIR / f'{comparison_name}_comprehensive.png'
    create_comprehensive_figure(
        df_results, divergence_df, df_raw,
        group1_ids, group2_ids,
        group1_label, group2_label,
        metric_col, metric_label,
        figure_path
    )

    # Save classification results
    results_path = CLASSIFICATION_DIR / f'{comparison_name}.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_path, index=False)
    print(f"  Saved classification results: {results_path}")

    print(f"\n{comparison_name} complete!")


def main():
    """Main execution function."""
    print("="*80)
    print("B9D2 PHENOTYPE COMPARISON ANALYSIS")
    print("="*80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    # Load phenotypes
    print("\n[Step 1/3] Loading phenotype lists...")
    phenotypes = load_all_phenotypes()

    # Load experiment data
    print("\n[Step 2/3] Loading experiment data...")
    df_raw = load_experiment_data()

    # Extract wildtype
    print("\n[Step 3/3] Extracting wildtype controls...")
    wildtype_ids = extract_wildtype_embryos(df_raw, phenotypes)

    # Define comparisons
    comparisons = [
        {
            'name': 'CE_vs_wildtype',
            'group1_ids': phenotypes['CE'],
            'group2_ids': wildtype_ids,
            'group1_label': 'CE',
            'group2_label': 'wildtype'
        },
        {
            'name': 'HTA_vs_wildtype',
            'group1_ids': phenotypes['HTA'],
            'group2_ids': wildtype_ids,
            'group1_label': 'HTA',
            'group2_label': 'wildtype'
        },
        {
            'name': 'BA_rescue_vs_wildtype',
            'group1_ids': phenotypes['BA_rescue'],
            'group2_ids': wildtype_ids,
            'group1_label': 'BA_rescue',
            'group2_label': 'wildtype'
        },
        {
            'name': 'HTA_vs_BA_rescue',
            'group1_ids': phenotypes['HTA'],
            'group2_ids': phenotypes['BA_rescue'],
            'group1_label': 'HTA',
            'group2_label': 'BA_rescue'
        },
        {
            'name': 'CE_vs_HTA',
            'group1_ids': phenotypes['CE'],
            'group2_ids': phenotypes['HTA'],
            'group1_label': 'CE',
            'group2_label': 'HTA'
        }
    ]

    # Run all comparisons
    for comp in comparisons:
        run_single_comparison(
            df_raw,
            comp['group1_ids'],
            comp['group2_ids'],
            comp['group1_label'],
            comp['group2_label'],
            comp['name']
        )

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Classification: {CLASSIFICATION_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
