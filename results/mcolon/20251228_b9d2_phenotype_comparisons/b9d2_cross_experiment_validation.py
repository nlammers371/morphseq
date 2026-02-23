"""
B9D2 Cross-Experiment Validation Analysis

Tests whether phenotype classifiers trained on one experiment generalize to another.
Uses k-fold CV models trained on experiment A to predict on experiment B (ensemble).

Key concept:
- Train K=5 models on experiment A (k-fold CV)
- Use those SAME K models to predict on experiment B
- Compare within-exp AUROC vs cross-exp AUROC
- Good generalization: similar AUROC; Poor: significant drop

Comparisons:
1. CE vs Wildtype
2. HTA vs Wildtype
3. BA-rescue vs Wildtype
4. HTA vs BA-rescue

Usage:
    python b9d2_cross_experiment_validation.py

Output:
    - cross_experiment_results/*.csv (results for each comparison)
    - figures/*_cross_exp_validation.png (plots)

Author: Generated via Claude Code
Date: 2025-12-29
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

# Import shared functions from b9d2_phenotype_comparison
sys.path.insert(0, str(Path(__file__).parent))
from b9d2_phenotype_comparison import (
    parse_phenotype_file,
    load_all_phenotypes,
    load_experiment_data,
    extract_wildtype_embryos,
)

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
OUTPUT_DIR = Path(__file__).parent / 'cross_exp_output'
CLASSIFICATION_DIR = OUTPUT_DIR / 'classification_results'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Create directories
CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
GENOTYPE_COL = 'genotype'
BIN_WIDTH = 4.0  # hours

# Statistical parameters
N_SPLITS = 5       # k-fold cross-validation folds
N_PERM = 100       # permutation test iterations
RANDOM_STATE = 42
MIN_SAMPLES_PER_GROUP = 5  # minimum per experiment per time bin

# Colors for plotting
COLOR_MAP = {
    'CE': '#d62728',           # Red
    'HTA': '#ff7f0e',          # Orange
    'BA_rescue': '#2ca02c',    # Green
    'wildtype': '#1f77b4',     # Blue
}


# =============================================================================
# Data Preparation for Cross-Experiment Testing
# =============================================================================

def prepare_cross_exp_data(
    df: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    time_bin: int
) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]]:
    """
    Prepare data for cross-experiment validation at a specific time bin.

    Returns data split by experiment, or None if insufficient samples in either experiment.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with all experiments
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    time_bin : int
        Time bin (hpf) to extract

    Returns
    -------
    data_by_exp : Dict[str, Tuple[X, y, sample_counts]] or None
        Keys: '20251121', '20251125'
        Values: (X, y, {'group1': n, 'group2': n})
        Returns None if either experiment has insufficient samples
    """
    # Filter to relevant embryos and time bin
    df_filtered = df[
        (df[EMBRYO_ID_COL].isin(group1_ids + group2_ids)) &
        (df['time_bin'] == time_bin)
    ].copy()

    if len(df_filtered) == 0:
        return None

    # Create labels
    df_filtered['phenotype_label'] = df_filtered[EMBRYO_ID_COL].apply(
        lambda x: 0 if x in group1_ids else 1
    )

    # Get VAE embedding columns (only z_mu_b, not z_mu_n)
    z_cols = [c for c in df_filtered.columns if 'z_mu_b' in c]

    if len(z_cols) == 0:
        return None

    # Prepare data by experiment
    data_by_exp = {}

    for exp_id in EXPERIMENT_IDS:
        df_exp = df_filtered[df_filtered['experiment_id'] == exp_id].copy()

        if len(df_exp) == 0:
            continue

        # Average per embryo per group per experiment
        # (each embryo may have multiple timepoints within this bin)
        df_binned = df_exp.groupby([EMBRYO_ID_COL, 'phenotype_label'], as_index=False)[z_cols].mean()

        # Check minimum samples per group
        counts = df_binned['phenotype_label'].value_counts()

        if len(counts) < 2:  # Both groups must be present
            continue

        if (counts[0] < MIN_SAMPLES_PER_GROUP) or (counts[1] < MIN_SAMPLES_PER_GROUP):
            continue

        # Extract features and labels
        X = df_binned[z_cols].values
        y = df_binned['phenotype_label'].values

        sample_counts = {
            'group1': int(counts.get(0, 0)),
            'group2': int(counts.get(1, 0))
        }

        data_by_exp[exp_id] = (X, y, sample_counts)

    # Return only if both experiments have data
    if len(data_by_exp) == 2:
        return data_by_exp
    else:
        return None


# =============================================================================
# Cross-Experiment Validation
# =============================================================================

def train_cv_models_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[List[LogisticRegression], float, np.ndarray]:
    """
    Train K models using stratified k-fold CV and evaluate on held-out folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Binary labels (n_samples,)
    random_state : int
        Random seed

    Returns
    -------
    trained_models : List[LogisticRegression]
        K trained models
    within_auroc : float
        AUROC computed on aggregated held-out fold predictions
    within_probs : np.ndarray
        Predicted probabilities on all held-out samples (in original order)
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    trained_models = []
    all_probs = np.zeros_like(y, dtype=float)
    fold_indices = np.zeros_like(y, dtype=int)  # Track which fold each sample is in

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        trained_models.append(clf)

        # Predict on held-out fold
        probs = clf.predict_proba(X_test)[:, 1]  # Probability of class 1
        all_probs[test_idx] = probs
        fold_indices[test_idx] = fold_idx

    # Compute AUROC on aggregated predictions
    within_auroc = roc_auc_score(y, all_probs)

    return trained_models, within_auroc, all_probs


def predict_with_ensemble(
    trained_models: List[LogisticRegression],
    X_test: np.ndarray
) -> np.ndarray:
    """
    Get ensemble predictions by averaging across K models.

    Parameters
    ----------
    trained_models : List[LogisticRegression]
        K trained models
    X_test : np.ndarray
        Test feature matrix

    Returns
    -------
    ensemble_probs : np.ndarray
        Averaged predicted probabilities (n_test,)
    """
    all_probs = []

    for model in trained_models:
        probs = model.predict_proba(X_test)[:, 1]
        all_probs.append(probs)

    ensemble_probs = np.mean(all_probs, axis=0)

    return ensemble_probs


def compute_cross_exp_auroc_with_perm(
    trained_models: List[LogisticRegression],
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute AUROC and p-value for cross-experiment predictions.

    Uses trained models to predict on test data, then runs permutation test
    on test labels to get p-value.

    Parameters
    ----------
    trained_models : List[LogisticRegression]
        K trained models (never trained on X_test)
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        Test labels
    random_state : int
        Random seed for permutation test

    Returns
    -------
    cross_auroc : float
        AUROC on test set
    pval : float
        Permutation test p-value
    """
    # Get ensemble predictions
    test_probs = predict_with_ensemble(trained_models, X_test)

    # Compute observed AUROC
    cross_auroc = roc_auc_score(y_test, test_probs)

    # Permutation test: shuffle labels
    rng = np.random.RandomState(random_state)
    null_aurocs = []

    for _ in range(N_PERM):
        y_perm = rng.permutation(y_test)
        null_auroc = roc_auc_score(y_perm, test_probs)
        null_aurocs.append(null_auroc)

    null_aurocs = np.array(null_aurocs)

    # Compute p-value
    k = np.sum(null_aurocs >= cross_auroc)
    pval = (k + 1) / (N_PERM + 1)

    return cross_auroc, pval


def run_cross_experiment_comparison(
    df: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    group1_name: str,
    group2_name: str
) -> pd.DataFrame:
    """
    Run cross-experiment validation for a phenotype comparison.

    For each time bin:
    1. Train K models on exp A, evaluate within-exp, use to predict on exp B
    2. Train K models on exp B, evaluate within-exp, use to predict on exp A

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with all experiments
    group1_ids : List[str]
        Embryo IDs for group 1
    group2_ids : List[str]
        Embryo IDs for group 2
    group1_name : str
        Label for group 1
    group2_name : str
        Label for group 2

    Returns
    -------
    results_df : pd.DataFrame
        Results with columns: time_bin, validation_type, train_exp, test_exp,
        auroc, pval, n_train_g1, n_train_g2, n_test_g1, n_test_g2
    """
    # Prepare data: create time bins for all embryos
    df_working = df[
        df[EMBRYO_ID_COL].isin(group1_ids + group2_ids)
    ].copy()

    df_working['time_bin'] = (
        np.floor(df_working[TIME_COL] / BIN_WIDTH) * BIN_WIDTH
    ).astype(int)

    time_bins = sorted(df_working['time_bin'].unique())

    results = []

    print(f"\n{'='*70}")
    print(f"Running: {group1_name} vs {group2_name}")
    print(f"  Total embryos: {len(group1_ids)} vs {len(group2_ids)}")
    print(f"  Time bins: {len(time_bins)}")
    print(f"{'='*70}")

    for time_bin in time_bins:
        # Get data for this time bin, split by experiment
        data_by_exp = prepare_cross_exp_data(df_working, group1_ids, group2_ids, time_bin)

        if data_by_exp is None:
            continue  # Skip if insufficient samples

        X_exp1, y_exp1, counts_exp1 = data_by_exp['20251121']
        X_exp2, y_exp2, counts_exp2 = data_by_exp['20251125']

        # ===== Train on exp1, test on exp1 and exp2 =====
        models_exp1, within_auroc_exp1, _ = train_cv_models_and_evaluate(X_exp1, y_exp1)

        results.append({
            'time_bin': time_bin,
            'validation_type': 'within',
            'train_exp': '20251121',
            'test_exp': '20251121',
            'auroc': within_auroc_exp1,
            'pval': np.nan,  # No permutation test for within-exp
            'n_train_g1': counts_exp1['group1'],
            'n_train_g2': counts_exp1['group2'],
            'n_test_g1': counts_exp1['group1'],
            'n_test_g2': counts_exp1['group2'],
        })

        # Use exp1 models to predict on exp2
        cross_auroc_1to2, pval_1to2 = compute_cross_exp_auroc_with_perm(
            models_exp1, X_exp2, y_exp2
        )

        results.append({
            'time_bin': time_bin,
            'validation_type': 'cross',
            'train_exp': '20251121',
            'test_exp': '20251125',
            'auroc': cross_auroc_1to2,
            'pval': pval_1to2,
            'n_train_g1': counts_exp1['group1'],
            'n_train_g2': counts_exp1['group2'],
            'n_test_g1': counts_exp2['group1'],
            'n_test_g2': counts_exp2['group2'],
        })

        # ===== Train on exp2, test on exp2 and exp1 =====
        models_exp2, within_auroc_exp2, _ = train_cv_models_and_evaluate(X_exp2, y_exp2)

        results.append({
            'time_bin': time_bin,
            'validation_type': 'within',
            'train_exp': '20251125',
            'test_exp': '20251125',
            'auroc': within_auroc_exp2,
            'pval': np.nan,  # No permutation test for within-exp
            'n_train_g1': counts_exp2['group1'],
            'n_train_g2': counts_exp2['group2'],
            'n_test_g1': counts_exp2['group1'],
            'n_test_g2': counts_exp2['group2'],
        })

        # Use exp2 models to predict on exp1
        cross_auroc_2to1, pval_2to1 = compute_cross_exp_auroc_with_perm(
            models_exp2, X_exp1, y_exp1
        )

        results.append({
            'time_bin': time_bin,
            'validation_type': 'cross',
            'train_exp': '20251125',
            'test_exp': '20251121',
            'auroc': cross_auroc_2to1,
            'pval': pval_2to1,
            'n_train_g1': counts_exp2['group1'],
            'n_train_g2': counts_exp2['group2'],
            'n_test_g1': counts_exp1['group1'],
            'n_test_g2': counts_exp1['group2'],
        })

        print(f"  Time bin {time_bin:3d} hpf: "
              f"within_exp1={within_auroc_exp1:.3f}, "
              f"cross_1→2={cross_auroc_1to2:.3f} (p={pval_1to2:.3f}), "
              f"within_exp2={within_auroc_exp2:.3f}, "
              f"cross_2→1={cross_auroc_2to1:.3f} (p={pval_2to1:.3f})")

    results_df = pd.DataFrame(results)
    print(f"\nCompleted: {len(results_df)} results rows")

    return results_df


# =============================================================================
# Plotting
# =============================================================================

def plot_cross_experiment_results(
    results_df: pd.DataFrame,
    group1_name: str,
    group2_name: str,
    output_path: Path
) -> None:
    """
    Create a 2-panel figure showing within-exp vs cross-exp AUROC.

    Panel A: AUROC over time
    Panel B: Generalization gap (within - cross)

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from run_cross_experiment_comparison
    group1_name : str
        Name of group 1 (for title)
    group2_name : str
        Name of group 2 (for title)
    output_path : Path
        Path to save figure
    """
    # Check if dataframe is empty
    if len(results_df) == 0:
        print(f"  Skipping plot for {group1_name} vs {group2_name}: No data (insufficient samples)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: AUROC over time
    ax = axes[0]

    # Get data for each line
    within_exp1 = results_df[
        (results_df['validation_type'] == 'within') &
        (results_df['train_exp'] == '20251121')
    ].sort_values('time_bin')

    within_exp2 = results_df[
        (results_df['validation_type'] == 'within') &
        (results_df['train_exp'] == '20251125')
    ].sort_values('time_bin')

    cross_1to2 = results_df[
        (results_df['validation_type'] == 'cross') &
        (results_df['train_exp'] == '20251121')
    ].sort_values('time_bin')

    cross_2to1 = results_df[
        (results_df['validation_type'] == 'cross') &
        (results_df['train_exp'] == '20251125')
    ].sort_values('time_bin')

    # Plot lines
    ax.plot(within_exp1['time_bin'], within_exp1['auroc'],
            'o-', color='#1f77b4', label='Within-exp (20251121)', linewidth=2)
    ax.plot(within_exp2['time_bin'], within_exp2['auroc'],
            'o-', color='#ff7f0e', label='Within-exp (20251125)', linewidth=2)
    ax.plot(cross_1to2['time_bin'], cross_1to2['auroc'],
            's--', color='#1f77b4', label='Cross (20251121→20251125)', linewidth=2)
    ax.plot(cross_2to1['time_bin'], cross_2to1['auroc'],
            's--', color='#ff7f0e', label='Cross (20251125→20251121)', linewidth=2)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'{group1_name} vs {group2_name}: Within vs Cross-Exp AUROC', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.45, 1.05])

    # Panel B: Generalization gap
    ax = axes[1]

    # Compute gap for each direction
    gap_1to2 = within_exp1.set_index('time_bin')['auroc'] - cross_1to2.set_index('time_bin')['auroc']
    gap_2to1 = within_exp2.set_index('time_bin')['auroc'] - cross_2to1.set_index('time_bin')['auroc']

    ax.plot(gap_1to2.index, gap_1to2.values, 'o-', color='#1f77b4',
            label='Gap (20251121→20251125)', linewidth=2)
    ax.plot(gap_2to1.index, gap_2to1.values, 's-', color='#ff7f0e',
            label='Gap (20251125→20251121)', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('Generalization Gap\n(Within - Cross AUROC)', fontsize=12)
    ax.set_title('Generalization Gap: Performance Drop Across Experiments', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run cross-experiment validation for all phenotype comparisons."""

    # Load data
    print("\nLoading phenotypes and experiment data...")
    phenotypes = load_all_phenotypes()
    df = load_experiment_data()
    wildtype_ids = extract_wildtype_embryos(df, phenotypes)

    # Add time bins to main dataframe
    df['time_bin'] = (
        np.floor(df[TIME_COL] / BIN_WIDTH) * BIN_WIDTH
    ).astype(int)

    # Define comparisons
    comparisons = [
        ('CE', phenotypes['CE'], 'wildtype', wildtype_ids),
        ('HTA', phenotypes['HTA'], 'wildtype', wildtype_ids),
        ('BA_rescue', phenotypes['BA_rescue'], 'wildtype', wildtype_ids),
        ('HTA', phenotypes['HTA'], 'BA_rescue', phenotypes['BA_rescue']),
    ]

    # Run each comparison
    for group1_name, group1_ids, group2_name, group2_ids in comparisons:
        # Create comparison label for output
        comp_label = f"{group1_name}_vs_{group2_name}"

        # Run analysis
        results_df = run_cross_experiment_comparison(
            df, group1_ids, group2_ids, group1_name, group2_name
        )

        # Save results
        output_csv = CLASSIFICATION_DIR / f"{comp_label}_cross_exp.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"  Saved CSV: {output_csv}")

        # Create plot (only if we have data)
        if len(results_df) > 0:
            output_fig = FIGURES_DIR / f"{comp_label}_cross_exp_validation.png"
            plot_cross_experiment_results(results_df, group1_name, group2_name, output_fig)
        else:
            print(f"  Skipping plot: No results for {comp_label} (insufficient samples across experiments)")

    print(f"\n{'='*70}")
    print("All comparisons complete!")
    print(f"Results: {CLASSIFICATION_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
