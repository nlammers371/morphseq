"""
B9D2 Trajectory Classification - Phase 3

Per-time-bin classification to predict penetrant vs non-penetrant phenotype
from binned embeddings using the cluster assignments from Phase 1.

Configuration:
    SELECTED_K = 3
    PENETRANT_CLUSTERS = [0]         # Lower length, more penetrant
    NON_PENETRANT_CLUSTERS = [2]     # Higher length, rescued

Usage:
    python b9d2_trajectory_classifier.py

Output:
    - classification_results.csv       # F1-score per time bin
    - figures/f1_vs_time_bin.png      # Main result plot

Author: Generated via Claude Code
Date: 2025-12-10
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
PENETRANT_CLUSTERS = [0]         # Lower length, more penetrant
NON_PENETRANT_CLUSTERS = [2]     # Higher length, non-penetrant
# Cluster 1 (outliers) is excluded

OUTPUT_DIR = Path(__file__).parent / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Trajectory parameters
TIME_COL = 'predicted_stage_hpf'
METRIC_COL = 'total_length_um'
EMBRYO_ID_COL = 'embryo_id'
GENOTYPE_COL = 'genotype'
PAIR_COL = 'pair'

# Binning parameters
BIN_WIDTH = 2.0  # Hours

# Classification parameters
N_FOLDS = 5
MIN_SAMPLES_PER_BIN = 10


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_and_prepare_data():
    """
    Load data from multiple experiments and filter for target pairs.

    Returns
    -------
    df : pd.DataFrame
        Combined dataframe with all experiments and pairs
    """
    print("Loading data from experiments...")

    dfs = []
    for exp_id in EXPERIMENT_IDS:
        print(f"  Loading {exp_id}...")
        df = load_experiment_dataframe(exp_id, format_version='df03')
        df['experiment_id'] = exp_id
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Handle merge suffixes (total_length_um_x and total_length_um_y)
    # Use the first available version
    if 'total_length_um_x' in df.columns and 'total_length_um' not in df.columns:
        df['total_length_um'] = df['total_length_um_x']
    elif 'total_length_um_y' in df.columns and 'total_length_um' not in df.columns:
        df['total_length_um'] = df['total_length_um_y']

    # Filter for valid embryos
    if 'use_embryo_flag' in df.columns:
        df = df[df['use_embryo_flag'] == 1].copy()

    # Filter for target pairs
    df = df[df[PAIR_COL].isin(PAIRS)].copy()

    # Drop rows with missing values
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, METRIC_COL, GENOTYPE_COL])

    print(f"  Total samples: {len(df)}")
    print(f"  Unique embryos: {df[EMBRYO_ID_COL].nunique()}")

    return df


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

    print(f"Loaded {len(cluster_assignments)} cluster assignments from k={SELECTED_K}")

    return cluster_assignments


def bin_embeddings_by_time(df, bin_width=2.0):
    """
    Bin embeddings by time for each embryo.

    Returns
    -------
    df_binned : pd.DataFrame
        One row per (embryo_id, time_bin) with averaged embeddings
    """
    print(f"Binning embeddings (bin_width={bin_width}h)...")

    # Detect latent columns
    z_cols = [c for c in df.columns if 'z_mu_n_' in c or 'z_mu_b' in c]
    if not z_cols:
        raise ValueError("No latent columns found!")

    print(f"  Found {len(z_cols)} latent columns")

    # Create time bins
    df = df.copy()
    df['time_bin'] = (np.floor(df[TIME_COL] / bin_width) * bin_width).astype(int)

    # Average latents per (embryo, time_bin)
    agg = df.groupby([EMBRYO_ID_COL, 'time_bin'], as_index=False)[z_cols].mean()

    # Add suffix
    agg.rename(columns={c: f'{c}_binned' for c in z_cols}, inplace=True)

    # Merge metadata
    meta_cols = [EMBRYO_ID_COL, GENOTYPE_COL, PAIR_COL]
    meta_df = df[meta_cols].drop_duplicates(subset=[EMBRYO_ID_COL])
    agg = agg.merge(meta_df, on=EMBRYO_ID_COL, how='left')

    print(f"  Binned to {len(agg)} (embryo, time_bin) samples")
    print(f"  Time bins: {sorted(agg['time_bin'].unique())}")

    return agg


def merge_cluster_labels(df_binned, cluster_assignments):
    """
    Merge cluster labels with binned embeddings.

    Returns
    -------
    df_labeled : pd.DataFrame
        Binned data with cluster labels and binary phenotype labels
    """
    print("Merging cluster labels...")

    # Add cluster assignments
    cluster_df = pd.DataFrame([
        {EMBRYO_ID_COL: eid, 'cluster_id': cid}
        for eid, cid in cluster_assignments.items()
    ])

    df_labeled = df_binned.merge(cluster_df, on=EMBRYO_ID_COL, how='left')

    # Create binary phenotype label
    # 1 = penetrant (cluster 0), 0 = non-penetrant (cluster 2)
    df_labeled['phenotype_label'] = 0  # Default to non-penetrant
    df_labeled.loc[df_labeled['cluster_id'].isin(PENETRANT_CLUSTERS), 'phenotype_label'] = 1

    # Filter out outlier clusters
    valid_clusters = PENETRANT_CLUSTERS + NON_PENETRANT_CLUSTERS
    df_labeled = df_labeled[df_labeled['cluster_id'].isin(valid_clusters)].copy()

    print(f"  Penetrant (label=1): {(df_labeled['phenotype_label'] == 1).sum()} samples")
    print(f"  Non-penetrant (label=0): {(df_labeled['phenotype_label'] == 0).sum()} samples")

    return df_labeled


# =============================================================================
# Classification
# =============================================================================

def classify_at_time_bin(df_labeled, time_bin, embedding_cols, n_folds=5):
    """
    Run classification at a single time bin.

    Returns
    -------
    dict with f1_mean, f1_std, precision, recall, auc, n_samples
    """
    # Filter to this time bin
    df_bin = df_labeled[df_labeled['time_bin'] == time_bin].copy()

    if len(df_bin) < MIN_SAMPLES_PER_BIN:
        return {
            'time_bin': time_bin,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'precision_mean': np.nan,
            'recall_mean': np.nan,
            'auc_mean': np.nan,
            'n_samples': len(df_bin),
            'n_penetrant': 0,
            'n_non_penetrant': 0,
        }

    # Prepare features and labels
    X = df_bin[embedding_cols].values
    y = df_bin['phenotype_label'].values
    groups = df_bin[EMBRYO_ID_COL].values

    # Check class balance
    n_penetrant = (y == 1).sum()
    n_non_penetrant = (y == 0).sum()

    if n_penetrant < 2 or n_non_penetrant < 2:
        return {
            'time_bin': time_bin,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'precision_mean': np.nan,
            'recall_mean': np.nan,
            'auc_mean': np.nan,
            'n_samples': len(df_bin),
            'n_penetrant': n_penetrant,
            'n_non_penetrant': n_non_penetrant,
        }

    # Cross-validation
    n_unique = len(np.unique(groups))
    n_folds_actual = min(n_folds, n_unique)

    if n_folds_actual < 2:
        return {
            'time_bin': time_bin,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'precision_mean': np.nan,
            'recall_mean': np.nan,
            'auc_mean': np.nan,
            'n_samples': len(df_bin),
            'n_penetrant': n_penetrant,
            'n_non_penetrant': n_non_penetrant,
        }

    gkf = GroupKFold(n_splits=n_folds_actual)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    auc_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip if test set has only one class
        if len(np.unique(y_test)) < 2:
            continue

        # Build pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))

        try:
            auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        except:
            pass

    if not f1_scores:
        return {
            'time_bin': time_bin,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'precision_mean': np.nan,
            'recall_mean': np.nan,
            'auc_mean': np.nan,
            'n_samples': len(df_bin),
            'n_penetrant': n_penetrant,
            'n_non_penetrant': n_non_penetrant,
        }

    return {
        'time_bin': time_bin,
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0,
        'precision_mean': np.mean(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'auc_mean': np.mean(auc_scores) if auc_scores else np.nan,
        'n_samples': len(df_bin),
        'n_penetrant': n_penetrant,
        'n_non_penetrant': n_non_penetrant,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_f1_vs_time(results_df, save_path):
    """
    Plot F1-score vs time bin.
    """
    print("Generating F1-score vs time bin plot...")

    df_valid = results_df[results_df['f1_mean'].notna()].copy()

    if len(df_valid) == 0:
        print("  No valid results to plot!")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot F1-score
    ax1 = axes[0]
    ax1.errorbar(df_valid['time_bin'], df_valid['f1_mean'],
                yerr=df_valid['f1_std'], marker='o', linewidth=2.5,
                markersize=8, capsize=4, color='#1f77b4', label='F1-score')

    # Add precision and recall
    ax1.plot(df_valid['time_bin'], df_valid['precision_mean'], '--',
            linewidth=2, markersize=6, color='#2ca02c', label='Precision', marker='s')
    ax1.plot(df_valid['time_bin'], df_valid['recall_mean'], '--',
            linewidth=2, markersize=6, color='#d62728', label='Recall', marker='^')

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Chance level')
    ax1.axhline(y=0.7, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='Target threshold (0.7)')

    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title(f'Phenotype Prediction Metrics vs Time Bin - B9D2 (Penetrant vs Non-Penetrant)\n'
                 f'(Penetrant=Cluster 0, Non-Penetrant=Cluster 2)',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Find earliest time with F1 > 0.7
    above_threshold = df_valid[df_valid['f1_mean'] >= 0.7]
    if not above_threshold.empty:
        earliest = above_threshold['time_bin'].min()
        ax1.axvline(x=earliest, color='green', linestyle='-', alpha=0.4, linewidth=2)
        ax1.text(earliest, 0.95, f'Earliest predictive:\n{earliest:.0f}h',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Plot sample sizes
    ax2 = axes[1]
    width = 1.5
    ax2.bar(df_valid['time_bin'] - width/2, df_valid['n_penetrant'],
           width=width, label='Penetrant (Cluster 0)', color='#d62728', alpha=0.7)
    ax2.bar(df_valid['time_bin'] + width/2, df_valid['n_non_penetrant'],
           width=width, label='Non-Penetrant (Cluster 2)', color='#1f77b4', alpha=0.7)

    ax2.set_xlabel('Time Bin (hpf)', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Size per Time Bin', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("B9D2 TRAJECTORY CLASSIFICATION - PHASE 3")
    print("=" * 80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Pairs: {PAIRS}")
    print(f"Selected K: {SELECTED_K}")
    print(f"Penetrant clusters: {PENETRANT_CLUSTERS}")
    print(f"Non-penetrant clusters: {NON_PENETRANT_CLUSTERS}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_and_prepare_data()

    # Load cluster assignments
    print("\n[2/4] Loading cluster assignments...")
    cluster_assignments = load_cluster_assignments()

    # Bin embeddings
    print("\n[3/4] Binning embeddings...")
    df_binned = bin_embeddings_by_time(df, bin_width=BIN_WIDTH)

    # Merge cluster labels
    print("\n[4/4] Merging cluster labels...")
    df_labeled = merge_cluster_labels(df_binned, cluster_assignments)

    # Get embedding columns
    embedding_cols = [c for c in df_labeled.columns if c.endswith('_binned') and 'z_mu' in c]
    print(f"  Using {len(embedding_cols)} embedding features")

    # Get unique time bins
    time_bins = sorted(df_labeled['time_bin'].unique())
    print(f"  Time bins to classify: {time_bins}")

    # Run classification at each time bin
    print("\nRunning classification at each time bin...")
    results = []

    for time_bin in time_bins:
        result = classify_at_time_bin(df_labeled, time_bin, embedding_cols, n_folds=N_FOLDS)
        results.append(result)

        if not np.isnan(result['f1_mean']):
            print(f"  Time {time_bin:3.0f}h: F1={result['f1_mean']:.3f} ± {result['f1_std']:.3f} "
                  f"(n={result['n_samples']}, pen={result['n_penetrant']}, non_pen={result['n_non_penetrant']})")
        else:
            print(f"  Time {time_bin:3.0f}h: insufficient samples (n={result['n_samples']})")

    results_df = pd.DataFrame(results)

    # Save results
    print("\nSaving results...")
    results_df.to_csv(OUTPUT_DIR / 'classification_results.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'classification_results.csv'}")

    # Generate plot
    plot_f1_vs_time(results_df, save_path=FIGURES_DIR / 'f1_vs_time_bin.png')

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)

    df_valid = results_df[results_df['f1_mean'].notna()]
    if len(df_valid) > 0:
        print(f"\nF1-Score Summary:")
        print(f"  Mean F1: {df_valid['f1_mean'].mean():.3f}")
        print(f"  Max F1:  {df_valid['f1_mean'].max():.3f} at time {df_valid.loc[df_valid['f1_mean'].idxmax(), 'time_bin']:.0f}h")
        print(f"  Min F1:  {df_valid['f1_mean'].min():.3f} at time {df_valid.loc[df_valid['f1_mean'].idxmin(), 'time_bin']:.0f}h")

        above_threshold = df_valid[df_valid['f1_mean'] >= 0.7]
        if not above_threshold.empty:
            earliest = above_threshold['time_bin'].min()
            print(f"\n  ✅ Earliest predictive timepoint (F1 >= 0.7): {earliest:.0f} hpf")
        else:
            print(f"\n  ⚠️  No timepoint achieved F1 >= 0.7")
            best_f1 = df_valid['f1_mean'].max()
            best_time = df_valid.loc[df_valid['f1_mean'].idxmax(), 'time_bin']
            print(f"     Best prediction: {best_f1:.3f} F1 at {best_time:.0f} hpf")

    print("\n" + "=" * 80)
    print("Results saved to: output/")
    print("=" * 80)


if __name__ == '__main__':
    main()
