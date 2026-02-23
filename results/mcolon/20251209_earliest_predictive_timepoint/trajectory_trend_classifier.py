"""
Trajectory Trend Classification Baseline

Test if binned embeddings can predict trajectory fate (increasing vs decreasing
curvature) using binary classification instead of continuous regression.

Two-phase workflow:
  Phase A: DTW clustering + visual inspection checkpoint
  Phase B: Per-time-bin classification with F1-score

Usage:
  python trajectory_trend_classifier.py --phase clustering   # Phase A only
  python trajectory_trend_classifier.py --phase classification  # Phase B (requires Phase A)
  python trajectory_trend_classifier.py  # Run both phases

Author: Generated via Claude Code
Date: 2025-12-09
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Add src to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

from analyze.trajectory_analysis.data_loading import load_experiment_dataframe
from analyze.trajectory_analysis import (
    compute_dtw_distance_matrix,
    run_bootstrap_hierarchical,
    analyze_bootstrap_results,
)

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_ID = '20250512'
OUTPUT_DIR = Path(__file__).parent / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Clustering parameters
K_CLUSTERS = 3  # Start with 3 clusters (WT-like, increasing, decreasing)
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42
DTW_WINDOW = 3

# Trajectory parameters
TIME_COL = 'predicted_stage_hpf'
METRIC_COL = 'baseline_deviation_normalized'
EMBRYO_ID_COL = 'embryo_id'
MIN_TIMEPOINTS = 5
GRID_STEP = 0.5  # HPF for interpolation

# Classification parameters
BIN_WIDTH = 2.0  # Hours
N_FOLDS = 5
WILDTYPE_THRESHOLD = 0.05  # Mean curvature below this = WT-like


# =============================================================================
# Phase A: DTW Clustering
# =============================================================================

def load_and_prepare_data():
    """
    Load experiment data with embeddings and curvature.

    Returns
    -------
    df : pd.DataFrame
        Full dataframe with all columns
    """
    print("Loading experiment data...")
    df = load_experiment_dataframe(EXPERIMENT_ID, format_version='df03')
    print(f"  Loaded {len(df)} rows")
    print(f"  Unique embryos: {df[EMBRYO_ID_COL].nunique()}")

    # Handle column naming
    if METRIC_COL not in df.columns and 'normalized_baseline_deviation' in df.columns:
        df[METRIC_COL] = df['normalized_baseline_deviation']

    # Filter out dead/invalid embryos if flag exists
    if 'dead_flag' in df.columns:
        n_before = len(df)
        df = df[df['dead_flag'] == 0].copy()
        print(f"  Filtered dead embryos: {len(df)}/{n_before}")

    return df


def extract_trajectories_for_clustering(df):
    """
    Extract curvature trajectories and interpolate to common grid.

    Returns
    -------
    trajectories : list of ndarray
        List of interpolated trajectory arrays
    embryo_ids : list
        Corresponding embryo IDs
    common_grid : ndarray
        Time grid (hpf)
    df_interpolated : pd.DataFrame
        Interpolated data in long format
    """
    print("Extracting trajectories for clustering...")

    # Filter to required columns
    df_traj = df[[EMBRYO_ID_COL, TIME_COL, METRIC_COL]].dropna().copy()

    # Filter embryos with enough timepoints
    embryo_counts = df_traj.groupby(EMBRYO_ID_COL).size()
    valid_embryos = embryo_counts[embryo_counts >= MIN_TIMEPOINTS].index.tolist()
    df_traj = df_traj[df_traj[EMBRYO_ID_COL].isin(valid_embryos)]

    print(f"  {len(valid_embryos)} embryos with >= {MIN_TIMEPOINTS} timepoints")

    # Create common time grid
    time_min = np.floor(df_traj[TIME_COL].min() / GRID_STEP) * GRID_STEP
    time_max = np.ceil(df_traj[TIME_COL].max() / GRID_STEP) * GRID_STEP
    common_grid = np.arange(time_min, time_max + GRID_STEP, GRID_STEP)

    print(f"  Time grid: {time_min:.1f} to {time_max:.1f} hpf ({len(common_grid)} points)")

    # Interpolate each embryo
    trajectories = []
    embryo_ids = []
    interpolated_records = []

    for embryo_id in valid_embryos:
        embryo_data = df_traj[df_traj[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)

        if len(embryo_data) < 2:
            continue

        # Interpolate
        interp_values = np.interp(
            common_grid,
            embryo_data[TIME_COL].values,
            embryo_data[METRIC_COL].values
        )

        trajectories.append(interp_values)
        embryo_ids.append(embryo_id)

        for t, v in zip(common_grid, interp_values):
            interpolated_records.append({
                'embryo_id': embryo_id,
                'hpf': t,
                'metric_value': v
            })

    df_interpolated = pd.DataFrame(interpolated_records)

    print(f"  Interpolated {len(trajectories)} trajectories")

    return trajectories, embryo_ids, common_grid, df_interpolated


def run_dtw_clustering(trajectories, embryo_ids):
    """
    Run DTW-based hierarchical clustering.

    Returns
    -------
    cluster_assignments : dict
        embryo_id -> cluster_id
    posteriors : dict
        Posterior probabilities from bootstrap
    """
    print(f"Computing DTW distance matrix (window={DTW_WINDOW})...")
    D = compute_dtw_distance_matrix(trajectories, window=DTW_WINDOW, verbose=False)
    print(f"  Distance matrix shape: {D.shape}")

    print(f"Running bootstrap clustering (k={K_CLUSTERS}, n={N_BOOTSTRAP})...")
    bootstrap_results = run_bootstrap_hierarchical(
        D, K_CLUSTERS, embryo_ids,
        n_bootstrap=N_BOOTSTRAP,
        frac=BOOTSTRAP_FRAC,
        random_state=RANDOM_SEED,
        verbose=False
    )

    posteriors = analyze_bootstrap_results(bootstrap_results)

    # Create assignment dict
    cluster_assignments = {
        eid: cluster
        for eid, cluster in zip(posteriors['embryo_ids'], posteriors['modal_cluster'])
    }

    return cluster_assignments, posteriors, D


def label_clusters_by_trend(df_interpolated, cluster_assignments, common_grid):
    """
    Label each cluster as 'increasing', 'decreasing', or 'wildtype' based on trajectory shape.

    Returns
    -------
    cluster_labels : dict
        cluster_id -> {'label': str, 'slope': float, 'mean': float}
    embryo_labels : pd.DataFrame
        embryo_id, cluster_id, trend_label
    """
    print("Labeling clusters by trajectory trend...")

    cluster_ids = sorted(set(cluster_assignments.values()))
    cluster_labels = {}

    for cluster_id in cluster_ids:
        # Get embryos in this cluster
        cluster_embryos = [eid for eid, cid in cluster_assignments.items() if cid == cluster_id]

        # Get trajectory data for this cluster
        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryos)]

        # Compute mean trajectory
        mean_traj = df_cluster.groupby('hpf')['metric_value'].mean()

        # Compute overall mean
        cluster_mean = mean_traj.mean()

        # Fit linear slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            mean_traj.index.values, mean_traj.values
        )

        # Classify
        if cluster_mean < WILDTYPE_THRESHOLD:
            label = 'wildtype'
        elif slope > 0:
            label = 'increasing'
        else:
            label = 'decreasing'

        cluster_labels[cluster_id] = {
            'label': label,
            'slope': slope,
            'mean': cluster_mean,
            'n_embryos': len(cluster_embryos),
            'r_squared': r_value ** 2
        }

        print(f"  Cluster {cluster_id}: {label} (n={len(cluster_embryos)}, "
              f"mean={cluster_mean:.4f}, slope={slope:.6f})")

    # Create embryo labels dataframe
    rows = []
    for embryo_id, cluster_id in cluster_assignments.items():
        rows.append({
            'embryo_id': embryo_id,
            'cluster_id': cluster_id,
            'trend_label': cluster_labels[cluster_id]['label']
        })

    embryo_labels = pd.DataFrame(rows)

    return cluster_labels, embryo_labels


def plot_cluster_inspection(df_interpolated, cluster_assignments, cluster_labels, save_path):
    """
    Generate cluster inspection plot for visual verification.
    """
    print(f"Generating cluster inspection plot...")

    n_clusters = len(cluster_labels)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), sharey=True)

    if n_clusters == 1:
        axes = [axes]

    # Color map for trend labels
    colors = {
        'wildtype': '#2ca02c',      # Green
        'increasing': '#d62728',    # Red
        'decreasing': '#1f77b4',    # Blue
    }

    for idx, (cluster_id, info) in enumerate(sorted(cluster_labels.items())):
        ax = axes[idx]

        # Get embryos in this cluster
        cluster_embryos = [eid for eid, cid in cluster_assignments.items() if cid == cluster_id]
        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryos)]

        # Plot individual trajectories (faded)
        for embryo_id in cluster_embryos:
            embryo_data = df_cluster[df_cluster['embryo_id'] == embryo_id]
            ax.plot(embryo_data['hpf'], embryo_data['metric_value'],
                   alpha=0.2, linewidth=0.5, color=colors[info['label']])

        # Plot mean trajectory (bold)
        mean_traj = df_cluster.groupby('hpf')['metric_value'].mean()
        ax.plot(mean_traj.index, mean_traj.values,
               linewidth=3, color=colors[info['label']], label='Mean')

        # Add trend line
        x = mean_traj.index.values
        y_trend = info['slope'] * x + (info['mean'] - info['slope'] * x.mean())
        ax.plot(x, y_trend, '--', linewidth=2, color='black', alpha=0.5, label='Trend')

        # Labels
        ax.set_title(f"Cluster {cluster_id}: {info['label'].upper()}\n"
                    f"n={info['n_embryos']}, slope={info['slope']:.4f}, mean={info['mean']:.3f}",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (hpf)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Curvature (normalized)', fontsize=10)
        ax.axhline(y=WILDTYPE_THRESHOLD, color='gray', linestyle=':', alpha=0.5, label=f'WT threshold ({WILDTYPE_THRESHOLD})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Cluster Inspection - Experiment {EXPERIMENT_ID}\n'
                 f'(k={n_clusters}, {len(cluster_assignments)} embryos)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {save_path}")

    plt.close(fig)


def generate_cluster_genotype_summary(cluster_assignments, df_with_genotype):
    """
    Create cross-tabulation of cluster_id × genotype.

    Parameters
    ----------
    cluster_assignments : dict
        embryo_id -> cluster_id
    df_with_genotype : pd.DataFrame
        DataFrame with embryo_id and genotype columns

    Returns
    -------
    pd.DataFrame
        Long-format with columns: cluster_id, genotype, count
    """
    # Create DataFrame from cluster assignments
    cluster_df = pd.DataFrame([
        {'embryo_id': eid, 'cluster_id': cid}
        for eid, cid in cluster_assignments.items()
    ])

    # Merge with genotype information
    genotype_map = df_with_genotype[[EMBRYO_ID_COL, 'genotype']].drop_duplicates()
    merged = cluster_df.merge(genotype_map, left_on='embryo_id', right_on=EMBRYO_ID_COL, how='left')

    # Count by cluster and genotype
    summary = merged.groupby(['cluster_id', 'genotype'], dropna=False).size().reset_index(name='count')

    return summary


def plot_cluster_inspection_by_genotype(df_interpolated, cluster_assignments,
                                         cluster_labels, df_with_genotype, save_path):
    """
    Generate cluster inspection plot with trajectories colored by genotype.

    Parameters
    ----------
    df_interpolated : pd.DataFrame
        Trajectory data (embryo_id, hpf, metric_value)
    cluster_assignments : dict
        embryo_id -> cluster_id
    cluster_labels : dict
        cluster_id -> info dict
    df_with_genotype : pd.DataFrame
        Original dataframe with genotype column
    save_path : Path
        Where to save figure
    """
    print(f"Generating cluster inspection plot (colored by genotype)...")

    # Merge genotype into trajectory data
    genotype_map = df_with_genotype[[EMBRYO_ID_COL, 'genotype']].drop_duplicates()
    df_interpolated = df_interpolated.merge(genotype_map, on='embryo_id', how='left')

    n_clusters = len(cluster_labels)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), sharey=True)

    if n_clusters == 1:
        axes = [axes]

    # Color map for genotypes
    genotype_colors = {
        'cep290_wildtype': '#2ca02c',       # Green
        'cep290_heterozygous': '#1f77b4',   # Blue
        'cep290_homozygous': '#d62728',     # Red
    }

    for idx, (cluster_id, info) in enumerate(sorted(cluster_labels.items())):
        ax = axes[idx]

        # Get embryos in this cluster
        cluster_embryos = [eid for eid, cid in cluster_assignments.items() if cid == cluster_id]
        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryos)]

        # Plot individual trajectories colored by genotype
        for embryo_id in cluster_embryos:
            embryo_data = df_cluster[df_cluster['embryo_id'] == embryo_id]
            if len(embryo_data) == 0:
                continue

            genotype = embryo_data['genotype'].iloc[0]
            color = genotype_colors.get(genotype, '#808080')  # Gray for unknown

            ax.plot(embryo_data['hpf'], embryo_data['metric_value'],
                   alpha=0.3, linewidth=0.8, color=color)

        # Plot mean trajectory (bold black)
        mean_traj = df_cluster.groupby('hpf')['metric_value'].mean()
        ax.plot(mean_traj.index, mean_traj.values,
               linewidth=3, color='black', label='Mean', zorder=100)

        # Add trend line
        x = mean_traj.index.values
        y_trend = info['slope'] * x + (info['mean'] - info['slope'] * x.mean())
        ax.plot(x, y_trend, '--', linewidth=2, color='black', alpha=0.5, label='Trend')

        # Labels
        ax.set_title(f"Cluster {cluster_id}: {info['label'].upper()}\n"
                    f"n={info['n_embryos']}, slope={info['slope']:.4f}, mean={info['mean']:.3f}",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (hpf)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Curvature (normalized)', fontsize=10)
        ax.axhline(y=WILDTYPE_THRESHOLD, color='gray', linestyle=':', alpha=0.5,
                  label=f'WT threshold ({WILDTYPE_THRESHOLD})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Add global legend for genotypes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=genotype_colors['cep290_wildtype'], label='Wildtype'),
        Patch(facecolor=genotype_colors['cep290_heterozygous'], label='Heterozygous'),
        Patch(facecolor=genotype_colors['cep290_homozygous'], label='Homozygous')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, title='Genotype')

    plt.suptitle(f'Cluster Inspection (Colored by Genotype) - Experiment {EXPERIMENT_ID}\n'
                 f'(k={n_clusters}, {len(cluster_assignments)} embryos)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {save_path}")

    plt.close(fig)


def run_phase_a():
    """
    Phase A: DTW clustering and visual inspection.
    """
    print("=" * 80)
    print("PHASE A: DTW CLUSTERING AND INSPECTION")
    print("=" * 80)

    # Load data
    df = load_and_prepare_data()

    # Extract trajectories
    trajectories, embryo_ids, common_grid, df_interpolated = extract_trajectories_for_clustering(df)

    # Run clustering
    cluster_assignments, posteriors, D = run_dtw_clustering(trajectories, embryo_ids)

    # Label clusters by trend
    cluster_labels, embryo_labels = label_clusters_by_trend(
        df_interpolated, cluster_assignments, common_grid
    )

    # Add genotype to embryo labels
    genotype_map = df[[EMBRYO_ID_COL, 'genotype']].drop_duplicates()
    embryo_labels = embryo_labels.merge(genotype_map, on='embryo_id', how='left')

    # Generate inspection plots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Colored by trend label
    plot_cluster_inspection(
        df_interpolated, cluster_assignments, cluster_labels,
        save_path=FIGURES_DIR / 'cluster_inspection.png'
    )

    # Plot 2: Colored by genotype
    plot_cluster_inspection_by_genotype(
        df_interpolated, cluster_assignments, cluster_labels, df,
        save_path=FIGURES_DIR / 'cluster_inspection_by_genotype.png'
    )

    # Generate cluster-genotype summary
    genotype_summary = generate_cluster_genotype_summary(cluster_assignments, df)
    genotype_summary.to_csv(OUTPUT_DIR / 'cluster_genotype_summary.csv', index=False)
    print(f"Saved cluster-genotype summary to: {OUTPUT_DIR / 'cluster_genotype_summary.csv'}")

    # Save embryo labels (now with genotype)
    embryo_labels.to_csv(OUTPUT_DIR / 'trajectory_trend_labels.csv', index=False)
    print(f"Saved embryo labels to: {OUTPUT_DIR / 'trajectory_trend_labels.csv'}")

    # Save cluster info
    cluster_info_df = pd.DataFrame([
        {'cluster_id': cid, **info}
        for cid, info in cluster_labels.items()
    ])
    cluster_info_df.to_csv(OUTPUT_DIR / 'cluster_info.csv', index=False)

    # Summary
    print("\n" + "=" * 80)
    print("PHASE A COMPLETE - PLEASE INSPECT CLUSTERS")
    print("=" * 80)
    print(f"\nCluster inspection plots:")
    print(f"  - By trend label: {FIGURES_DIR / 'cluster_inspection.png'}")
    print(f"  - By genotype:    {FIGURES_DIR / 'cluster_inspection_by_genotype.png'}")

    print(f"\nCluster summary:")
    for cid, info in sorted(cluster_labels.items()):
        print(f"  Cluster {cid}: {info['label']} (n={info['n_embryos']})")

    # Count trend distribution
    trend_counts = embryo_labels['trend_label'].value_counts()
    print(f"\nTrend distribution:")
    for label, count in trend_counts.items():
        print(f"  {label}: {count} embryos")

    # Print genotype × cluster table
    print(f"\nGenotype composition by cluster:")
    genotype_pivot = genotype_summary.pivot(index='cluster_id', columns='genotype', values='count').fillna(0).astype(int)
    print(genotype_pivot.to_string())

    return df, embryo_labels


# =============================================================================
# Phase B: Classification
# =============================================================================

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
    meta_cols = [EMBRYO_ID_COL, 'genotype'] if 'genotype' in df.columns else [EMBRYO_ID_COL]
    meta_df = df[meta_cols].drop_duplicates(subset=[EMBRYO_ID_COL])
    agg = agg.merge(meta_df, on=EMBRYO_ID_COL, how='left')

    print(f"  Binned to {len(agg)} (embryo, time_bin) samples")
    print(f"  Time bins: {sorted(agg['time_bin'].unique())}")

    return agg


def classify_at_time_bin(df_labeled, time_bin, embedding_cols, n_folds=5):
    """
    Run classification at a single time bin.

    Returns
    -------
    dict with f1_mean, f1_std, precision, recall, n_samples
    """
    # Filter to this time bin and mutant embryos only (exclude wildtype)
    df_bin = df_labeled[
        (df_labeled['time_bin'] == time_bin) &
        (df_labeled['trend_label'] != 'wildtype')
    ].copy()

    if len(df_bin) < 10:
        return {
            'time_bin': time_bin,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'precision_mean': np.nan,
            'recall_mean': np.nan,
            'n_samples': len(df_bin),
            'n_increasing': 0,
            'n_decreasing': 0
        }

    # Prepare features and labels
    X = df_bin[embedding_cols].values
    y = (df_bin['trend_label'] == 'increasing').astype(int).values  # 1=increasing, 0=decreasing
    groups = df_bin[EMBRYO_ID_COL].values

    # Check class balance
    n_increasing = y.sum()
    n_decreasing = len(y) - n_increasing

    if n_increasing < 2 or n_decreasing < 2:
        return {
            'time_bin': time_bin,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'precision_mean': np.nan,
            'recall_mean': np.nan,
            'n_samples': len(df_bin),
            'n_increasing': n_increasing,
            'n_decreasing': n_decreasing
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
            'n_samples': len(df_bin),
            'n_increasing': n_increasing,
            'n_decreasing': n_decreasing
        }

    gkf = GroupKFold(n_splits=n_folds_actual)

    f1_scores = []
    precision_scores = []
    recall_scores = []

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
            ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))

    if not f1_scores:
        return {
            'time_bin': time_bin,
            'f1_mean': np.nan,
            'f1_std': np.nan,
            'precision_mean': np.nan,
            'recall_mean': np.nan,
            'n_samples': len(df_bin),
            'n_increasing': n_increasing,
            'n_decreasing': n_decreasing
        }

    return {
        'time_bin': time_bin,
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0,
        'precision_mean': np.mean(precision_scores),
        'recall_mean': np.mean(recall_scores),
        'n_samples': len(df_bin),
        'n_increasing': n_increasing,
        'n_decreasing': n_decreasing
    }


def plot_f1_vs_time(results_df, save_path):
    """
    Plot F1-score vs time bin.
    """
    print("Generating F1-score vs time bin plot...")

    df_valid = results_df[results_df['f1_mean'].notna()].copy()

    if len(df_valid) == 0:
        print("  No valid results to plot!")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot F1-score
    ax1 = axes[0]
    ax1.errorbar(df_valid['time_bin'], df_valid['f1_mean'],
                yerr=df_valid['f1_std'], marker='o', linewidth=2,
                markersize=8, capsize=4, color='#1f77b4')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')
    ax1.axhline(y=0.7, color='green', linestyle=':', alpha=0.7, label='Target threshold (0.7)')
    ax1.set_ylabel('F1-Score', fontsize=12)
    ax1.set_title(f'F1-Score vs Time Bin - Experiment {EXPERIMENT_ID}\n'
                 f'(Predicting trajectory trend: increasing vs decreasing)',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Find earliest time with F1 > 0.7
    above_threshold = df_valid[df_valid['f1_mean'] >= 0.7]
    if not above_threshold.empty:
        earliest = above_threshold['time_bin'].min()
        ax1.axvline(x=earliest, color='green', linestyle='-', alpha=0.5)
        ax1.annotate(f'Earliest predictive: {earliest}h',
                    xy=(earliest, 0.7), xytext=(earliest + 5, 0.85),
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))

    # Plot sample sizes
    ax2 = axes[1]
    width = 1.5
    ax2.bar(df_valid['time_bin'] - width/2, df_valid['n_increasing'],
           width=width, label='Increasing', color='#d62728', alpha=0.7)
    ax2.bar(df_valid['time_bin'] + width/2, df_valid['n_decreasing'],
           width=width, label='Decreasing', color='#1f77b4', alpha=0.7)
    ax2.set_xlabel('Time Bin (hpf)', fontsize=12)
    ax2.set_ylabel('Number of Embryos', fontsize=12)
    ax2.set_title('Sample Size per Time Bin', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {save_path}")

    plt.close(fig)


def run_phase_b(df=None, embryo_labels=None):
    """
    Phase B: Per-time-bin classification.
    """
    print("\n" + "=" * 80)
    print("PHASE B: PER-TIME-BIN CLASSIFICATION")
    print("=" * 80)

    # Load data if not provided
    if df is None:
        df = load_and_prepare_data()

    if embryo_labels is None:
        labels_path = OUTPUT_DIR / 'trajectory_trend_labels.csv'
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Embryo labels not found at {labels_path}. "
                "Please run Phase A first (--phase clustering)"
            )
        embryo_labels = pd.read_csv(labels_path)
        print(f"Loaded embryo labels from: {labels_path}")

    # Bin embeddings
    df_binned = bin_embeddings_by_time(df, bin_width=BIN_WIDTH)

    # Merge with trend labels
    df_labeled = df_binned.merge(embryo_labels, on=EMBRYO_ID_COL, how='inner')
    print(f"Merged: {len(df_labeled)} samples with trend labels")

    # Get embedding columns
    embedding_cols = [c for c in df_labeled.columns if c.endswith('_binned') and 'z_mu' in c]
    print(f"Using {len(embedding_cols)} embedding features")

    # Get unique time bins
    time_bins = sorted(df_labeled['time_bin'].unique())
    print(f"Time bins to classify: {time_bins}")

    # Run classification at each time bin
    print("\nRunning classification...")
    results = []

    for time_bin in time_bins:
        result = classify_at_time_bin(df_labeled, time_bin, embedding_cols, n_folds=N_FOLDS)
        results.append(result)

        if not np.isnan(result['f1_mean']):
            print(f"  Time {time_bin:3d}h: F1={result['f1_mean']:.3f} +/- {result['f1_std']:.3f} "
                  f"(n={result['n_samples']}, inc={result['n_increasing']}, dec={result['n_decreasing']})")

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'classification_results.csv', index=False)
    print(f"\nSaved results to: {OUTPUT_DIR / 'classification_results.csv'}")

    # Generate plot
    plot_f1_vs_time(results_df, save_path=FIGURES_DIR / 'f1_vs_time_bin.png')

    # Summary
    print("\n" + "=" * 80)
    print("PHASE B COMPLETE")
    print("=" * 80)

    df_valid = results_df[results_df['f1_mean'].notna()]
    if len(df_valid) > 0:
        print(f"\nF1-Score Summary:")
        print(f"  Mean F1: {df_valid['f1_mean'].mean():.3f}")
        print(f"  Max F1:  {df_valid['f1_mean'].max():.3f} at time {df_valid.loc[df_valid['f1_mean'].idxmax(), 'time_bin']}h")
        print(f"  Min F1:  {df_valid['f1_mean'].min():.3f} at time {df_valid.loc[df_valid['f1_mean'].idxmin(), 'time_bin']}h")

        above_threshold = df_valid[df_valid['f1_mean'] >= 0.7]
        if not above_threshold.empty:
            earliest = above_threshold['time_bin'].min()
            print(f"\n  Earliest predictive timepoint (F1 >= 0.7): {earliest}h")
        else:
            print(f"\n  No timepoint achieved F1 >= 0.7")

    return results_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Trajectory Trend Classification')
    parser.add_argument('--phase', choices=['clustering', 'classification', 'both'],
                       default='both', help='Which phase to run')
    args = parser.parse_args()

    print("=" * 80)
    print("TRAJECTORY TREND CLASSIFICATION BASELINE")
    print("=" * 80)
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Phase: {args.phase}")
    print("=" * 80)

    df = None
    embryo_labels = None

    if args.phase in ['clustering', 'both']:
        df, embryo_labels = run_phase_a()

    if args.phase in ['classification', 'both']:
        run_phase_b(df=df, embryo_labels=embryo_labels)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
