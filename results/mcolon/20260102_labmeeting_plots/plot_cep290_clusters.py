"""
Plot CEP290 homozygous phenotype clusters vs WT and Het backgrounds.

Creates two plots:
1. Single row faceted by cluster
2. Two rows: first row faceted by experiment_id, second row faceted by cluster
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Paths
DATA_DIR = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction')
OUTPUT_DIR = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260102_labmeeting_plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the clustering data
print("Loading clustering data...")
with open(DATA_DIR / 'data/clustering_data__early_homo.pkl', 'rb') as f:
    clustering_data = pickle.load(f)

# Unpack
D_cep290 = clustering_data['D_cep290']
embryo_ids_cep290 = clustering_data['embryo_ids_cep290']
time_grid_cep290 = clustering_data['time_grid_cep290']
metrics_cep290 = clustering_data['metrics_cep290']
df = clustering_data['df_cep290_earyltimepoints']

# Load k-medoids results
print("Loading k-medoids results...")
with open(DATA_DIR / 'kmedoids_k_selection_early_timepoints_cep290_data/k_results.pkl', 'rb') as f:
    k_results_kmedoids = pickle.load(f)

# Extract cluster assignments for k=5
cluster_labels = k_results_kmedoids['clustering_by_k'][5]['assignments']['cluster_labels']

# Define cluster names
cluster_names_k5 = {
    0: 'outlier',
    1: 'bumby',
    2: 'low_to_high',
    3: 'low_to_high',
    4: 'high_to_low',
}

# Create a mapping from embryo_id to cluster name
embryo_cluster_map = {}
for embryo_id, cluster_id in zip(embryo_ids_cep290, cluster_labels):
    embryo_cluster_map[embryo_id] = cluster_names_k5[cluster_id]

# Add cluster column to homozygous embryos
df['cluster'] = None
homo_mask = df['genotype'].str.contains('homo', case=False, na=False)
df.loc[homo_mask, 'cluster'] = df.loc[homo_mask, 'embryo_id'].map(embryo_cluster_map)

# Check distribution
print("\nCluster distribution:")
print(df[homo_mask]['cluster'].value_counts())

# Define colors
COLORS = {
    'cep290_wildtype': '#2E7D32',      # Green
    'cep290_wt': '#2E7D32',            # Green
    'cep290_heterozygous': '#FFA500',  # Orange
    'cep290_het': '#FFA500',           # Orange
    'outlier': '#D32F2F',              # Red
    'bumby': '#9467BD',                # Purple
    'low_to_high': '#17BECF',          # Cyan
    'high_to_low': '#E377C2',          # Pink
}

# Helper function to compute trend line
def compute_trend_line(df_subset, x_col='predicted_stage_hpf', y_col='baseline_deviation_normalized',
                       bin_width=0.5, smooth_sigma=1.5):
    """Compute binned median trend line with smoothing."""
    times = df_subset[x_col].values
    metrics = df_subset[y_col].values

    # Remove NaNs
    mask = ~(np.isnan(times) | np.isnan(metrics))
    times = times[mask]
    metrics = metrics[mask]

    if len(times) == 0:
        return None, None

    # Create bins
    t_min, t_max = times.min(), times.max()
    bins = np.arange(t_min, t_max + bin_width, bin_width)

    if len(bins) < 2:
        return None, None

    # Assign to bins and compute median
    bin_indices = np.digitize(times, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    bin_times = []
    bin_medians = []

    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        bin_values = metrics[bin_mask]

        if len(bin_values) >= 2:
            bin_center = (bins[i] + bins[i + 1]) / 2
            bin_times.append(bin_center)
            bin_medians.append(np.median(bin_values))

    if len(bin_times) == 0:
        return None, None

    bin_times = np.array(bin_times)
    bin_medians = np.array(bin_medians)

    # Apply gaussian smoothing
    if smooth_sigma and len(bin_medians) > 3:
        bin_medians = gaussian_filter1d(bin_medians, sigma=smooth_sigma)

    return bin_times, bin_medians


def plot_cluster_facets(df, unique_clusters, output_path, title_suffix=''):
    """Plot clusters in faceted layout."""
    # Create figure with subplots
    n_clusters = len(unique_clusters)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), sharey=True)

    if n_clusters == 1:
        axes = [axes]

    # Get global y-limits
    y_col = 'baseline_deviation_normalized'
    x_col = 'predicted_stage_hpf'
    y_min = df[y_col].min()
    y_max = df[y_col].max()

    # Plot each cluster in its own facet
    for idx, cluster_name in enumerate(unique_clusters):
        ax = axes[idx]

        # 1. Plot WT individual trajectories (very faded green in background)
        wt_df = df[df['genotype'].str.contains('wildtype|wt', case=False, na=False)]
        for embryo_id in wt_df['embryo_id'].unique():
            embryo_data = wt_df[wt_df['embryo_id'] == embryo_id].sort_values(x_col)
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=COLORS['cep290_wildtype'], alpha=0.02, linewidth=0.5, zorder=1)

        # 2. Plot het individual trajectories (very faded orange in background)
        het_df = df[df['genotype'].str.contains('het', case=False, na=False)]
        for embryo_id in het_df['embryo_id'].unique():
            embryo_data = het_df[het_df['embryo_id'] == embryo_id].sort_values(x_col)
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=COLORS['cep290_heterozygous'], alpha=0.02, linewidth=0.5, zorder=1)

        # 3. Plot this cluster's individual trajectories (faded cluster color)
        cluster_df = df[df['cluster'] == cluster_name]
        cluster_color = COLORS.get(cluster_name, '#808080')
        for embryo_id in cluster_df['embryo_id'].unique():
            embryo_data = cluster_df[cluster_df['embryo_id'] == embryo_id].sort_values(x_col)
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=cluster_color, alpha=0.3, linewidth=1.0, zorder=2)

        # 4. Plot WT trend line (subdued green)
        wt_times, wt_trend = compute_trend_line(wt_df, x_col, y_col)
        if wt_times is not None:
            ax.plot(wt_times, wt_trend, color=COLORS['cep290_wildtype'],
                    linewidth=2.0, label='WT', zorder=5, alpha=0.6)

        # 5. Plot het trend line (subdued orange)
        het_times, het_trend = compute_trend_line(het_df, x_col, y_col)
        if het_times is not None:
            ax.plot(het_times, het_trend, color=COLORS['cep290_heterozygous'],
                    linewidth=2.0, label='Het', zorder=5, alpha=0.6)

        # 6. Plot this cluster's trend line (THICK cluster color)
        cluster_times, cluster_trend = compute_trend_line(cluster_df, x_col, y_col)
        if cluster_times is not None:
            ax.plot(cluster_times, cluster_trend, color=cluster_color,
                    linewidth=4.0, label=cluster_name, zorder=6)

        # Styling
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (hpf)', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Baseline Deviation (Normalized)', fontsize=11, fontweight='bold')
        ax.set_title(f'{cluster_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Overall title
    fig.suptitle(f'CEP290 Homozygous Phenotype Clusters vs WT and Het{title_suffix}',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_experiment_and_cluster_facets(df, unique_experiments, unique_clusters, output_path):
    """Plot with two rows: experiments in first row, clusters in second row."""
    n_experiments = len(unique_experiments)
    n_clusters = len(unique_clusters)

    # Create figure with GridSpec for flexible layout
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, max(n_experiments, n_clusters), figure=fig, hspace=0.3, wspace=0.3)

    y_col = 'baseline_deviation_normalized'
    x_col = 'predicted_stage_hpf'
    y_min = df[y_col].min()
    y_max = df[y_col].max()

    # ROW 1: Experiments
    for idx, exp_id in enumerate(unique_experiments):
        ax = fig.add_subplot(gs[0, idx])

        # Filter by experiment
        exp_df = df[df['experiment_id'] == exp_id]

        # Plot WT
        wt_df = exp_df[exp_df['genotype'].str.contains('wildtype|wt', case=False, na=False)]
        for embryo_id in wt_df['embryo_id'].unique():
            embryo_data = wt_df[wt_df['embryo_id'] == embryo_id].sort_values(x_col)
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=COLORS['cep290_wildtype'], alpha=0.02, linewidth=0.5, zorder=1)

        # Plot het
        het_df = exp_df[exp_df['genotype'].str.contains('het', case=False, na=False)]
        for embryo_id in het_df['embryo_id'].unique():
            embryo_data = het_df[het_df['embryo_id'] == embryo_id].sort_values(x_col)
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=COLORS['cep290_heterozygous'], alpha=0.02, linewidth=0.5, zorder=1)

        # Plot homo (all clusters together)
        homo_df = exp_df[exp_df['genotype'].str.contains('homo', case=False, na=False)]
        for embryo_id in homo_df['embryo_id'].unique():
            embryo_data = homo_df[homo_df['embryo_id'] == embryo_id].sort_values(x_col)
            cluster = embryo_data['cluster'].iloc[0] if len(embryo_data) > 0 else None
            cluster_color = COLORS.get(cluster, '#D32F2F')
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=cluster_color, alpha=0.2, linewidth=0.8, zorder=2)

        # Trend lines
        wt_times, wt_trend = compute_trend_line(wt_df, x_col, y_col)
        if wt_times is not None:
            ax.plot(wt_times, wt_trend, color=COLORS['cep290_wildtype'],
                    linewidth=2.0, label='WT', zorder=5, alpha=0.7)

        het_times, het_trend = compute_trend_line(het_df, x_col, y_col)
        if het_times is not None:
            ax.plot(het_times, het_trend, color=COLORS['cep290_heterozygous'],
                    linewidth=2.0, label='Het', zorder=5, alpha=0.7)

        homo_times, homo_trend = compute_trend_line(homo_df, x_col, y_col)
        if homo_times is not None:
            ax.plot(homo_times, homo_trend, color='#D32F2F',
                    linewidth=3.0, label='Homo (all)', zorder=6)

        # Styling
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (hpf)', fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Row 1: By Experiment\nBaseline Deviation (Normalized)',
                         fontsize=10, fontweight='bold')
        ax.set_title(f'Exp: {exp_id}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ROW 2: Clusters
    for idx, cluster_name in enumerate(unique_clusters):
        ax = fig.add_subplot(gs[1, idx])

        # Plot WT
        wt_df = df[df['genotype'].str.contains('wildtype|wt', case=False, na=False)]
        for embryo_id in wt_df['embryo_id'].unique():
            embryo_data = wt_df[wt_df['embryo_id'] == embryo_id].sort_values(x_col)
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=COLORS['cep290_wildtype'], alpha=0.05, linewidth=0.5, zorder=1)

        # Plot het
        het_df = df[df['genotype'].str.contains('het', case=False, na=False)]
        for embryo_id in het_df['embryo_id'].unique():
            embryo_data = het_df[het_df['embryo_id'] == embryo_id].sort_values(x_col)
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=COLORS['cep290_heterozygous'], alpha=0.05, linewidth=0.5, zorder=1)

        # Plot this cluster
        cluster_df = df[df['cluster'] == cluster_name]
        cluster_color = COLORS.get(cluster_name, '#808080')
        for embryo_id in cluster_df['embryo_id'].unique():
            embryo_data = cluster_df[cluster_df['embryo_id'] == embryo_id].sort_values(x_col)
            ax.plot(embryo_data[x_col], embryo_data[y_col],
                    color=cluster_color, alpha=0.3, linewidth=1.0, zorder=2)

        # Trend lines
        wt_times, wt_trend = compute_trend_line(wt_df, x_col, y_col)
        if wt_times is not None:
            ax.plot(wt_times, wt_trend, color=COLORS['cep290_wildtype'],
                    linewidth=2.0, label='WT', zorder=5, alpha=0.6)

        het_times, het_trend = compute_trend_line(het_df, x_col, y_col)
        if het_times is not None:
            ax.plot(het_times, het_trend, color=COLORS['cep290_heterozygous'],
                    linewidth=2.0, label='Het', zorder=5, alpha=0.6)

        cluster_times, cluster_trend = compute_trend_line(cluster_df, x_col, y_col)
        if cluster_times is not None:
            ax.plot(cluster_times, cluster_trend, color=cluster_color,
                    linewidth=4.0, label=cluster_name, zorder=6)

        # Styling
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (hpf)', fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Row 2: By Cluster\nBaseline Deviation (Normalized)',
                         fontsize=10, fontweight='bold')
        ax.set_title(f'{cluster_name.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Overall title
    fig.suptitle('CEP290 Phenotypes: By Experiment (top) and By Cluster (bottom)',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_clusters_by_experiment(df, unique_experiments, unique_clusters, output_path):
    """Plot clusters separated by experiment to check for batch effects.

    Row 1: Each cluster from Experiment 1 only
    Row 2: Each cluster from Experiment 2 only
    Each overlaid with very faint WT and Het backgrounds.
    """
    from matplotlib.gridspec import GridSpec

    n_experiments = len(unique_experiments)
    n_clusters = len(unique_clusters)

    fig = plt.figure(figsize=(5 * n_clusters, 10))
    gs = GridSpec(n_experiments, n_clusters, figure=fig, hspace=0.3, wspace=0.3)

    y_col = 'baseline_deviation_normalized'
    x_col = 'predicted_stage_hpf'
    y_min = df[y_col].min()
    y_max = df[y_col].max()

    # Get WT and Het data for backgrounds (across all experiments)
    wt_df = df[df['genotype'].str.contains('wildtype|wt', case=False, na=False)]
    het_df = df[df['genotype'].str.contains('het', case=False, na=False)]

    for exp_idx, exp_id in enumerate(unique_experiments):
        for cluster_idx, cluster_name in enumerate(unique_clusters):
            ax = fig.add_subplot(gs[exp_idx, cluster_idx])

            # Filter to this experiment
            exp_df = df[df['experiment_id'] == exp_id]

            # 1. Plot WT individual trajectories (very faint green)
            wt_exp_df = wt_df[wt_df['experiment_id'] == exp_id]
            for embryo_id in wt_exp_df['embryo_id'].unique():
                embryo_data = wt_exp_df[wt_exp_df['embryo_id'] == embryo_id].sort_values(x_col)
                ax.plot(embryo_data[x_col], embryo_data[y_col],
                        color=COLORS['cep290_wildtype'], alpha=0.03, linewidth=0.4, zorder=1)

            # 2. Plot het individual trajectories (very faint orange)
            het_exp_df = het_df[het_df['experiment_id'] == exp_id]
            for embryo_id in het_exp_df['embryo_id'].unique():
                embryo_data = het_exp_df[het_exp_df['embryo_id'] == embryo_id].sort_values(x_col)
                ax.plot(embryo_data[x_col], embryo_data[y_col],
                        color=COLORS['cep290_heterozygous'], alpha=0.03, linewidth=0.4, zorder=1)

            # 3. Plot this cluster's embryos from this experiment
            cluster_exp_df = exp_df[exp_df['cluster'] == cluster_name]
            cluster_color = COLORS.get(cluster_name, '#808080')

            for embryo_id in cluster_exp_df['embryo_id'].unique():
                embryo_data = cluster_exp_df[cluster_exp_df['embryo_id'] == embryo_id].sort_values(x_col)
                ax.plot(embryo_data[x_col], embryo_data[y_col],
                        color=cluster_color, alpha=0.3, linewidth=1.0, zorder=2)

            # 4. Plot WT trend line (faint green)
            wt_times, wt_trend = compute_trend_line(wt_exp_df, x_col, y_col)
            if wt_times is not None:
                ax.plot(wt_times, wt_trend, color=COLORS['cep290_wildtype'],
                        linewidth=1.5, label='WT', zorder=5, alpha=0.4)

            # 5. Plot het trend line (faint orange)
            het_times, het_trend = compute_trend_line(het_exp_df, x_col, y_col)
            if het_times is not None:
                ax.plot(het_times, het_trend, color=COLORS['cep290_heterozygous'],
                        linewidth=1.5, label='Het', zorder=5, alpha=0.4)

            # 6. Plot cluster trend line (THICK cluster color)
            cluster_times, cluster_trend = compute_trend_line(cluster_exp_df, x_col, y_col)
            if cluster_times is not None:
                ax.plot(cluster_times, cluster_trend, color=cluster_color,
                        linewidth=4.0, label=cluster_name, zorder=6)

            # Styling
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('Time (hpf)', fontsize=10, fontweight='bold')

            # Y-label for first column
            if cluster_idx == 0:
                ax.set_ylabel(f'Exp: {exp_id}\nBaseline Deviation (Normalized)',
                             fontsize=10, fontweight='bold')

            # Title for top row
            if exp_idx == 0:
                ax.set_title(f'{cluster_name.replace("_", " ").title()}',
                            fontsize=11, fontweight='bold')

            ax.legend(loc='best', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add n count annotation
            n_embryos = cluster_exp_df['embryo_id'].nunique()
            ax.text(0.02, 0.98, f'n={n_embryos}', transform=ax.transAxes,
                   fontsize=8, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Overall title
    fig.suptitle('CEP290 Clusters by Experiment (checking for batch effects)',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# Get unique cluster names (excluding None and NaN)
unique_clusters = [c for c in df[homo_mask]['cluster'].unique() if c is not None and pd.notna(c)]
unique_clusters = sorted(unique_clusters)
print(f"\nUnique clusters: {unique_clusters}")

# PLOT 1: Single row by cluster
print("\nGenerating Plot 1: Single row by cluster...")
plot_cluster_facets(
    df,
    unique_clusters,
    OUTPUT_DIR / 'cep290_clusters_faceted.png'
)

# PLOT 2: Two rows - experiments and clusters
print("\nGenerating Plot 2: Two rows (experiments + clusters)...")
unique_experiments = sorted(df['experiment_id'].unique())
print(f"Unique experiments: {unique_experiments}")

plot_experiment_and_cluster_facets(
    df,
    unique_experiments,
    unique_clusters,
    OUTPUT_DIR / 'cep290_experiments_and_clusters.png'
)

# PLOT 3: Clusters by experiment (batch effect check)
print("\nGenerating Plot 3: Clusters separated by experiment (batch effect check)...")
plot_clusters_by_experiment(
    df,
    unique_experiments,
    unique_clusters,
    OUTPUT_DIR / 'cep290_clusters_by_experiment_batch_check.png'
)

print("\n✓ All plots generated successfully!")
