"""
Plotting utilities for tmem67 clustering analysis.

Focus on cluster-level averages rather than individual embryo trajectories.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import CLUSTER_COLORS, PLOTLY_ALPHA_INDIVIDUAL, PLOTLY_WIDTH_MEAN, PLOTLY_WIDTH_INDIVIDUAL
from cluster_analysis_utils import compute_cluster_mean_trajectory


def plot_trajectories_by_cluster(df_interpolated, posteriors, classification,
                                  cluster_chars, output_path):
    """
    Plot individual trajectories colored by cluster assignment (PNG).

    Layout: 1 row × k columns (one subplot per cluster)
    Shows individual trajectories (faded) + mean trajectory (bold)
    Colors indicate membership quality (core=solid, uncertain=dashed, outlier=dotted)

    Parameters
    ----------
    df_interpolated : DataFrame
        Interpolated trajectory data
    posteriors : dict
        Output from analyze_bootstrap_results()
    classification : dict
        Output from classify_membership_2d()
    cluster_chars : DataFrame
        Cluster characteristics table
    output_path : Path or str
        Path to save figure
    """
    modal_cluster = posteriors['modal_cluster']
    embryo_ids = posteriors['embryo_ids']
    categories = classification['category']
    n_clusters = posteriors['n_clusters']

    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))
    if n_clusters == 1:
        axes = [axes]

    fig.suptitle(f'Trajectories by Cluster (k={n_clusters})', fontsize=14, fontweight='bold')

    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]

        # Get embryos in this cluster
        cluster_mask = modal_cluster == cluster_id
        cluster_embryo_ids = [eid for eid, m in zip(embryo_ids, cluster_mask) if m]

        # Filter DataFrame
        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryo_ids)]

        if len(df_cluster) == 0:
            ax.text(0.5, 0.5, 'No embryos', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            continue

        # Get cluster characteristics
        cluster_row = cluster_chars[cluster_chars['cluster_id'] == cluster_id].iloc[0]
        is_mutant = cluster_row['is_putative_mutant']
        cluster_avg = cluster_row['cluster_average']

        # Plot individual trajectories with quality-based linestyle
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        for eid in cluster_embryo_ids:
            embryo_data = df_cluster[df_cluster['embryo_id'] == eid]

            # Determine linestyle based on membership quality
            idx = embryo_ids.index(eid)
            category = categories[idx]

            if category == 'core':
                linestyle = '-'
                alpha = 0.3
            elif category == 'uncertain':
                linestyle = '--'
                alpha = 0.25
            else:  # outlier
                linestyle = ':'
                alpha = 0.2

            ax.plot(embryo_data['hpf'], embryo_data['metric_value'],
                   color=color, linestyle=linestyle, alpha=alpha, linewidth=0.8)

        # Plot mean trajectory (MAIN FOCUS)
        mean_traj = compute_cluster_mean_trajectory(df_cluster)
        ax.plot(mean_traj['hpf'], mean_traj['mean_value'],
               color=color, linewidth=2.5, label='Cluster Mean', zorder=10)

        # Add horizontal line at mutant threshold
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5,
                  label='Mutant threshold')

        # Styling
        title_prefix = "MUTANT" if is_mutant else "WT-like"
        ax.set_title(f'{title_prefix} Cluster {cluster_id}\n'
                    f'n={len(cluster_embryo_ids)}, avg={cluster_avg:.3f}',
                    fontweight='bold')
        ax.set_xlabel('Time (hpf)')
        ax.set_ylabel('Normalized Baseline Deviation')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cluster_means(df_interpolated, posteriors, cluster_chars, output_path):
    """
    Plot cluster mean trajectories on single plot for comparison (PNG).

    This is the MAIN visualization showing cluster-level averages.

    Parameters
    ----------
    df_interpolated : DataFrame
        Interpolated trajectory data
    posteriors : dict
        Output from analyze_bootstrap_results()
    cluster_chars : DataFrame
        Cluster characteristics table
    output_path : Path or str
        Path to save figure
    """
    modal_cluster = posteriors['modal_cluster']
    embryo_ids = posteriors['embryo_ids']
    n_clusters = posteriors['n_clusters']

    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster_id in range(n_clusters):
        # Get embryos in cluster
        cluster_mask = modal_cluster == cluster_id
        cluster_embryo_ids = [eid for eid, m in zip(embryo_ids, cluster_mask) if m]

        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryo_ids)]

        if len(df_cluster) == 0:
            continue

        # Compute mean
        mean_traj = compute_cluster_mean_trajectory(df_cluster)

        # Get characteristics
        cluster_row = cluster_chars[cluster_chars['cluster_id'] == cluster_id].iloc[0]
        is_mutant = cluster_row['is_putative_mutant']
        cluster_avg = cluster_row['cluster_average']

        # Plot
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        marker = 'o' if is_mutant else 's'
        label = f"Cluster {cluster_id} ({'MUTANT' if is_mutant else 'WT-like'}, n={len(cluster_embryo_ids)}, avg={cluster_avg:.3f})"

        ax.plot(mean_traj['hpf'], mean_traj['mean_value'],
               color=color, marker=marker, markersize=4, linewidth=2.5, label=label)

    # Add threshold line
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
              label='Mutant threshold (0.05)')

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('Normalized Baseline Deviation', fontsize=12)
    ax.set_title(f'Cluster Mean Trajectories (k={n_clusters})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_interactive_cluster_means(df_interpolated, posteriors, cluster_chars, output_path):
    """
    Create interactive Plotly plot focusing on CLUSTER MEANS (HTML).

    Shows all cluster mean trajectories on a single plot with hover showing
    cluster-level information (not individual embryo IDs).

    Parameters
    ----------
    df_interpolated : DataFrame
        Interpolated trajectory data
    posteriors : dict
        Output from analyze_bootstrap_results()
    cluster_chars : DataFrame
        Cluster characteristics table
    output_path : Path or str
        Path to save HTML file
    """
    modal_cluster = posteriors['modal_cluster']
    embryo_ids = posteriors['embryo_ids']
    n_clusters = posteriors['n_clusters']

    fig = go.Figure()

    # Plot cluster means
    for cluster_id in range(n_clusters):
        # Get embryos in cluster
        cluster_mask = modal_cluster == cluster_id
        cluster_embryo_ids = [eid for eid, m in zip(embryo_ids, cluster_mask) if m]

        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryo_ids)]

        if len(df_cluster) == 0:
            continue

        # Compute mean
        mean_traj = compute_cluster_mean_trajectory(df_cluster)

        # Get characteristics
        cluster_row = cluster_chars[cluster_chars['cluster_id'] == cluster_id].iloc[0]
        is_mutant = cluster_row['is_putative_mutant']
        cluster_avg = cluster_row['cluster_average']
        n_core = cluster_row['n_core']
        n_uncertain = cluster_row['n_uncertain']
        n_outlier = cluster_row['n_outlier']

        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        # Create customdata for hover
        customdata = np.column_stack((
            [cluster_id] * len(mean_traj['hpf']),
            [len(cluster_embryo_ids)] * len(mean_traj['hpf']),
            ['MUTANT' if is_mutant else 'WT-like'] * len(mean_traj['hpf']),
            [cluster_avg] * len(mean_traj['hpf']),
            [n_core] * len(mean_traj['hpf']),
            [n_uncertain] * len(mean_traj['hpf']),
            [n_outlier] * len(mean_traj['hpf'])
        ))

        # Plot cluster mean trajectory
        fig.add_trace(go.Scatter(
            x=mean_traj['hpf'],
            y=mean_traj['mean_value'],
            mode='lines+markers',
            line=dict(color=color, width=PLOTLY_WIDTH_MEAN),
            marker=dict(size=6),
            customdata=customdata,
            hovertemplate=(
                '<b>Cluster %{customdata[0]}</b> (%{customdata[2]})<br>'
                '<b>Time:</b> %{x:.2f} hpf<br>'
                '<b>Mean Value:</b> %{y:.4f}<br>'
                '<b>Cluster Avg (all time):</b> %{customdata[3]:.4f}<br>'
                '<b>n_embryos:</b> %{customdata[1]}<br>'
                '<b>Core/Uncertain/Outlier:</b> %{customdata[4]}/%{customdata[5]}/%{customdata[6]}'
                '<extra></extra>'
            ),
            name=f"Cluster {cluster_id} ({'MUTANT' if is_mutant else 'WT'})"
        ))

    # Add threshold line
    fig.add_hline(
        y=0.05,
        line=dict(color='red', dash='dash', width=1.5),
        opacity=0.7,
        annotation_text='Mutant threshold (0.05)',
        annotation_position='right'
    )

    fig.update_layout(
        title=f'Interactive Cluster Mean Trajectories (k={n_clusters}) - Hover for Details',
        xaxis_title='Time (hpf)',
        yaxis_title='Normalized Baseline Deviation',
        height=600,
        width=1000,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


def plot_metrics_vs_k(comparison_df, output_path):
    """
    Plot posterior metrics across k values (4-panel figure).

    Panels:
    1. Average max_p vs k
    2. Average entropy vs k
    3. Core fraction vs k
    4. Silhouette score vs k

    Parameters
    ----------
    comparison_df : DataFrame
        Comparison table with metrics for each k
    output_path : Path or str
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Posterior Metrics vs. Number of Clusters (k)', fontsize=14, fontweight='bold')

    # Panel 1: max_p
    ax = axes[0, 0]
    ax.plot(comparison_df['k'], comparison_df['avg_max_p'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('k (number of clusters)')
    ax.set_ylabel('Average max_p')
    ax.set_title('Confidence (higher is better)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(comparison_df['k'])

    # Panel 2: entropy
    ax = axes[0, 1]
    ax.plot(comparison_df['k'], comparison_df['avg_entropy'], marker='o', linewidth=2,
            markersize=8, color='orange')
    ax.set_xlabel('k (number of clusters)')
    ax.set_ylabel('Average entropy')
    ax.set_title('Uncertainty (lower is better)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(comparison_df['k'])

    # Panel 3: core fraction
    ax = axes[1, 0]
    ax.plot(comparison_df['k'], comparison_df['core_fraction'], marker='o', linewidth=2,
            markersize=8, color='green')
    ax.set_xlabel('k (number of clusters)')
    ax.set_ylabel('Core membership fraction')
    ax.set_title('Cluster Stability (higher is better)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(comparison_df['k'])

    # Panel 4: silhouette
    ax = axes[1, 1]
    ax.plot(comparison_df['k'], comparison_df['silhouette'], marker='o', linewidth=2,
            markersize=8, color='red')
    ax.set_xlabel('k (number of clusters)')
    ax.set_ylabel('Silhouette score')
    ax.set_title('Cluster Separation (higher is better)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(comparison_df['k'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_optimal_k_recommendation(comparison_df, scores, optimal_k, output_path):
    """
    Bar plot showing composite scores for each k value.

    Highlights the recommended optimal k.

    Parameters
    ----------
    comparison_df : DataFrame
        Comparison table with metrics
    scores : dict
        Dictionary mapping k -> composite_score
    optimal_k : int
        Recommended optimal k value
    output_path : Path or str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = list(scores.keys())
    score_values = list(scores.values())

    colors = ['#2ca02c' if k == optimal_k else '#1f77b4' for k in k_values]

    bars = ax.bar(k_values, score_values, color=colors, alpha=0.8, edgecolor='black')

    # Annotate bars with values
    for bar, k, score in zip(bars, k_values, score_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
               f'{score:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('k (number of clusters)', fontsize=12)
    ax.set_ylabel('Composite Score', fontsize=12)
    ax.set_title(f'Optimal k Selection (Recommended: k={optimal_k})',
                fontsize=14, fontweight='bold')
    ax.set_xticks(k_values)
    ax.set_ylim(0, max(score_values) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label=f'Recommended (k={optimal_k})'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
