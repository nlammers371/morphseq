# 3_membership.py
"""Classify embryos as core, uncertain, or outlier."""

import numpy as np
from sklearn.metrics import silhouette_samples
from typing import Dict, Tuple

# ============ CORE FUNCTIONS ============

def compute_membership_scores(C: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute membership strength for each embryo."""
    n = len(labels)
    scores = {}
    
    for i in range(n):
        cluster = labels[i]
        # Get co-associations with cluster mates
        cluster_mask = labels == cluster
        cluster_coassoc = C[i, cluster_mask]
        
        # Median co-association with cluster
        median_intra = np.median(cluster_coassoc[cluster_coassoc < 1])  # Exclude self
        
        # Mean co-association with other clusters
        other_mask = labels != cluster
        if other_mask.any():
            mean_inter = np.mean(C[i, other_mask])
        else:
            mean_inter = 0
        
        scores[i] = {
            'cluster': cluster,
            'intra_coassoc': median_intra,
            'inter_coassoc': mean_inter,
            'total_coassoc': np.mean(C[i, :])
        }
    
    return scores


def classify_members(C: np.ndarray, labels: np.ndarray, D: np.ndarray,
                     core_thresh: float = 0.7, outlier_thresh: float = 0.3) -> Dict:
    """Classify each embryo as core, uncertain, or outlier."""
    membership = compute_membership_scores(C, labels)
    silhouettes = silhouette_samples(D, labels, metric='precomputed')
    
    classification = {}
    for i, scores in membership.items():
        # Adaptive threshold based on bootstrap variance
        cluster = scores['cluster']
        cluster_mask = labels == cluster
        cluster_coassocs = C[cluster_mask][:, cluster_mask]
        variance = np.var(cluster_coassocs[np.triu_indices_from(cluster_coassocs, k=1)])
        
        # Adjust threshold if high variance
        adj_thresh = core_thresh - 0.1 if variance > 0.1 else core_thresh
        
        # Classify
        if scores['total_coassoc'] < outlier_thresh:
            category = 'outlier'
        elif scores['intra_coassoc'] >= adj_thresh and silhouettes[i] >= 0.2:
            category = 'core'
        else:
            category = 'uncertain'
        
        classification[i] = {
            'category': category,
            'cluster': cluster,
            'intra_coassoc': scores['intra_coassoc'],
            'silhouette': silhouettes[i],
            'threshold_used': adj_thresh
        }
    
    return classification


def get_core_indices(classification: Dict) -> np.ndarray:
    """Extract indices of core members."""
    return np.array([i for i, c in classification.items() if c['category'] == 'core'])


def get_uncertain_indices(classification: Dict) -> np.ndarray:
    """Extract indices of uncertain members."""
    return np.array([i for i, c in classification.items() if c['category'] == 'uncertain'])


def get_outlier_indices(classification: Dict) -> np.ndarray:
    """Extract indices of outliers."""
    return np.array([i for i, c in classification.items() if c['category'] == 'outlier'])


# ============ WRAPPER FUNCTIONS ============

def analyze_membership(D: np.ndarray, labels: np.ndarray, C: np.ndarray,
                       core_thresh: float = 0.7) -> Dict:
    """Full membership analysis."""
    classification = classify_members(C, labels, D, core_thresh)
    
    # Summary stats
    n_total = len(labels)
    n_core = sum(1 for c in classification.values() if c['category'] == 'core')
    n_uncertain = sum(1 for c in classification.values() if c['category'] == 'uncertain')
    n_outlier = sum(1 for c in classification.values() if c['category'] == 'outlier')
    
    # Per-cluster breakdown
    cluster_stats = {}
    for k in np.unique(labels):
        mask = labels == k
        cluster_members = [i for i in np.where(mask)[0]]
        cluster_stats[k] = {
            'total': len(cluster_members),
            'core': sum(1 for i in cluster_members if classification[i]['category'] == 'core'),
            'uncertain': sum(1 for i in cluster_members if classification[i]['category'] == 'uncertain'),
            'outlier': sum(1 for i in cluster_members if classification[i]['category'] == 'outlier')
        }
    
    return {
        'classification': classification,
        'summary': {
            'n_core': n_core,
            'n_uncertain': n_uncertain, 
            'n_outlier': n_outlier,
            'core_fraction': n_core / n_total
        },
        'cluster_stats': cluster_stats,
        'core_indices': get_core_indices(classification),
        'uncertain_indices': get_uncertain_indices(classification),
        'outlier_indices': get_outlier_indices(classification)
    }


# ============ PLOTTING FUNCTIONS ============

def plot_membership_distribution(classification, cluster_stats=None, title="Membership Distribution", dpi=100):
    """
    Plot membership category distribution.

    Parameters
    ----------
    classification : dict
        Classification results from analyze_membership
    cluster_stats : dict, optional
        Per-cluster statistics
    title : str
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)

    # ========== LEFT PANEL: Overall distribution ==========
    ax = axes[0]

    # Count categories
    categories = {'core': 0, 'uncertain': 0, 'outlier': 0}
    for c in classification.values():
        cat = c['category']
        if cat in categories:
            categories[cat] += 1

    colors = {'core': 'green', 'uncertain': 'yellow', 'outlier': 'red'}
    cat_colors = [colors[cat] for cat in categories.keys()]
    bars = ax.bar(categories.keys(), categories.values(), color=cat_colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Overall Membership Distribution', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(categories.values()) * 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    # ========== RIGHT PANEL: Per-cluster breakdown ==========
    ax = axes[1]

    if cluster_stats is not None:
        clusters = sorted(cluster_stats.keys())
        core_counts = [cluster_stats[c]['core'] for c in clusters]
        uncertain_counts = [cluster_stats[c]['uncertain'] for c in clusters]
        outlier_counts = [cluster_stats[c]['outlier'] for c in clusters]

        x = np.arange(len(clusters))
        width = 0.6

        p1 = ax.bar(x, core_counts, width, label='Core', color='green', alpha=0.7)
        p2 = ax.bar(x, uncertain_counts, width, bottom=core_counts,
                   label='Uncertain', color='yellow', alpha=0.7)
        p3 = ax.bar(x, outlier_counts, width,
                   bottom=np.array(core_counts) + np.array(uncertain_counts),
                   label='Outlier', color='red', alpha=0.7)

        ax.set_ylabel('Count', fontsize=11)
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_title('Per-Cluster Membership Breakdown', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in clusters])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No per-cluster statistics provided',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_membership_vs_k(all_k_membership, best_k=None, title="Membership Distribution Across K Values", dpi=100):
    """
    Plot membership category percentages as k varies.

    Parameters
    ----------
    all_k_membership : dict
        Membership results for all k values (k -> membership_results)
    best_k : int, optional
        Best k value to highlight with vertical line
    title : str
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    # Extract k values and compute percentages
    k_values = sorted(all_k_membership.keys())
    core_pcts = []
    uncertain_pcts = []
    outlier_pcts = []

    for k in k_values:
        summary = all_k_membership[k]['summary']
        n_total = (summary['n_core'] + summary['n_uncertain'] + summary['n_outlier'])
        if n_total > 0:
            core_pcts.append(100.0 * summary['n_core'] / n_total)
            uncertain_pcts.append(100.0 * summary['n_uncertain'] / n_total)
            outlier_pcts.append(100.0 * summary['n_outlier'] / n_total)
        else:
            core_pcts.append(0)
            uncertain_pcts.append(0)
            outlier_pcts.append(0)

    # Create two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)

    # ========== LEFT PANEL: Line plot ==========
    ax = axes[0]
    ax.plot(k_values, core_pcts, 'o-', color='green', linewidth=2.5, markersize=8,
           label='Core', alpha=0.8)
    ax.plot(k_values, uncertain_pcts, 's-', color='orange', linewidth=2.5, markersize=8,
           label='Uncertain', alpha=0.8)
    ax.plot(k_values, outlier_pcts, '^-', color='red', linewidth=2.5, markersize=8,
           label='Outlier', alpha=0.8)

    if best_k is not None and best_k in k_values:
        ax.axvline(best_k, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.text(best_k, ax.get_ylim()[1] * 0.95, f'best k={best_k}',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('k (number of clusters)', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Membership Category Percentages', fontsize=12, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # ========== RIGHT PANEL: Stacked area ==========
    ax = axes[1]
    ax.fill_between(k_values, 0, core_pcts, label='Core', color='green', alpha=0.6)
    ax.fill_between(k_values, core_pcts, np.array(core_pcts) + np.array(uncertain_pcts),
                   label='Uncertain', color='orange', alpha=0.6)
    ax.fill_between(k_values, np.array(core_pcts) + np.array(uncertain_pcts), 100,
                   label='Outlier', color='red', alpha=0.6)

    if best_k is not None and best_k in k_values:
        ax.axvline(best_k, color='black', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('k (number of clusters)', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Membership Category Composition', fontsize=12, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_membership_scatter(D, classification, title="Membership Visualization"):
    """2D projection colored by membership category."""
    pass

def plot_cluster_breakdown(cluster_stats, title="Per-Cluster Membership"):
    """Stacked bar chart showing membership by cluster."""
    pass