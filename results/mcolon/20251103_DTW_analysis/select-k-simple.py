# 2_select_k.py
"""Functions for selecting optimal number of clusters."""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple

# ============ CORE FUNCTIONS ============

def compute_elbow(D: np.ndarray, labels_dict: Dict) -> Dict:
    """Compute within-cluster distances for elbow method."""
    wcss = {}
    for k, data in labels_dict.items():
        labels = data['labels']
        wc_dist = 0
        for cluster in np.unique(labels):
            mask = labels == cluster
            cluster_D = D[np.ix_(mask, mask)]
            wc_dist += cluster_D.sum() / 2  # Sum of pairwise distances
        wcss[k] = wc_dist
    return wcss


def consensus_clustering(C: np.ndarray, k: int) -> np.ndarray:
    """Cluster the co-association matrix to find consensus."""
    # Convert to distance and cluster
    dist = 1 - C
    condensed = dist[np.triu_indices(len(dist), k=1)]
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, k, criterion='maxclust') - 1
    return labels


def eigengap_analysis(D: np.ndarray, sigma: float = None) -> np.ndarray:
    """Compute eigengaps from affinity matrix."""
    if sigma is None:
        sigma = np.median(D[D > 0])
    
    # Create affinity matrix
    A = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(A, 0)
    
    # Normalized Laplacian
    D_diag = np.sum(A, axis=1)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(D_diag + 1e-10))
    L = np.eye(len(A)) - D_sqrt_inv @ A @ D_sqrt_inv
    
    # Eigenvalues
    eigenvals = np.linalg.eigvalsh(L)
    eigengaps = np.diff(eigenvals)
    
    return eigenvals, eigengaps


def gap_statistic_simple(D: np.ndarray, labels: np.ndarray, n_refs: int = 10) -> float:
    """Simplified gap statistic."""
    # Observed within-cluster sum
    obs_wss = 0
    for k in np.unique(labels):
        mask = labels == k
        cluster_D = D[np.ix_(mask, mask)]
        obs_wss += cluster_D.sum() / (2 * mask.sum())
    
    # Reference distribution (uniform random)
    ref_wss = []
    n = len(D)
    for _ in range(n_refs):
        rand_labels = np.random.randint(0, len(np.unique(labels)), n)
        ref_w = 0
        for k in np.unique(rand_labels):
            mask = rand_labels == k
            cluster_D = D[np.ix_(mask, mask)]
            ref_w += cluster_D.sum() / (2 * mask.sum())
        ref_wss.append(ref_w)
    
    gap = np.mean(ref_wss) - obs_wss
    gap_std = np.std(ref_wss)
    return gap, gap_std


# ============ WRAPPER FUNCTIONS ============

def evaluate_all_k(D: np.ndarray, baseline_results: Dict,
                   bootstrap_results: Dict = None, verbose: bool = False) -> Dict:
    """Evaluate all k values with multiple metrics.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    baseline_results : Dict
        Results from clustering for each k
    bootstrap_results : Dict
        Results from bootstrap for each k
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Metrics for each k value
    """
    from sklearn.metrics import adjusted_rand_score

    metrics = {}

    for k, data in baseline_results.items():
        labels = data['labels']

        if verbose:
            print(f"    Evaluating metrics for k={k}...")

        metrics[k] = {
            'silhouette': silhouette_score(D, labels, metric='precomputed'),
        }

        # WCSS (elbow)
        wc_dist = 0
        for cluster in np.unique(labels):
            mask = labels == cluster
            cluster_D = D[np.ix_(mask, mask)]
            wc_dist += cluster_D.sum() / 2
        metrics[k]['wcss'] = wc_dist

        # Gap statistic
        gap, gap_std = gap_statistic_simple(D, labels)
        metrics[k]['gap_statistic'] = (gap, gap_std)

        # Eigengap analysis
        eigenvals, eigengaps = eigengap_analysis(D)
        if k < len(eigengaps):
            metrics[k]['eigengap'] = eigengaps[k-1]
        else:
            metrics[k]['eigengap'] = np.nan

        # If we have bootstrap results, compute consensus
        if bootstrap_results and k in bootstrap_results:
            C = bootstrap_results[k]['coassoc']
            try:
                consensus_labels = consensus_clustering(C, k)
                # How well do consensus labels match original?
                metrics[k]['consensus_ari'] = adjusted_rand_score(labels, consensus_labels)
                metrics[k]['consensus_k'] = len(np.unique(consensus_labels))
            except Exception as e:
                metrics[k]['consensus_ari'] = np.nan
                metrics[k]['consensus_k'] = np.nan

            # Bootstrap stability
            metrics[k]['mean_ari'] = bootstrap_results[k].get('mean_ari', np.nan)
            metrics[k]['mean_silhouette_boot'] = bootstrap_results[k].get('mean_silhouette', np.nan)

    return metrics


def suggest_k(metrics: Dict, prior_k: int = 3, verbose: bool = False) -> int:
    """Suggest best k using multiple heuristics.

    Parameters
    ----------
    metrics : Dict
        Metrics for each k from evaluate_all_k
    prior_k : int
        Prior expected k (gets bonus)
    verbose : bool
        Print reasoning

    Returns
    -------
    int
        Suggested k value
    """
    scores = {}

    for k, m in metrics.items():
        score = 0

        # Higher silhouette is better (0 to 1, normalize to -5 to 5)
        if 'silhouette' in m:
            score += 10 * m['silhouette']

        # Higher gap is better (gap statistic)
        if 'gap_statistic' in m:
            gap, gap_std = m['gap_statistic']
            score += gap / (gap_std + 1e-6)

        # Higher eigengap is better
        if 'eigengap' in m and not np.isnan(m['eigengap']):
            score += m['eigengap']

        # Higher bootstrap ARI is better
        if 'mean_ari' in m and not np.isnan(m['mean_ari']):
            score += 5 * m['mean_ari']

        # Small penalty for complexity (prefer fewer clusters)
        score -= 0.1 * k

        # Bonus for prior
        if k == prior_k:
            score += 1.0

        scores[k] = score

        if verbose:
            print(f"      k={k}: score={score:.3f}")

    best_k = max(scores, key=scores.get)

    if verbose:
        print(f"\n    Selected k={best_k} (score={scores[best_k]:.3f})")

    return best_k


# ============ PLOTTING FUNCTIONS ============

def plot_metric_comparison(metrics, best_k=None, title="K-Selection Metrics Comparison", dpi=200):
    """
    Plot all k-selection metrics for comparison.

    Parameters
    ----------
    metrics : dict
        Metrics from evaluate_all_k (k -> metric_dict)
    best_k : int, optional
        Best k value (will be highlighted)
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)

    k_values = sorted(metrics.keys())

    # ========== Panel 1: Silhouette Scores ==========
    ax = axes[0, 0]
    silhouette_scores = [metrics[k].get('silhouette', np.nan) for k in k_values]
    ax.plot(k_values, silhouette_scores, 'o-', linewidth=2.5, markersize=8, color='green')
    if best_k is not None:
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Silhouette Coefficient', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    # ========== Panel 2: Gap Statistic ==========
    ax = axes[0, 1]
    gap_scores = []
    gap_errs = []
    for k in k_values:
        gap, gap_std = metrics[k].get('gap_statistic', (np.nan, np.nan))
        gap_scores.append(gap)
        gap_errs.append(gap_std)
    ax.errorbar(k_values, gap_scores, yerr=gap_errs, fmt='o-', linewidth=2.5,
                markersize=8, color='purple', capsize=5, capthick=2)
    if best_k is not None:
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Gap Statistic', fontsize=11)
    ax.set_title('Gap Statistic', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    # ========== Panel 3: Eigengap ==========
    ax = axes[1, 0]
    eigengaps = [metrics[k].get('eigengap', np.nan) for k in k_values]
    ax.plot(k_values, eigengaps, 's-', linewidth=2.5, markersize=8, color='orange')
    if best_k is not None:
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Eigengap Value', fontsize=11)
    ax.set_title('Spectral Eigengap', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    # ========== Panel 4: Bootstrap Stability (ARI) ==========
    ax = axes[1, 1]
    ari_scores = [metrics[k].get('mean_ari', np.nan) for k in k_values]
    ax.plot(k_values, ari_scores, '^-', linewidth=2.5, markersize=8, color='brown')
    if best_k is not None:
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Mean ARI', fontsize=11)
    ax.set_title('Bootstrap Stability (Adjusted Rand Index)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    ax.set_ylim([-0.1, 1.05])

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    # Add best k annotation if provided
    if best_k is not None:
        fig.text(0.5, 0.02, f'Recommended: k={best_k}', ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig


def plot_elbow(wcss_dict, title="Elbow Plot"):
    """Plot elbow curve for WCSS."""
    pass

def plot_silhouettes(metrics, title="Silhouette Scores"):
    """Bar plot of silhouette scores by k."""
    pass

def plot_eigengaps(eigenvals, eigengaps, title="Eigengap Analysis"):
    """Plot eigenvalues and gaps."""
    pass

def plot_consensus_blocks(C, labels, title="Consensus Matrix"):
    """Plot reordered co-association matrix showing block structure."""
    pass