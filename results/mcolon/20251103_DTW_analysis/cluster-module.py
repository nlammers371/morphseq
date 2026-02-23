# 1_cluster.py
"""Core clustering and bootstrap stability functions."""

import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score, silhouette_score
from typing import Dict, List, Tuple

# ============ CORE FUNCTIONS ============

def cluster_kmedoids(D: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-medoids clustering."""
    km = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = km.fit_predict(D)
    return labels, km.medoid_indices_


def bootstrap_once(D: np.ndarray, k: int, frac: float = 0.8) -> Dict:
    """Single bootstrap: sample, cluster, return results."""
    n = len(D)
    idx = np.random.choice(n, int(n * frac), replace=False)
    
    # Submatrix and cluster
    D_sub = D[np.ix_(idx, idx)]
    labels, medoids = cluster_kmedoids(D_sub, k)
    
    # Map back to full indices
    full_labels = np.full(n, -1)
    full_labels[idx] = labels
    
    return {
        'labels': full_labels,
        'indices': idx,
        'silhouette': silhouette_score(D_sub, labels, metric='precomputed')
    }


def compute_coassoc(bootstrap_results: List[Dict]) -> np.ndarray:
    """Compute co-association matrix from bootstrap results."""
    n = len(bootstrap_results[0]['labels'])
    C = np.zeros((n, n))
    counts = np.zeros((n, n))
    
    for res in bootstrap_results:
        labels = res['labels']
        idx = res['indices']
        
        # Count co-occurrences
        for i in idx:
            for j in idx:
                if i < j:
                    counts[i,j] += 1
                    counts[j,i] += 1
                    if labels[i] == labels[j]:
                        C[i,j] += 1
                        C[j,i] += 1
    
    # Normalize
    C = np.divide(C, counts, where=(counts > 0))
    np.fill_diagonal(C, 1.0)
    return C


# ============ WRAPPER FUNCTIONS ============

def run_baseline(D: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run baseline clustering for a single k value.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    k : int
        Number of clusters

    Returns
    -------
    labels : np.ndarray
        Cluster labels
    medoids : np.ndarray
        Indices of medoid points
    """
    labels, medoids = cluster_kmedoids(D, k)
    return labels, medoids


def run_bootstrap(D: np.ndarray, k: int, n_bootstrap: int = 100,
                   frac: float = 0.8, verbose: bool = False) -> Dict:
    """Run full bootstrap stability analysis.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix
    k : int
        Number of clusters
    n_bootstrap : int
        Number of bootstrap iterations
    frac : float
        Fraction of data to sample in each iteration
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Bootstrap results including co-association matrix and stability scores
    """
    # Reference labels
    ref_labels, ref_medoids = cluster_kmedoids(D, k)

    # Bootstrap
    boot_results = []
    ari_scores = []
    silhouette_scores = []

    for i in range(n_bootstrap):
        res = bootstrap_once(D, k, frac=frac)
        boot_results.append(res)

        # ARI for sampled points
        idx = res['indices']
        if len(idx) > 0:
            ari = adjusted_rand_score(ref_labels[idx], res['labels'][idx])
            ari_scores.append(ari)

        silhouette_scores.append(res['silhouette'])

        if verbose and (i + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"    Bootstrap {i+1}/{n_bootstrap} complete")

    # Co-association matrix
    C = compute_coassoc(boot_results)

    return {
        'reference_labels': ref_labels,
        'reference_medoids': ref_medoids,
        'coassoc': C,
        'ari_scores': np.array(ari_scores),
        'silhouette_scores': np.array(silhouette_scores),
        'mean_ari': np.mean(ari_scores) if ari_scores else np.nan,
        'mean_silhouette': np.mean(silhouette_scores) if silhouette_scores else np.nan,
        'bootstrap_results': boot_results
    }


# ============ PLOTTING FUNCTIONS ============

def plot_coassoc_matrix(C, labels=None, k=None, title=None, dpi=100):
    """
    Plot co-association matrix as heatmap.

    Parameters
    ----------
    C : np.ndarray
        Co-association matrix (symmetric)
    labels : np.ndarray, optional
        Cluster labels (if provided, sort by cluster for block visualization)
    k : int, optional
        Number of clusters (for title)
    title : str, optional
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    if title is None:
        if k is not None:
            title = f"Co-association Matrix (k={k})"
        else:
            title = "Co-association Matrix"

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    # Sort by cluster if labels provided (creates block structure)
    if labels is not None:
        sort_idx = np.argsort(labels)
        C_sorted = C[np.ix_(sort_idx, sort_idx)]
        im = ax.imshow(C_sorted, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    else:
        im = ax.imshow(C, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)

    ax.set_xlabel('Embryo', fontsize=11)
    ax.set_ylabel('Embryo', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, label='Co-association Frequency')

    plt.tight_layout()
    return fig


def plot_clustering(D, labels, title="Clustering Result"):
    """Plot clustering result (implement with MDS/PCA projection)."""
    pass

def plot_stability_scores(ari_scores, title="Bootstrap Stability (ARI)"):
    """Plot distribution of ARI scores."""
    pass