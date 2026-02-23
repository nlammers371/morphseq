"""
Dendrogram Visualization for Hierarchical Clustering

Provides functions for visualizing hierarchical clustering dendrograms with
optional categorical color bars (e.g., genotype, pair). Useful for K selection
and understanding cluster composition.

Functions
=========
- plot_dendrogram : Basic hierarchical clustering dendrogram
- plot_dendrogram_with_categories : Dendrogram with categorical color bars

Constants
=========
- PASTEL_COLORS : Pastel color palette for pair/category coloring

Created: 2025-12-19
Migrated from: results/mcolon/20251218_MD-DTW-morphseq_analysis/md_dtw_prototype.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path

from .styling import get_color_for_genotype


# Pastel color palette for pairs (softer, distinguishable colors)
PASTEL_COLORS = [
    '#FFB6C1',  # Light pink
    '#B0E0E6',  # Powder blue
    '#98FB98',  # Pale green
    '#FFE4B5',  # Moccasin
    '#DDA0DD',  # Plum
    '#F0E68C',  # Khaki
    '#AFEEEE',  # Pale turquoise
    '#FFA07A',  # Light salmon
    '#D8BFD8',  # Thistle
    '#F5DEB3',  # Wheat
    '#B0C4DE',  # Light steel blue
    '#FFDAB9',  # Peach puff
    '#E0BBE4',  # Mauve
    '#C1FFC1',  # Pale spring green
    '#FFE4E1',  # Misty rose
]


def generate_dendrograms(
    D: np.ndarray,
    embryo_ids: List[str],
    *,
    coassociation_matrix: Optional[np.ndarray] = None,
    linkage_method: str = 'average',
    k_highlight: Optional[List[int]] = None,
    color_threshold: Optional[float] = None,
    truncate_mode: Optional[str] = None,
    truncate_p: int = 30,
    orientation: str = 'top',
    figsize: Tuple[float, float] = (14, 8),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    verbose: bool = True,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Generate hierarchical clustering dendrogram with cluster assignments.

    Creates dendrogram visualization from distance matrix and computes cluster
    assignments for multiple K values. Helps select optimal K by visualizing
    clustering hierarchy with multiple K cutoff lines.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n_embryos, n_embryos) - output from compute_dtw_distance_matrix()
        or compute_md_dtw_distance_matrix()
    embryo_ids : List[str]
        List of embryo identifiers (same order as D rows)
    coassociation_matrix : np.ndarray, optional
        Co-association matrix from bootstrap consensus clustering (n_embryos, n_embryos).
        If provided, dendrogram will be built from consensus distances (1 - M) instead of D.
        Use compute_coassociation_matrix() to generate this matrix.
    linkage_method : str, default='average'
        Linkage method for hierarchical clustering.
        Options: 'average' (UPGMA), 'single', 'complete', 'ward' (ward requires euclidean)
        Default: 'average' (best for DTW distances)
    k_highlight : List[int], optional
        List of K values to show as horizontal cutoff lines.
        Example: [2, 3, 4] shows lines where dendrogram would be cut for each K.
    color_threshold : float, optional
        Height at which to color branches (clusters below this are same color).
        If None, uses scipy default (0.7 * max height).
    truncate_mode : str, optional
        Truncate dendrogram for large N. Options: 'lastp', 'level', None
    truncate_p : int, default=30
        Parameter for truncation (e.g., show last p clusters)
    orientation : str, default='top'
        Dendrogram orientation: 'top', 'bottom', 'left', 'right'
    figsize : Tuple[float, float], default=(14, 8)
        Figure size (width, height)
    title : str, optional
        Plot title. If None, auto-generated.
    save_path : str or Path, optional
        Path to save figure. If None, not saved.
    dpi : int, default=150
        Resolution for saved figure.
    verbose : bool, default=True
        Print diagnostic information.

    Returns
    -------
    fig : matplotlib.figure.Figure
        matplotlib Figure object
    info : Dict[str, Any]
        Dict with clustering results:
        - 'linkage_matrix': scipy linkage matrix Z
        - 'dendrogram_data': scipy dendrogram output dict
        - 'cluster_labels': Dict[k, np.ndarray] with cluster labels (array) for each k
        - 'clusters_by_k': Dict[k, Dict[cluster_id, List[embryo_ids]]] - embryos grouped by cluster
        - 'embryo_to_cluster': Dict[k, Dict[embryo_id, cluster_id]] - embryo ID to cluster mapping
        - 'k_cutoff_heights': Dict[k, float] with cutoff heights for each k

    Examples
    --------
    >>> D = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3)
    >>> fig, info = plot_dendrogram(
    ...     D, embryo_ids,
    ...     k_highlight=[2, 3, 4],
    ...     title='b9d2 Mutant Clustering'
    ... )
    >>> # Access cluster assignments for k=3
    >>> labels_k3 = info['cluster_assignments'][3]

    Notes
    -----
    - Uses condensed distance format internally (scipy requirement)
    - Average linkage (UPGMA) is recommended for DTW distances
    - Ward linkage requires Euclidean distances (not recommended for DTW)
    """
    n = len(D)

    if verbose:
        print(f"Generating dendrogram...")
        print(f"  Embryos: {n}")
        print(f"  Linkage method: {linkage_method}")
        if coassociation_matrix is not None:
            print(f"  Using consensus distances (1 - co-association)")

    # Choose distance matrix for linkage
    if coassociation_matrix is not None:
        # Use consensus distances: D = 1 - M
        from ..clustering.bootstrap_clustering import coassociation_to_distance
        D_for_linkage = coassociation_to_distance(coassociation_matrix)
    else:
        # Use provided distance matrix
        D_for_linkage = D

    # Convert square distance matrix to condensed form for scipy
    # Ensure matrix is exactly symmetric (handle floating point)
    D_sym = (D_for_linkage + D_for_linkage.T) / 2
    np.fill_diagonal(D_sym, 0)
    D_condensed = squareform(D_sym)

    # Compute linkage
    Z = linkage(D_condensed, method=linkage_method)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set color threshold
    if color_threshold is None:
        # Default: disable coloring (set above max height)
        color_threshold = Z[:, 2].max() * 1.1

    # Plot dendrogram
    dendro_kwargs = {
        'Z': Z,
        'labels': embryo_ids,
        'ax': ax,
        'orientation': orientation,
        'color_threshold': color_threshold,
        'above_threshold_color': 'gray',
        'leaf_rotation': 90 if orientation in ['top', 'bottom'] else 0,
        'leaf_font_size': max(6, min(10, 200 // n)),  # Scale font with N
    }

    if truncate_mode:
        dendro_kwargs['truncate_mode'] = truncate_mode
        dendro_kwargs['p'] = truncate_p

    dendro_data = dendrogram(**dendro_kwargs)

    # Prepare output info
    info = {
        'linkage_matrix': Z,
        'dendrogram_data': dendro_data,
        'cluster_labels': {},           # k -> np.ndarray of labels
        'cluster_assignments': {},      # DEPRECATED: kept for backward compatibility
        'clusters_by_k': {},            # NEW: k -> {cluster_id: [embryo_ids]}
        'embryo_to_cluster': {},        # NEW: k -> {embryo_id: cluster_id}
        'k_cutoff_heights': {},
    }

    # Add horizontal lines for k_highlight values
    if k_highlight:
        # Sort merge heights
        heights = Z[:, 2]

        # For k clusters, we need n-k merges, so cutoff is between merge n-k-1 and n-k
        for k in sorted(k_highlight):
            if k < 2 or k > n:
                if verbose:
                    print(f"  WARNING: k={k} out of range [2, {n}], skipping")
                continue

            # Height to cut at: midpoint between merge that creates k clusters and k-1 clusters
            # After n-k merges we have k clusters
            merge_idx_before = n - k - 1  # Last merge before having k clusters
            merge_idx_after = n - k       # First merge after having k clusters

            if merge_idx_before >= 0:
                h_before = heights[merge_idx_before]
            else:
                h_before = 0

            if merge_idx_after < len(heights):
                h_after = heights[merge_idx_after]
            else:
                h_after = heights[-1] * 1.1

            # Cut at midpoint
            cutoff_height = (h_before + h_after) / 2
            info['k_cutoff_heights'][k] = cutoff_height

            # Get cluster assignments at this k
            labels = fcluster(Z, k, criterion='maxclust')
            # Convert to 0-indexed
            labels = labels - 1

            # Store in both formats
            info['cluster_labels'][k] = labels
            info['cluster_assignments'][k] = labels  # DEPRECATED: backward compatibility

            # NEW: Create clusters_by_k structure (cluster_id -> [embryo_ids])
            cluster_dict = {}
            for cluster_id in range(k):
                mask = labels == cluster_id
                cluster_dict[cluster_id] = [embryo_ids[i] for i in range(len(embryo_ids)) if mask[i]]
            info['clusters_by_k'][k] = cluster_dict

            # NEW: Create embryo_to_cluster mapping (embryo_id -> cluster_id)
            info['embryo_to_cluster'][k] = dict(zip(embryo_ids, labels))

            # Draw horizontal line
            color = plt.cm.tab10(k % 10)
            ax.axhline(y=cutoff_height, color=color, linestyle='--', alpha=0.7,
                      label=f'k={k} (h={cutoff_height:.2f})')

            if verbose:
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"  k={k}: cutoff height={cutoff_height:.3f}, cluster sizes: {dict(zip(unique_labels, counts))}")

    # Formatting
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Hierarchical Clustering Dendrogram (n={n}, linkage={linkage_method})',
                    fontsize=14, fontweight='bold')

    if orientation in ['top', 'bottom']:
        ax.set_xlabel('Embryo ID', fontsize=11)
        ax.set_ylabel('Distance', fontsize=11)
    else:
        ax.set_xlabel('Distance', fontsize=11)
        ax.set_ylabel('Embryo ID', fontsize=11)

    if k_highlight:
        ax.legend(loc='upper right', fontsize=10)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"  Saved to: {save_path}")

    if verbose:
        print(f"✓ Dendrogram generated")

    return fig, info


def plot_dendrogram(
    D: np.ndarray,
    embryo_ids: List[str],
    *,
    linkage_method: str = 'average',
    k_highlight: Optional[List[int]] = None,
    color_threshold: Optional[float] = None,
    truncate_mode: Optional[str] = None,
    truncate_p: int = 30,
    orientation: str = 'top',
    figsize: Tuple[float, float] = (14, 8),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    verbose: bool = True,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    DEPRECATED: Use generate_dendrograms() instead.

    This function is kept for backward compatibility.
    All functionality has been moved to generate_dendrograms().
    """
    import warnings
    warnings.warn(
        "plot_dendrogram() is deprecated. Use generate_dendrograms() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return generate_dendrograms(
        D, embryo_ids,
        linkage_method=linkage_method,
        k_highlight=k_highlight,
        color_threshold=color_threshold,
        truncate_mode=truncate_mode,
        truncate_p=truncate_p,
        orientation=orientation,
        figsize=figsize,
        title=title,
        save_path=save_path,
        dpi=dpi,
        verbose=verbose
    )


def add_cluster_column(
    df: pd.DataFrame,
    dendro_info: Dict[str, Any],
    k: int,
    column_name: str = 'cluster',
    embryo_id_col: str = 'embryo_id',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Add cluster assignments from dendrogram to DataFrame.

    Simplifies adding cluster assignments to a DataFrame by using the
    embryo_to_cluster mapping from dendrogram results.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with embryo_id column
    dendro_info : Dict[str, Any]
        Output dict from generate_dendrograms() or plot_dendrogram()
    k : int
        K value (number of clusters) to use
    column_name : str, default='cluster'
        Name of new column to add
    embryo_id_col : str, default='embryo_id'
        Name of embryo ID column in df
    inplace : bool, default=False
        If True, modify df in place. If False, return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with cluster assignments added as new column

    Examples
    --------
    >>> fig, dendro_info = generate_dendrograms(D, embryo_ids, k_highlight=[3])
    >>> df_clustered = add_cluster_column(df, dendro_info, k=3, column_name='md_dtw_cluster')
    """
    if k not in dendro_info['embryo_to_cluster']:
        raise ValueError(f"k={k} not found in dendrogram results. Available k values: {list(dendro_info['embryo_to_cluster'].keys())}")

    if inplace:
        result = df
    else:
        result = df.copy()

    # Use embryo_to_cluster mapping
    cluster_map = dendro_info['embryo_to_cluster'][k]
    result[column_name] = result[embryo_id_col].map(cluster_map)

    return result


def plot_dendrogram_with_categories(
    D: np.ndarray,
    embryo_ids: List[str],
    category_df: pd.DataFrame,
    category_cols: List[str] = ['pair', 'genotype'],
    *,
    linkage_method: str = 'average',
    k_highlight: Optional[List[int]] = None,
    color_threshold: Optional[float] = None,
    truncate_mode: Optional[str] = None,
    truncate_p: int = 30,
    orientation: str = 'top',
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    spacer_height: float = .7,
    verbose: bool = True,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Plot hierarchical clustering dendrogram with categorical color bars.

    Extended version of plot_dendrogram() that adds colored bars below the dendrogram
    showing categorical groupings (e.g., pair, genotype). Useful for visualizing how
    clusters relate to experimental design or biological categories.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n_embryos, n_embryos) - output from compute_md_dtw_distance_matrix()
    embryo_ids : List[str]
        List of embryo identifiers (same order as D rows)
    category_df : pd.DataFrame
        DataFrame with 'embryo_id' column plus categorical columns.
        Example: ['embryo_id', 'pair', 'genotype']
    category_cols : List[str], default=['pair', 'genotype']
        List of category column names to show as color bars.
    linkage_method : str, default='average'
        Linkage method ('average', 'single', 'complete', 'ward')
    k_highlight : List[int], optional
        List of K values to show as horizontal cutoff lines
    color_threshold : float, optional
        Height at which to color branches (if None, uses scipy default)
    truncate_mode : str, optional
        Truncate dendrogram for large N ('lastp', 'level', None)
    truncate_p : int, default=30
        Parameter for truncation
    orientation : str, default='top'
        Dendrogram orientation ('top', 'bottom', 'left', 'right')
    figsize : Tuple[float, float], optional
        Figure size (width, height). Auto-calculated if None.
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save figure
    dpi : int, default=150
        Resolution for saved figure
    spacer_height : float, default=0.7
        Height (in inches) of white space between dendrogram and first category bar.
        Increase this value if leaf labels overlap with category bars.
    verbose : bool, default=True
        Print diagnostic information

    Returns
    -------
    fig : matplotlib.figure.Figure
        matplotlib Figure object
    info : Dict[str, Any]
        Dict with clustering results:
        - 'linkage_matrix': scipy linkage matrix Z
        - 'dendrogram_data': scipy dendrogram output dict
        - 'cluster_labels': Dict[k, np.ndarray] with cluster labels (array) for each k
        - 'cluster_assignments': Dict[k, np.ndarray] with cluster labels for each k (DEPRECATED)
        - 'embryo_to_cluster': Dict[k, Dict[embryo_id, cluster_id]] - embryo ID to cluster mapping
        - 'clusters_by_k': Dict[k, Dict[cluster_id, List[embryo_ids]]] - embryos grouped by cluster
        - 'k_cutoff_heights': Dict[k, float] with cutoff heights for each k
        - 'category_colors': Dict[category_col, Dict[value, color]] color mappings

    Examples
    --------
    >>> # Prepare category data
    >>> category_df = pd.DataFrame({
    ...     'embryo_id': embryo_ids,
    ...     'pair': ['pair1', 'pair1', 'pair2', ...],
    ...     'genotype': ['wt', 'mut', 'wt', ...]
    ... })
    >>>
    >>> # Plot with both pair and genotype bars
    >>> fig, info = plot_dendrogram_with_categories(
    ...     D, embryo_ids, category_df,
    ...     category_cols=['pair', 'genotype'],
    ...     k_highlight=[2, 3, 4]
    ... )

    Notes
    -----
    - Color bars are shown below dendrogram (for orientation='top')
    - Each category gets its own colored bar with legend
    - Bars are aligned with dendrogram leaves
    - Missing categories are filled with 'unknown'
    """
    n = len(D)

    if verbose:
        print(f"Generating dendrogram with category bars...")
        print(f"  Embryos: {n}")
        print(f"  Categories: {category_cols}")
        print(f"  Linkage method: {linkage_method}")

    # Step 1: Validate and prepare category data
    # Create lookup: embryo_id -> {category_col: value}
    category_lookup = {}
    for col in category_cols:
        if col not in category_df.columns:
            raise ValueError(f"Category column '{col}' not found in category_df. Available: {list(category_df.columns)}")

        # Create mapping for this category
        col_map = dict(zip(category_df['embryo_id'], category_df[col]))
        # Fill missing with 'unknown'
        col_map_filled = {eid: col_map.get(eid, 'unknown') for eid in embryo_ids}
        category_lookup[col] = col_map_filled

    if verbose:
        for col in category_cols:
            unique_vals = set(category_lookup[col].values())
            print(f"  '{col}': {len(unique_vals)} unique values: {sorted(unique_vals)}")

    # Step 2: Compute linkage (same as original)
    D_sym = (D + D.T) / 2
    np.fill_diagonal(D_sym, 0)
    D_condensed = squareform(D_sym)
    Z = linkage(D_condensed, method=linkage_method)

    # Step 3: Create figure with GridSpec
    n_category_bars = len(category_cols)
    bar_height_per_category = 0.4  # inches
    # spacer_height is now a parameter - white space between dendrogram and first bar
    total_bar_height = n_category_bars * bar_height_per_category + spacer_height
    extra_label_space = 1.5  # Extra height to keep rotated labels off the category bars

    if figsize is None:
        dendro_height = 8
        fig_width = 14
        fig_height = dendro_height + total_bar_height + 0.5 + extra_label_space
        figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize)
    # Use a spacer row between dendrogram and category bars
    gs = gridspec.GridSpec(
        nrows=2 + n_category_bars,  # dendro + spacer + category bars
        ncols=1,
        height_ratios=[8, spacer_height] + [bar_height_per_category] * n_category_bars,
        hspace=0.15  # More breathing room between dendrogram and category bars
    )

    ax_dendro = fig.add_subplot(gs[0, 0])
    # gs[1, 0] is the spacer - we don't create an axis for it
    ax_categories = [fig.add_subplot(gs[i+2, 0]) for i in range(n_category_bars)]

    # Step 4: Plot dendrogram (same as original)
    if color_threshold is None:
        # Default: disable coloring (set above max height)
        color_threshold = Z[:, 2].max() * 1.1

    dendro_kwargs = {
        'Z': Z,
        'labels': embryo_ids,
        'ax': ax_dendro,
        'orientation': orientation,
        'color_threshold': color_threshold,
        'above_threshold_color': 'gray',
        'leaf_rotation': 90 if orientation in ['top', 'bottom'] else 0,
        'leaf_font_size': max(6, min(10, 200 // n)),
    }

    if truncate_mode:
        dendro_kwargs['truncate_mode'] = truncate_mode
        dendro_kwargs['p'] = truncate_p

    dendro_data = dendrogram(**dendro_kwargs)

    # Get leaf order
    leaf_order = dendro_data['leaves']  # Indices into embryo_ids

    # Step 5: Prepare output info
    info = {
        'linkage_matrix': Z,
        'dendrogram_data': dendro_data,
        'cluster_labels': {},          # k -> np.ndarray of labels (NEW: for consistency with generate_dendrograms)
        'cluster_assignments': {},     # k -> np.ndarray of labels (DEPRECATED: kept for backward compatibility)
        'embryo_to_cluster': {},       # k -> {embryo_id: cluster_id} (NEW: for add_cluster_column)
        'clusters_by_k': {},           # k -> {cluster_id: [embryo_ids]} (NEW: for consistency)
        'k_cutoff_heights': {},
        'category_colors': {},
    }

    # Add k_highlight lines (same as original)
    if k_highlight:
        heights = Z[:, 2]
        for k in sorted(k_highlight):
            if k < 2 or k > n:
                if verbose:
                    print(f"  WARNING: k={k} out of range [2, {n}], skipping")
                continue

            merge_idx_before = n - k - 1
            merge_idx_after = n - k

            h_before = heights[merge_idx_before] if merge_idx_before >= 0 else 0
            h_after = heights[merge_idx_after] if merge_idx_after < len(heights) else heights[-1] * 1.1

            cutoff_height = (h_before + h_after) / 2
            info['k_cutoff_heights'][k] = cutoff_height

            labels = fcluster(Z, k, criterion='maxclust') - 1
            info['cluster_labels'][k] = labels
            info['cluster_assignments'][k] = labels  # DEPRECATED: backward compatibility

            # NEW: Add embryo_to_cluster and clusters_by_k for consistency with generate_dendrograms
            info['embryo_to_cluster'][k] = dict(zip(embryo_ids, labels))
            cluster_dict = {}
            for cluster_id in range(k):
                mask = labels == cluster_id
                cluster_dict[cluster_id] = [embryo_ids[i] for i in range(len(embryo_ids)) if mask[i]]
            info['clusters_by_k'][k] = cluster_dict

            color = plt.cm.tab10(k % 10)
            ax_dendro.axhline(y=cutoff_height, color=color, linestyle='--', alpha=0.7,
                             label=f'k={k} (h={cutoff_height:.2f})')

    # Step 6: Plot category color bars
    for i, cat_col in enumerate(category_cols):
        ax = ax_categories[i]

        # Get category values in dendrogram leaf order
        ordered_embryo_ids = [embryo_ids[idx] for idx in leaf_order]
        cat_values = [category_lookup[cat_col][eid] for eid in ordered_embryo_ids]

        # Map category values to colors using smart assignment
        unique_cats = sorted(set(cat_values))
        color_map = {}

        # Use appropriate color scheme based on category type
        if cat_col in ['genotype', 'Genotype', 'geno']:
            # Use standard genotype colors
            for cat in unique_cats:
                color_map[cat] = get_color_for_genotype(str(cat))
        elif cat_col in ['pair', 'Pair', 'cross', 'Cross']:
            # Use pastel colors for pairs
            for j, cat in enumerate(unique_cats):
                color_map[cat] = PASTEL_COLORS[j % len(PASTEL_COLORS)]
        else:
            # Use standard palette for other categories
            standard_palette = plt.cm.tab20.colors
            for j, cat in enumerate(unique_cats):
                color_map[cat] = standard_palette[j % len(standard_palette)]

        info['category_colors'][cat_col] = color_map

        # Create color array for imshow
        color_indices = [unique_cats.index(val) for val in cat_values]
        color_array = np.array([color_indices])

        # Plot as image
        cmap = plt.matplotlib.colors.ListedColormap([color_map[cat] for cat in unique_cats])
        ax.imshow(color_array, aspect='auto', interpolation='nearest', cmap=cmap,
                 vmin=0, vmax=len(unique_cats)-1)

        # Formatting
        ax.set_yticks([0])
        ax.set_yticklabels([cat_col], fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_xlim(-0.5, len(ordered_embryo_ids) - 0.5)  # Match dendrogram width

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add legend
        legend_elements = [Patch(facecolor=color_map[cat], edgecolor='black', label=cat)
                          for cat in unique_cats]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                 fontsize=9, frameon=False)

    # Step 7: Format dendrogram axis
    if title:
        ax_dendro.set_title(title, fontsize=14, fontweight='bold', pad=10)
    else:
        ax_dendro.set_title(f'Hierarchical Clustering Dendrogram with Categories (n={n}, linkage={linkage_method})',
                           fontsize=14, fontweight='bold', pad=10)

    if orientation in ['top', 'bottom']:
        ax_dendro.set_xlabel('', fontsize=11)  # Labels shown by category bars
        ax_dendro.set_ylabel('Distance', fontsize=11)
    else:
        ax_dendro.set_xlabel('Distance', fontsize=11)
        ax_dendro.set_ylabel('', fontsize=11)

    if k_highlight:
        ax_dendro.legend(loc='upper right', fontsize=10, framealpha=0.9)

    ax_dendro.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"  Saved to: {save_path}")

    if verbose:
        print(f"✓ Dendrogram with categories generated")

    return fig, info
