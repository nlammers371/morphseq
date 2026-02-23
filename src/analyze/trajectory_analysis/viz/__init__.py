"""
Visualization subpackage for trajectory analysis.

Contains:
- dendrogram: Hierarchical clustering visualization
- styling: Genotype color and style utilities
- plotting: Plotting functions (core + proportions + 3D)
"""

# Dendrogram functions
from .dendrogram import (
    generate_dendrograms,
    plot_dendrogram,
    add_cluster_column,
    plot_dendrogram_with_categories,
    PASTEL_COLORS,
)

# Styling utilities
from .styling import (
    extract_genotype_suffix,
    extract_genotype_prefix,
    get_color_for_genotype,
    get_membership_category_colors,
    sort_genotypes_by_suffix,
    build_genotype_style_config,
    format_genotype_label,
)

# Re-export plotting subpackage (users can do: from trajectory_analysis.viz import plot_cluster_trajectories_df)
from .plotting import (
    # Core plotting
    plot_cluster_trajectories_df,
    plot_membership_trajectories_df,
    plot_posterior_heatmap,
    plot_2d_scatter,
    plot_membership_vs_k,
    plot_cluster_trajectories,  # deprecated
    plot_membership_trajectories,  # deprecated
    # Faceted plotting
    plot_proportions,
    # Flow plotting
    plot_cluster_flow,
    # 3D plotting
    plot_3d_scatter,
)

__all__ = [
    # Dendrogram
    'generate_dendrograms',
    'plot_dendrogram',
    'add_cluster_column',
    'plot_dendrogram_with_categories',
    'PASTEL_COLORS',
    # Styling
    'extract_genotype_suffix',
    'extract_genotype_prefix',
    'get_color_for_genotype',
    'get_membership_category_colors',
    'sort_genotypes_by_suffix',
    'build_genotype_style_config',
    'format_genotype_label',
    # Core plotting
    'plot_cluster_trajectories_df',
    'plot_membership_trajectories_df',
    'plot_posterior_heatmap',
    'plot_2d_scatter',
    'plot_membership_vs_k',
    'plot_cluster_trajectories',  # deprecated
    'plot_membership_trajectories',  # deprecated
    # Faceted plotting
    'plot_proportions',
    # Flow plotting
    'plot_cluster_flow',
    # 3D plotting
    'plot_3d_scatter',
]
