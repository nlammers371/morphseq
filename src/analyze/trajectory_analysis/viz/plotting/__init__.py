"""
Plotting subpackage for trajectory analysis visualization.

Contains:
- core: Main plotting functions (cluster trajectories, membership, heatmaps, scatter)
- proportions: Faceted proportion plots (generic implementation)

NOTE: 3D plotting has been moved to src.analyze.viz.plotting.plotting_3d (generic location).
The import is re-exported here for backward compatibility.
"""

# Core plotting functions
from .core import (
    # DataFrame-first API (recommended)
    plot_cluster_trajectories_df,
    plot_membership_trajectories_df,
    plot_posterior_heatmap,
    plot_2d_scatter,
    plot_membership_vs_k,
    # Legacy API (deprecated but kept for backward compatibility)
    plot_cluster_trajectories,
    plot_membership_trajectories,
)

# Flow plotting
from .flow import plot_cluster_flow

from analyze.viz.plotting.proportions import plot_proportions

# 3D plotting - re-exported from generic location for backward compatibility
from analyze.viz.plotting import plot_3d_scatter

__all__ = [
    # Core plotting
    'plot_cluster_trajectories_df',
    'plot_membership_trajectories_df',
    'plot_posterior_heatmap',
    'plot_2d_scatter',
    'plot_membership_vs_k',
    'plot_cluster_trajectories',  # deprecated
    'plot_membership_trajectories',  # deprecated
    # Faceted plotting (main functions)
    'plot_proportions',
    # Flow plotting
    'plot_cluster_flow',
    # 3D plotting (re-exported from viz.plotting)
    'plot_3d_scatter',
]
