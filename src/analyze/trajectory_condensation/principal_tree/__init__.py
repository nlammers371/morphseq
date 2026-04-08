"""Principal tree fitting and visualization for condensed trajectories."""

from . import core, viz
from .core import (
    BranchTestResult,
    build_embryo_spacetime_cloud,
    fit_principal_tree,
    project_observations_to_tree,
    identify_branch_nodes,
    assign_embryos_to_arms,
    run_all_branch_tests,
    branch_results_to_df,
    extract_segments,
    prune_phantom_segments,
    segments_to_edges_df,
)
from .viz import (
    plot_tree_schematic,
    plot_branch_allocation_bars,
    plot_tree_3d,
    save_tree_3d_gif,
    plot_tree_metromap,
)

__all__ = [
    "core",
    "viz",
    "BranchTestResult",
    "build_embryo_spacetime_cloud",
    "fit_principal_tree",
    "project_observations_to_tree",
    "identify_branch_nodes",
    "assign_embryos_to_arms",
    "run_all_branch_tests",
    "branch_results_to_df",
    "extract_segments",
    "prune_phantom_segments",
    "segments_to_edges_df",
    "plot_tree_schematic",
    "plot_branch_allocation_bars",
    "plot_tree_3d",
    "save_tree_3d_gif",
    "plot_tree_metromap",
]
