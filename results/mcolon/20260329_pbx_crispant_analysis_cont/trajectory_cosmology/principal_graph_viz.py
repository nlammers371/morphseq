"""
principal_graph_viz.py
----------------------
2D schematic renderer for the phenotypic principal graph.

Layout
------
Nodes are placed at their condensed (x, y) positions from the MST layout.
Edges are drawn as straight lines between node centroids.
Branch points are annotated with permutation p-values and Cramér's V.

Embryo scatter is optionally overlaid behind the graph skeleton.

Public API
----------
  plot_principal_graph(nodes_df, edges, branch_results, positions, mask,
                       labels, time_values, color_map, ...) -> (fig, ax)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .principal_graph import BranchTestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_embryo_counts(
    assignments_df: pd.DataFrame,
    edges: list[tuple[int, int]],
) -> dict[int, int]:
    """Total embryo count per edge index."""
    if assignments_df is None or len(assignments_df) == 0:
        return {i: 0 for i in range(len(edges))}
    return assignments_df.groupby("edge_idx").size().to_dict()


def _pval_label(pval: float) -> str:
    if pval < 0.001:
        return "p<0.001"
    elif pval < 0.01:
        return f"p={pval:.3f}"
    else:
        return f"p={pval:.2f}"


def _sig_star(pval: float) -> str:
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_principal_graph(
    nodes_df: pd.DataFrame,
    edges: list[tuple[int, int]],
    branch_results: list[BranchTestResult],
    positions: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    time_values: np.ndarray | None = None,
    assignments_df: pd.DataFrame | None = None,
    color_map: dict[str, str] | None = None,
    figsize: tuple[float, float] = (10, 8),
    node_size: float = 200.0,
    max_edge_lw: float = 6.0,
    min_edge_lw: float = 1.0,
    scatter_alpha: float = 0.25,
    scatter_size: float = 12.0,
    show_scatter: bool = True,
    t_idx: int | None = None,
    title: str = "Phenotypic Principal Graph",
    alpha_threshold: float = 0.05,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw the phenotypic principal graph schematic.

    Parameters
    ----------
    nodes_df : DataFrame — from build_centroid_mst (node_id, condition, x, y, n_embryos)
    edges : list of (node_id_a, node_id_b)
    branch_results : list of BranchTestResult — one per branch node
    positions : (N_e, T, 2) condensed positions — for embryo scatter
    mask : (N_e, T) bool
    labels : (N_e,) genotype strings
    time_values : (T,)
    assignments_df : embryo edge-assignment table (for edge thickness)
    color_map : genotype → hex color
    figsize : figure size in inches
    node_size : scatter marker size for condition centroids
    max_edge_lw : maximum edge line width (mapped to most-populated edge)
    min_edge_lw : minimum edge line width
    scatter_alpha : alpha for individual embryo scatter
    scatter_size : marker size for embryo scatter
    show_scatter : whether to draw individual embryo positions
    t_idx : time bin index to use for embryo scatter; defaults to layout time
    title : axes title
    alpha_threshold : p-value below which annotations are emphasised

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)

    coords = nodes_df[["x", "y"]].values   # (n_nodes, 2)
    default_color = "#999999"

    # ------------------------------------------------------------------
    # 1. Optional individual embryo scatter
    # ------------------------------------------------------------------
    if show_scatter and positions is not None and mask is not None and labels is not None:
        # Determine time bin
        layout_t = t_idx
        if layout_t is None:
            layout_t = int(mask.sum(axis=0).argmax())
        for emb_idx in range(positions.shape[0]):
            if not mask[emb_idx, layout_t]:
                continue
            geno = str(labels[emb_idx])
            color = color_map.get(geno, default_color) if color_map else default_color
            pos = positions[emb_idx, layout_t, :]
            ax.scatter(pos[0], pos[1],
                       c=color, s=scatter_size,
                       alpha=scatter_alpha, linewidths=0, zorder=1)

    # ------------------------------------------------------------------
    # 2. Edge embryo counts (for thickness scaling)
    # ------------------------------------------------------------------
    edge_counts = _edge_embryo_counts(assignments_df, edges)
    max_count = max(edge_counts.values()) if edge_counts else 1
    max_count = max(max_count, 1)

    # ------------------------------------------------------------------
    # 3. Draw edges
    # ------------------------------------------------------------------
    for eidx, (i, j) in enumerate(edges):
        count = edge_counts.get(eidx, 0)
        lw = min_edge_lw + (max_edge_lw - min_edge_lw) * (count / max_count)
        x_vals = [coords[i, 0], coords[j, 0]]
        y_vals = [coords[i, 1], coords[j, 1]]
        ax.plot(x_vals, y_vals,
                color="#555555", linewidth=lw,
                solid_capstyle="round", zorder=2)

    # ------------------------------------------------------------------
    # 4. Draw condition centroid nodes
    # ------------------------------------------------------------------
    for _, row in nodes_df.iterrows():
        cond = row["condition"]
        color = color_map.get(cond, default_color) if color_map else default_color
        ax.scatter(row["x"], row["y"],
                   c=color, s=node_size,
                   zorder=5, edgecolors="white", linewidths=1.5)

    # Condition labels (offset slightly to avoid overlap)
    for _, row in nodes_df.iterrows():
        cond = row["condition"]
        short = cond.replace("_crispant", "").replace("_", " ")
        ax.annotate(
            short,
            xy=(row["x"], row["y"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8, fontweight="bold",
            color=color_map.get(cond, default_color) if color_map else default_color,
            zorder=6,
        )

    # ------------------------------------------------------------------
    # 5. Annotate branch points
    # ------------------------------------------------------------------
    branch_result_map = {r.node_id: r for r in branch_results}

    for bn, res in branch_result_map.items():
        bx, by = coords[bn, 0], coords[bn, 1]
        is_sig = res.pval < alpha_threshold
        marker_color = "#D32F2F" if is_sig else "#616161"
        marker_size = 180 if is_sig else 120

        # Draw a diamond marker at the branch point
        ax.scatter(bx, by, marker="D",
                   c=marker_color, s=marker_size,
                   zorder=7, edgecolors="white", linewidths=1.0)

        # P-value annotation
        pval_text = _pval_label(res.pval)
        star = _sig_star(res.pval)
        v_text = f"V={res.effect_size:.2f}"
        label = f"{star}  {pval_text}\n{v_text}  (n={res.n_embryos_tested})"

        ax.annotate(
            label,
            xy=(bx, by),
            xytext=(10, -18),
            textcoords="offset points",
            fontsize=7.5,
            color=marker_color,
            fontweight="bold" if is_sig else "normal",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=marker_color,
                alpha=0.85,
                linewidth=0.8,
            ),
            zorder=8,
        )

    # ------------------------------------------------------------------
    # 6. Legend
    # ------------------------------------------------------------------
    legend_handles = []
    if color_map:
        for cond, color in color_map.items():
            if cond in nodes_df["condition"].values:
                short = cond.replace("_crispant", "").replace("_", " ")
                legend_handles.append(
                    mpatches.Patch(color=color, label=short)
                )

    # Add branch point legend entries
    legend_handles += [
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="#D32F2F", markersize=8,
               label="Branch (p < 0.05)"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="#616161", markersize=7,
               label="Branch (ns)"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # ------------------------------------------------------------------
    # 7. Styling
    # ------------------------------------------------------------------
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Condensed x", fontsize=10)
    ax.set_ylabel("Condensed y", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Per-branch stacked bar chart
# ---------------------------------------------------------------------------

def plot_branch_allocation_bars(
    branch_results: list[BranchTestResult],
    color_map: dict[str, str] | None = None,
    figsize_per_branch: tuple[float, float] = (4.5, 3.5),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Stacked bar chart of genotype allocation per edge at each branch node.

    One subplot per branch node, bars = outgoing edges, colors = genotypes.

    Returns
    -------
    fig, axes
    """
    n = len(branch_results)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize_per_branch)
        ax.text(0.5, 0.5, "No branch points found", ha="center", va="center",
                transform=ax.transAxes)
        return fig, [ax]

    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_branch[0] * n, figsize_per_branch[1]),
        squeeze=False,
    )
    axes = axes[0]

    default_color = "#999999"

    for ax, res in zip(axes, branch_results):
        counts = res.counts_gk   # (genotypes × edge_k)
        genotypes = counts.index.tolist()
        edge_cols = counts.columns.tolist()

        x = np.arange(len(edge_cols))
        bottom = np.zeros(len(edge_cols))

        for geno in genotypes:
            vals = counts.loc[geno].values.astype(float)
            # Normalise to fraction
            totals = counts.sum(axis=0).values.astype(float)
            totals = np.where(totals == 0, 1, totals)
            fracs = vals / totals
            color = color_map.get(geno, default_color) if color_map else default_color
            ax.bar(x, fracs, bottom=bottom,
                   color=color, label=geno.replace("_crispant", "").replace("_", " "),
                   edgecolor="white", linewidth=0.5)
            bottom += fracs

        sig_label = _pval_label(res.pval)
        star = _sig_star(res.pval)
        ax.set_title(f"Branch node {res.node_id}\n{star} {sig_label}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.replace("edge_", "edge ") for c in edge_cols],
            fontsize=8, rotation=30, ha="right"
        )
        ax.set_ylabel("Fraction of embryos", fontsize=8)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.tight_layout()
    return fig, list(axes)
