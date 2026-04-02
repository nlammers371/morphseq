"""
principal_graph_viz.py  (v2)
-----------------------------
2D schematic renderers for the space-time principal graph.

Two schematics:
  (t_hpf, y)  — time on x-axis, condensed-y on y-axis  [primary]
  (t_hpf, x)  — time on x-axis, condensed-x on y-axis  [diagnostic]

Both show the skeleton graph overlaid on the full embryo-timepoint scatter.
Branch nodes are annotated with permutation p-values and Cramér's V.

Public API
----------
  plot_spacetime_schematic(skel_nodes_df, skel_edges, branch_results,
                            positions, mask, labels, time_values,
                            color_map, spatial_axis, ...) -> (fig, ax)

  plot_branch_allocation_bars(branch_results, color_map, ...) -> (fig, axes)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .principal_graph import BranchTestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pval_label(pval: float) -> str:
    if pval < 0.001:
        return "p<0.001"
    elif pval < 0.01:
        return f"p={pval:.3f}"
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
# Main schematic
# ---------------------------------------------------------------------------

def plot_spacetime_schematic(
    skel_nodes_df: pd.DataFrame,
    skel_edges: list[tuple[int, int]],
    branch_results: list[BranchTestResult],
    positions: np.ndarray,          # (N_e, T, 2)
    mask: np.ndarray,               # (N_e, T)
    labels: np.ndarray,             # (N_e,)
    time_values: np.ndarray,        # (T,)
    color_map: dict[str, str] | None = None,
    spatial_axis: str = "y",        # "y" or "x"
    figsize: tuple[float, float] = (14, 7),
    scatter_alpha: float = 0.12,
    scatter_size: float = 8.0,
    max_edge_lw: float = 5.0,
    min_edge_lw: float = 1.0,
    node_size: float = 80.0,
    alpha_threshold: float = 0.05,
    annotate_ns: bool = False,      # if False, only label p < alpha_threshold nodes
    min_n_embryos: int = 0,         # skip branch annotation if fewer embryos than this
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Render the space-time principal graph schematic.

    Parameters
    ----------
    skel_nodes_df : from contract_mst_skeleton
    skel_edges : skeleton edge list
    branch_results : from run_all_branch_tests
    positions : (N_e, T, 2)
    mask, labels, time_values
    color_map : genotype → hex color
    spatial_axis : "y" uses condensed-y on y-axis; "x" uses condensed-x
    """
    if spatial_axis not in ("x", "y"):
        raise ValueError("spatial_axis must be 'x' or 'y'")

    sp_col = "y_mean" if spatial_axis == "y" else "x_mean"
    sp_dim = 1 if spatial_axis == "y" else 0
    sp_label = "Condensed y" if spatial_axis == "y" else "Condensed x"

    fig, ax = plt.subplots(figsize=figsize)
    default_color = "#aaaaaa"

    # ------------------------------------------------------------------
    # 1. Background scatter — all embryo-timepoint observations
    # ------------------------------------------------------------------
    for emb_i in range(positions.shape[0]):
        geno = str(labels[emb_i])
        color = color_map.get(geno, default_color) if color_map else default_color
        for t in range(positions.shape[1]):
            if not mask[emb_i, t]:
                continue
            ax.scatter(
                time_values[t],
                positions[emb_i, t, sp_dim],
                c=color, s=scatter_size,
                alpha=scatter_alpha,
                linewidths=0, zorder=1,
            )

    # ------------------------------------------------------------------
    # 2. Skeleton edges
    # ------------------------------------------------------------------
    # Edge thickness: proportional to n_obs on the thicker of two nodes
    node_n_obs = skel_nodes_df["n_obs"].values
    max_n = max(node_n_obs.max(), 1)

    for (i, j) in skel_edges:
        n_ij = max(node_n_obs[i], node_n_obs[j])
        lw = min_edge_lw + (max_edge_lw - min_edge_lw) * (n_ij / max_n)
        xi, xj = skel_nodes_df.loc[i, "t_hpf_mean"], skel_nodes_df.loc[j, "t_hpf_mean"]
        yi, yj = skel_nodes_df.loc[i, sp_col], skel_nodes_df.loc[j, sp_col]
        ax.plot([xi, xj], [yi, yj],
                color="#444444", linewidth=lw,
                solid_capstyle="round", zorder=2)

    # ------------------------------------------------------------------
    # 3. Skeleton nodes (non-branch)
    # ------------------------------------------------------------------
    for _, row in skel_nodes_df.iterrows():
        is_branch = row.get("is_branch", False)
        if is_branch:
            continue
        ax.scatter(
            row["t_hpf_mean"], row[sp_col],
            c="#666666", s=node_size * 0.5,
            zorder=4, edgecolors="white", linewidths=0.8,
        )

    # ------------------------------------------------------------------
    # 4. Branch nodes annotated
    # ------------------------------------------------------------------
    br_map = {r.node_id: r for r in branch_results}

    for bn, res in br_map.items():
        bx = skel_nodes_df.loc[bn, "t_hpf_mean"]
        by = skel_nodes_df.loc[bn, sp_col]

        is_sig = res.pval < alpha_threshold
        skip_annotation = (not is_sig and not annotate_ns) or \
                          (res.n_embryos < min_n_embryos)

        marker_color = "#C62828" if is_sig else "#757575"
        ms = 180 if is_sig else (80 if not skip_annotation else 50)

        ax.scatter(bx, by, marker="D",
                   c=marker_color, s=ms,
                   zorder=6, edgecolors="white", linewidths=1.0,
                   alpha=1.0 if not skip_annotation else 0.4)

        if skip_annotation:
            continue

        star = _sig_star(res.pval)
        pval_text = _pval_label(res.pval)
        label = f"{star}  {pval_text}\nV={res.effect_size:.2f}  n={res.n_embryos}"

        ax.annotate(
            label,
            xy=(bx, by),
            xytext=(8, -22),
            textcoords="offset points",
            fontsize=7.5,
            color=marker_color,
            fontweight="bold" if is_sig else "normal",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=marker_color,
                alpha=0.88,
                linewidth=0.8,
            ),
            zorder=7,
        )

    # ------------------------------------------------------------------
    # 5. Legend
    # ------------------------------------------------------------------
    handles = []
    if color_map:
        for geno, color in color_map.items():
            short = geno.replace("_crispant", "").replace("_", " ")
            handles.append(mpatches.Patch(color=color, label=short))

    handles += [
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="#C62828", markersize=8,
               label="Branch (p < 0.05)"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="#757575", markersize=7,
               label="Branch (ns)"),
    ]
    ax.legend(handles=handles, loc="upper left",
              fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    # ------------------------------------------------------------------
    # 6. Styling
    # ------------------------------------------------------------------
    if title is None:
        title = f"Phenotypic Principal Graph  (space-time MST, axis={spatial_axis})"
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Time (hpf)", fontsize=10)
    ax.set_ylabel(sp_label, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Branch allocation bar charts
# ---------------------------------------------------------------------------

def plot_branch_allocation_bars(
    branch_results: list[BranchTestResult],
    color_map: dict[str, str] | None = None,
    figsize_per: tuple[float, float] = (4.5, 3.5),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Stacked bar: fraction of embryos per arm per genotype at each branch."""
    n = len(branch_results)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize_per)
        ax.text(0.5, 0.5, "No branch points found",
                ha="center", va="center", transform=ax.transAxes)
        return fig, [ax]

    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per[0] * n, figsize_per[1]),
        squeeze=False,
    )
    axes = list(axes[0])
    default_color = "#aaaaaa"

    for ax, res in zip(axes, branch_results):
        counts = res.counts_ge   # (genotypes × arms)
        genotypes = counts.index.tolist()
        arm_cols = counts.columns.tolist()
        x = np.arange(len(arm_cols))
        bottom = np.zeros(len(arm_cols))

        totals = counts.sum(axis=0).values.astype(float)
        totals = np.where(totals == 0, 1, totals)

        for geno in genotypes:
            vals = counts.loc[geno].values.astype(float)
            fracs = vals / totals
            color = color_map.get(geno, default_color) if color_map else default_color
            short = geno.replace("_crispant", "").replace("_", " ")
            ax.bar(x, fracs, bottom=bottom,
                   color=color, label=short,
                   edgecolor="white", linewidth=0.5)
            bottom += fracs

        star = _sig_star(res.pval)
        pval_text = _pval_label(res.pval)
        ax.set_title(
            f"Branch {res.node_id}  (t≈{res.t_hpf_branch:.0f} hpf)\n"
            f"{star} {pval_text}  V={res.effect_size:.2f}",
            fontsize=9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [c.replace("arm_", "arm ") for c in arm_cols],
            fontsize=8, rotation=30, ha="right",
        )
        ax.set_ylabel("Fraction of embryos", fontsize=8)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    return fig, axes
