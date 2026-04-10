"""
principal_tree_viz.py
---------------------
Visualization for the ElPiGraph elastic principal tree fitted to the
embryo-level 3D space-time cloud.

Public API
----------
  plot_tree_schematic(nodes_df, edges_df, branch_results, obs_df,
                      labels, color_map, spatial_axis, ...)  -> (fig, ax)

  plot_branch_allocation_bars(branch_results, color_map, ...) -> (fig, axes)

  plot_tree_3d(nodes_df, edges_df, obs_df, labels, color_map, ...) -> (fig, ax)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from .principal_tree import BranchTestResult


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


def _t_scaled_to_hpf(t_sc: float, t_min_hpf: float, t_max_hpf: float, t_weight: float) -> float:
    """Invert the t_scaled = (hpf - t_min) / (t_max - t_min) * t_weight transform."""
    t_range = (t_max_hpf - t_min_hpf) or 1.0
    return t_min_hpf + t_sc / t_weight * t_range


# ---------------------------------------------------------------------------
# 2D schematic
# ---------------------------------------------------------------------------

def plot_tree_schematic(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    branch_results: list[BranchTestResult],
    obs_df: pd.DataFrame,
    labels: np.ndarray,
    color_map: dict[str, str] | None = None,
    spatial_axis: str = "y",
    t_min_hpf: float | None = None,
    t_max_hpf: float | None = None,
    t_weight: float = 3.0,
    figsize: tuple[float, float] = (14, 7),
    scatter_alpha: float = 0.45,
    scatter_size: float = 10.0,
    max_edge_lw: float = 4.0,
    min_edge_lw: float = 1.0,
    node_size: float = 60.0,
    alpha_threshold: float = 0.05,
    annotate_ns: bool = False,
    min_n_embryos: int = 0,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Render the 2D space-time schematic.

    x-axis: time (hpf, back-converted from t_scaled)
    y-axis: condensed x or y (depending on spatial_axis)

    Parameters
    ----------
    nodes_df, edges_df : from fit_principal_tree
    branch_results : from run_all_branch_tests
    obs_df : from build_embryo_spacetime_cloud
    labels : (N_e,) genotype strings
    t_min_hpf, t_max_hpf : time range for axis conversion; inferred from obs_df if None
    """
    if spatial_axis not in ("x", "y"):
        raise ValueError("spatial_axis must be 'x' or 'y'")

    sp_col = spatial_axis          # "x" or "y" in obs_df and nodes_df
    sp_label = f"Condensed {spatial_axis}"

    if t_min_hpf is None:
        t_min_hpf = float(obs_df["t_hpf"].min())
    if t_max_hpf is None:
        t_max_hpf = float(obs_df["t_hpf"].max())

    # Back-convert node t_scaled → hpf
    nodes_hpf = nodes_df["t_scaled"].apply(
        lambda ts: _t_scaled_to_hpf(ts, t_min_hpf, t_max_hpf, t_weight)
    )

    fig, ax = plt.subplots(figsize=figsize)
    default_color = "#aaaaaa"

    # ------------------------------------------------------------------
    # 1. Background scatter
    # ------------------------------------------------------------------
    for emb_i in obs_df["embryo_idx"].unique():
        geno = str(labels[int(emb_i)])
        color = color_map.get(geno, default_color) if color_map else default_color
        sub = obs_df[obs_df["embryo_idx"] == emb_i]
        ax.scatter(
            sub["t_hpf"], sub[sp_col],
            c=color, s=scatter_size, alpha=scatter_alpha,
            linewidths=0, zorder=1,
        )

    # ------------------------------------------------------------------
    # 2. Tree edges
    # ------------------------------------------------------------------
    for _, e in edges_df.iterrows():
        a, b = int(e["source"]), int(e["target"])
        xa, xb = float(nodes_hpf.iloc[a]), float(nodes_hpf.iloc[b])
        ya, yb = float(nodes_df.loc[a, sp_col]), float(nodes_df.loc[b, sp_col])
        ax.plot([xa, xb], [ya, yb],
                color="#444444", linewidth=2.0,
                solid_capstyle="round", zorder=2)

    # ------------------------------------------------------------------
    # 3. Non-branch nodes
    # ------------------------------------------------------------------
    for _, row in nodes_df.iterrows():
        if row["degree"] >= 3:
            continue
        ax.scatter(
            nodes_hpf.iloc[int(row["node_id"])], row[sp_col],
            c="#666666", s=node_size * 0.5,
            zorder=4, edgecolors="white", linewidths=0.8,
        )

    # ------------------------------------------------------------------
    # 4. Branch nodes annotated
    # ------------------------------------------------------------------
    br_map = {r.node_id: r for r in branch_results}

    for bn, res in br_map.items():
        bx = float(nodes_hpf.iloc[bn])
        by = float(nodes_df.loc[bn, sp_col])

        is_sig = res.pval < alpha_threshold
        skip_annotation = (not is_sig and not annotate_ns) or \
                          (res.n_embryos < min_n_embryos)

        # When annotate_ns=False, hide ns branch nodes entirely
        if skip_annotation and not is_sig:
            continue

        marker_color = "#C62828" if is_sig else "#757575"
        ms = 180 if is_sig else 80

        ax.scatter(bx, by, marker="D",
                   c=marker_color, s=ms,
                   zorder=6, edgecolors="white", linewidths=1.0,
                   alpha=1.0)

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

    handles.append(
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor="#C62828", markersize=8,
               label="Branch (p < 0.05)"),
    )
    if annotate_ns:
        handles.append(
            Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="#757575", markersize=7,
                   label="Branch (ns)"),
        )
    ax.legend(handles=handles, loc="upper left",
              fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    # ------------------------------------------------------------------
    # 6. Styling
    # ------------------------------------------------------------------
    if title is None:
        title = f"Phenotypic Principal Tree  (ElPiGraph, axis={spatial_axis})"
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Time (hpf)", fontsize=10)
    ax.set_ylabel(sp_label, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Branch allocation bars
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
        counts = res.counts_ge
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
            f"Branch {res.node_id}  (t_sc≈{res.t_hpf_branch:.2f})\n"
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


# ---------------------------------------------------------------------------
# 3D diagnostic
# ---------------------------------------------------------------------------

def plot_tree_3d(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    labels: np.ndarray,
    color_map: dict[str, str] | None = None,
    scatter_alpha: float = 0.35,
    scatter_size: float = 8.0,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """3D scatter of the embryo cloud with fitted tree overlaid.

    Axes: x=condensed_x, y=condensed_y, z=t_scaled.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    default_color = "#aaaaaa"

    for emb_i in obs_df["embryo_idx"].unique():
        geno = str(labels[int(emb_i)])
        color = color_map.get(geno, default_color) if color_map else default_color
        sub = obs_df[obs_df["embryo_idx"] == emb_i]
        ax.scatter(
            sub["x"], sub["y"], sub["t_scaled"],
            c=color, s=scatter_size, alpha=scatter_alpha,
            linewidths=0, zorder=1,
        )

    # Tree edges
    for _, e in edges_df.iterrows():
        a, b = int(e["source"]), int(e["target"])
        xa, xb = float(nodes_df.loc[a, "x"]), float(nodes_df.loc[b, "x"])
        ya, yb = float(nodes_df.loc[a, "y"]), float(nodes_df.loc[b, "y"])
        ta, tb = float(nodes_df.loc[a, "t_scaled"]), float(nodes_df.loc[b, "t_scaled"])
        ax.plot([xa, xb], [ya, yb], [ta, tb],
                color="#333333", linewidth=1.5, zorder=3)

    # Branch nodes
    branch_nodes = nodes_df[nodes_df["degree"] >= 3]
    if len(branch_nodes) > 0:
        ax.scatter(
            branch_nodes["x"], branch_nodes["y"], branch_nodes["t_scaled"],
            c="#C62828", s=80, marker="D", zorder=5,
            edgecolors="white", linewidths=0.8,
        )

    ax.set_xlabel("Condensed x", fontsize=9)
    ax.set_ylabel("Condensed y", fontsize=9)
    ax.set_zlabel("t scaled", fontsize=9)

    if title is None:
        title = "Principal Tree — 3D diagnostic"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Rotating 3D GIF
# ---------------------------------------------------------------------------

def save_tree_3d_gif(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    labels: np.ndarray,
    out_path: str,
    color_map: dict[str, str] | None = None,
    scatter_alpha: float = 0.35,
    scatter_size: float = 8.0,
    figsize: tuple[float, float] = (8, 7),
    n_frames: int = 72,
    fps: int = 18,
    elev: float = 25.0,
    title: str | None = None,
) -> None:
    """Save a rotating 3D GIF of the embryo cloud + fitted principal tree.

    Parameters
    ----------
    out_path : path ending in .gif
    n_frames : number of rotation frames (360 / n_frames = degrees per frame)
    fps : frames per second
    elev : elevation angle (degrees)
    """
    default_color = "#aaaaaa"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Scatter per genotype
    for emb_i in obs_df["embryo_idx"].unique():
        geno = str(labels[int(emb_i)])
        color = color_map.get(geno, default_color) if color_map else default_color
        sub = obs_df[obs_df["embryo_idx"] == emb_i]
        ax.scatter(
            sub["x"], sub["y"], sub["t_scaled"],
            c=color, s=scatter_size, alpha=scatter_alpha,
            linewidths=0, zorder=1,
        )

    # Tree edges
    for _, e in edges_df.iterrows():
        a, b = int(e["source"]), int(e["target"])
        ax.plot(
            [nodes_df.loc[a, "x"], nodes_df.loc[b, "x"]],
            [nodes_df.loc[a, "y"], nodes_df.loc[b, "y"]],
            [nodes_df.loc[a, "t_scaled"], nodes_df.loc[b, "t_scaled"]],
            color="#111111", linewidth=2.0, zorder=3,
        )

    # Branch nodes
    branch_nodes = nodes_df[nodes_df["degree"] >= 3]
    if len(branch_nodes) > 0:
        ax.scatter(
            branch_nodes["x"], branch_nodes["y"], branch_nodes["t_scaled"],
            c="#C62828", s=100, marker="D", zorder=5,
            edgecolors="white", linewidths=0.8,
        )

    ax.set_xlabel("Condensed x", fontsize=9)
    ax.set_ylabel("Condensed y", fontsize=9)
    ax.set_zlabel("t scaled", fontsize=9)

    if title is None:
        title = "Principal Tree — 3D"
    ax.set_title(title, fontsize=10)

    def _update(frame: int) -> None:
        ax.view_init(elev=elev, azim=frame * 360 / n_frames)

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 / fps,
    )
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metro-map layout
# ---------------------------------------------------------------------------

def _assign_segment_tracks(
    segments: list[list[tuple[int, int]]],
    nodes_df: pd.DataFrame,
    track_spacing: float = 1.0,
) -> dict[int, float]:
    """Assign a y-track to each segment index.

    Strategy:
    - Build a graph of segments connected at shared structural endpoints.
    - Root connected components at the segment whose mean t_scaled is earliest.
    - BFS from root: root segment gets y=0; at each branch node, child segments
      fan out symmetrically around the parent's track.
    - Disconnected components are stacked below the main component with a gap.

    Returns
    -------
    dict mapping segment index -> y track position
    """
    from collections import defaultdict, deque

    def ek(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    # Structural endpoint nodes for each segment
    def seg_endpoints(seg: list[tuple[int, int]]) -> tuple[int, int]:
        return seg[0][0], seg[-1][1]

    n = len(segments)
    endpoints = [seg_endpoints(s) for s in segments]

    # Mean t_scaled per segment (for rooting)
    node_t = dict(zip(nodes_df["node_id"].astype(int), nodes_df["t_scaled"]))
    seg_mean_t = [
        np.mean([node_t.get(a, 0), node_t.get(b, 0)])
        for a, b in endpoints
    ]

    # Build segment adjacency: two segments are adjacent if they share an endpoint
    node_to_segs: dict[int, list[int]] = defaultdict(list)
    for si, (a, b) in enumerate(endpoints):
        node_to_segs[a].append(si)
        node_to_segs[b].append(si)

    seg_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for node, segs in node_to_segs.items():
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                seg_adj[segs[i]].append((segs[j], node))
                seg_adj[segs[j]].append((segs[i], node))

    # Find connected components of segments
    visited = set()
    components: list[list[int]] = []
    for start in range(n):
        if start in visited:
            continue
        comp = []
        q = deque([start])
        while q:
            si = q.popleft()
            if si in visited:
                continue
            visited.add(si)
            comp.append(si)
            for (nb, _) in seg_adj[si]:
                if nb not in visited:
                    q.append(nb)
        components.append(comp)

    track: dict[int, float] = {}
    component_base = 0.0

    for comp in components:
        # Root = segment with earliest mean t_scaled
        root_si = min(comp, key=lambda si: seg_mean_t[si])

        # BFS to assign tracks within this component
        # State: (seg_idx, parent_seg_idx, shared_node, parent_track)
        visited_segs: set[int] = set()
        # Map: branch_node -> list of child segment indices already assigned
        branch_children: dict[int, list[int]] = defaultdict(list)

        q: deque[tuple[int, float]] = deque([(root_si, component_base)])
        # We'll collect all assignments then center them
        raw_assignments: list[tuple[int, float]] = []

        while q:
            si, parent_y = q.popleft()
            if si in visited_segs:
                continue
            visited_segs.add(si)
            track[si] = parent_y
            raw_assignments.append((si, parent_y))

            # Find the shared node with parent (exit node of this segment)
            ep_a, ep_b = endpoints[si]
            neighbors = [(nb_si, shared_node) for nb_si, shared_node in seg_adj[si]
                         if nb_si not in visited_segs]

            # Group neighbors by which endpoint they share
            by_node: dict[int, list[int]] = defaultdict(list)
            for nb_si, shared_node in neighbors:
                by_node[shared_node].append(nb_si)

            for shared_node, children in by_node.items():
                nc = len(children)
                # Fan children symmetrically around parent_y
                offsets = np.arange(nc) * track_spacing
                offsets -= offsets.mean()
                for child_si, offset in zip(children, offsets):
                    q.append((child_si, parent_y + offset))

        # Offset this component below the previous one
        if track:
            used_ys = [track[si] for si in comp if si in track]
            component_base = min(used_ys) - track_spacing * 2.0

    return track


def plot_tree_metromap(
    segments: list[list[tuple[int, int]]],
    nodes_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    labels: np.ndarray,
    branch_results: list["BranchTestResult"] | None = None,
    color_map: dict[str, str] | None = None,
    t_min_hpf: float | None = None,
    t_max_hpf: float | None = None,
    t_weight: float = 3.0,
    track_spacing: float = 1.0,
    scatter_alpha: float = 0.4,
    scatter_size: float = 8.0,
    jitter_scale: float = 0.08,
    figsize: tuple[float, float] = (14, 7),
    title: str | None = None,
    seed: int = 42,
) -> tuple[plt.Figure, plt.Axes]:
    """Metro-map layout: x=time (hpf), y=segment track.

    Each pruned segment is a horizontal track. Embryo observations are
    scattered along their segment's track with small vertical jitter.
    Branch points are annotated if branch_results are provided.

    Parameters
    ----------
    segments    : pruned segments from prune_phantom_segments
    nodes_df    : tree nodes with t_scaled
    proj_df     : observation_projections (nearest_edge_a/b, embryo_idx, t_hpf)
    labels      : (N_e,) genotype strings
    branch_results : optional — for annotating significant branch points
    t_min_hpf, t_max_hpf : for converting t_scaled → hpf (inferred if None)
    track_spacing : vertical distance between tracks
    jitter_scale  : std of vertical jitter within a track
    """
    rng = np.random.default_rng(seed)
    default_color = "#aaaaaa"

    node_t = dict(zip(nodes_df["node_id"].astype(int), nodes_df["t_scaled"]))

    # Infer time range from proj_df if not provided
    if t_min_hpf is None:
        t_min_hpf = float(proj_df["t_hpf"].min())
    if t_max_hpf is None:
        t_max_hpf = float(proj_df["t_hpf"].max())
    t_range = (t_max_hpf - t_min_hpf) or 1.0

    def t_scaled_to_hpf(ts: float) -> float:
        return t_min_hpf + ts / t_weight * t_range

    # Assign tracks
    seg_track = _assign_segment_tracks(segments, nodes_df, track_spacing)

    # Build lookup: edge -> segment index
    def ek(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    edge_to_seg: dict[tuple[int, int], int] = {}
    for si, seg in enumerate(segments):
        for a, b in seg:
            edge_to_seg[ek(a, b)] = si

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # 1. Scatter embryo observations along tracks
    # ------------------------------------------------------------------
    for _, row in proj_df.iterrows():
        emb_i = int(row["embryo_idx"])
        edge = ek(int(row["nearest_edge_a"]), int(row["nearest_edge_b"]))
        si = edge_to_seg.get(edge)
        if si is None:
            continue  # pruned edge — skip
        y_track = seg_track.get(si, 0.0)
        geno = str(labels[emb_i])
        color = color_map.get(geno, default_color) if color_map else default_color
        jitter = rng.normal(0, jitter_scale)
        ax.scatter(
            y_track + jitter, row["t_hpf"],
            c=color, s=scatter_size, alpha=scatter_alpha,
            linewidths=0, zorder=1,
        )

    # ------------------------------------------------------------------
    # 2. Draw segment lines
    # ------------------------------------------------------------------
    for si, seg in enumerate(segments):
        x_track = seg_track.get(si, 0.0)
        node_ids_in_seg = [seg[0][0]] + [b for a, b in seg]
        t_hpfs = [t_scaled_to_hpf(node_t[n]) for n in node_ids_in_seg]
        t_start, t_end = min(t_hpfs), max(t_hpfs)
        ax.plot([x_track, x_track], [t_start, t_end],
                color="#333333", linewidth=2.0,
                solid_capstyle="round", zorder=3)

    # ------------------------------------------------------------------
    # 3. Draw branch connections (horizontal lines at branch nodes)
    # ------------------------------------------------------------------
    from collections import defaultdict
    node_to_segs: dict[int, list[int]] = defaultdict(list)
    for si, seg in enumerate(segments):
        ep_a, ep_b = seg[0][0], seg[-1][1]
        node_to_segs[ep_a].append(si)
        node_to_segs[ep_b].append(si)

    branch_node_ids = set(nodes_df.loc[nodes_df["degree"] >= 3, "node_id"].astype(int))

    for node_id in branch_node_ids:
        connected_segs = node_to_segs.get(node_id, [])
        if len(connected_segs) < 2:
            continue
        t_hpf_node = t_scaled_to_hpf(node_t[node_id])
        xs = [seg_track[si] for si in connected_segs if si in seg_track]
        if len(xs) < 2:
            continue
        ax.plot([min(xs), max(xs)], [t_hpf_node, t_hpf_node],
                color="#333333", linewidth=1.5,
                linestyle="--", zorder=2, alpha=0.6)

    # ------------------------------------------------------------------
    # 4. Annotate significant branch nodes
    # ------------------------------------------------------------------
    if branch_results:
        br_map = {r.node_id: r for r in branch_results}
        for node_id, res in br_map.items():
            if node_id not in node_to_segs:
                continue
            connected_segs = node_to_segs[node_id]
            xs = [seg_track[si] for si in connected_segs if si in seg_track]
            if not xs:
                continue
            t_hpf_node = t_scaled_to_hpf(node_t[node_id])
            x_mid = np.mean(xs)
            is_sig = res.pval < 0.05
            if not is_sig:
                continue
            color = "#C62828"
            ax.scatter(x_mid, t_hpf_node, marker="D",
                       c=color, s=180, zorder=6,
                       edgecolors="white", linewidths=1.0)
            star = _sig_star(res.pval)
            ax.annotate(
                f"{star}  {_pval_label(res.pval)}\nV={res.effect_size:.2f}  n={res.n_embryos}",
                xy=(x_mid, t_hpf_node),
                xytext=(8, -22), textcoords="offset points",
                fontsize=7.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=color, alpha=0.88, linewidth=0.8),
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
    if branch_results:
        handles.append(Line2D([0], [0], marker="D", color="w",
                               markerfacecolor="#C62828", markersize=8,
                               label="Branch (p < 0.05)"))
    ax.legend(handles=handles, loc="upper left",
              fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    # ------------------------------------------------------------------
    # 6. Styling
    # ------------------------------------------------------------------
    ax.set_ylabel("Time (hpf)", fontsize=10)
    ax.set_xlabel("Segment track", fontsize=10)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    return fig, ax
