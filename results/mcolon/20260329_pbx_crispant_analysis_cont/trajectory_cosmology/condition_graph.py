"""
condition_graph.py
------------------
Level 1: condition-level similarity graph from pairwise AUROC.

At each time bin, builds a G×G weighted graph where edge weight = 1 - AUROC.
High weight = conditions are similar. Low weight = diverged.

AUROC is used raw and directional — no symmetrization. An AUROC < 0.5
(model is anti-predictive in that direction) is itself a meaningful signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_condition_graphs(
    auroc_df: pd.DataFrame,
    conditions: list[str],
    pair_col: str = "pair_id",
    time_col: str = "time_bin_center",
    auroc_col: str = "auroc_obs",
    group1_col: str = "group1",
    group2_col: str = "group2",
) -> dict[float, np.ndarray]:
    """Build G×G similarity matrices at each time bin.

    Parameters
    ----------
    auroc_df
        pairwise_auroc_bins_*.csv — one row per pair per time bin.
    conditions
        Ordered list of condition names; defines matrix row/col order.
    ...

    Returns
    -------
    graphs : dict mapping time_bin_center → (G, G) similarity matrix
        similarity[i, j] = 1 - AUROC(conditions[i] vs conditions[j])
        Diagonal is 1.0. Missing pairs are NaN.
    """
    G = len(conditions)
    cond_index = {c: i for i, c in enumerate(conditions)}
    time_bins = sorted(auroc_df[time_col].unique())
    graphs = {}

    for t in time_bins:
        W = np.full((G, G), np.nan)
        np.fill_diagonal(W, 1.0)
        rows = auroc_df[auroc_df[time_col] == t]
        for _, row in rows.iterrows():
            g1, g2 = row[group1_col], row[group2_col]
            if g1 not in cond_index or g2 not in cond_index:
                continue
            i, j = cond_index[g1], cond_index[g2]
            W[i, j] = 1.0 - row[auroc_col]
        graphs[t] = W

    return graphs


def layout_condition_graphs(
    graphs: dict[float, np.ndarray],
    conditions: list[str],
    seed: int = 42,
) -> dict[float, dict[str, tuple[float, float]]]:
    """Compute warm-started spring layouts for each time bin.

    Parameters
    ----------
    graphs : output of build_condition_graphs
    conditions : ordered condition names

    Returns
    -------
    layouts : dict mapping time_bin_center → {condition: (x, y)}
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for condition graph layout")

    time_bins = sorted(graphs.keys())
    layouts = {}
    prev_pos = None

    for t in time_bins:
        W = graphs[t]
        G_graph = nx.Graph()
        G_graph.add_nodes_from(conditions)

        for i, c1 in enumerate(conditions):
            for j, c2 in enumerate(conditions):
                if i >= j:
                    continue
                w = W[i, j]
                if not np.isnan(w):
                    G_graph.add_edge(c1, c2, weight=w)

        pos = nx.spring_layout(
            G_graph,
            weight="weight",
            pos=prev_pos,
            seed=seed,
        )
        layouts[t] = {c: (float(pos[c][0]), float(pos[c][1])) for c in conditions}
        prev_pos = pos

    return layouts


def plot_condition_graph_snapshots(
    graphs: dict[float, np.ndarray],
    layouts: dict[float, dict[str, tuple[float, float]]],
    conditions: list[str],
    snapshot_times: list[float],
    color_map: dict[str, str] | None = None,
    figsize: tuple | None = None,
):
    """Plot G×G condition graph at selected time bins.

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    n = len(snapshot_times)
    if figsize is None:
        figsize = (4 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    if color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {c: cmap(i) for i, c in enumerate(conditions)}

    for ax, t in zip(axes, snapshot_times):
        W = graphs.get(t)
        if W is None:
            ax.set_title(f"{t} hpf (no data)")
            continue

        G_graph = nx.Graph()
        G_graph.add_nodes_from(conditions)
        for i, c1 in enumerate(conditions):
            for j, c2 in enumerate(conditions):
                if i >= j:
                    continue
                w = W[i, j]
                if not np.isnan(w):
                    G_graph.add_edge(c1, c2, weight=float(w))

        pos = {c: layouts[t][c] for c in conditions}
        node_colors = [color_map[c] for c in conditions]
        weights = [G_graph[u][v]["weight"] * 3 for u, v in G_graph.edges()]

        nx.draw_networkx(
            G_graph, pos=pos, ax=ax,
            node_color=node_colors,
            width=weights,
            node_size=400,
            font_size=7,
        )
        ax.set_title(f"{t} hpf")
        ax.axis("off")

    fig.tight_layout()
    return fig, axes
