"""
Cluster flow visualization utilities.
"""

from typing import Dict, List, Optional
from pathlib import Path


def plot_cluster_flow(
    results: Dict,
    k_range: Optional[List[int]] = None,
    title: str = "Cluster Flow Across k Values",
    output_path: Optional[Path] = None,
    color_palette: Optional[List[str]] = None,
):
    """
    Create a Sankey diagram showing how clusters split as k increases.

    Parameters
    ----------
    results : Dict
        Output from run_k_selection_with_plots() or evaluate_k_range().
        Must contain 'clustering_by_k' with assignments for each k.
    k_range : List[int], optional
        Which k values to include. If None, uses all available k values.
    title : str
        Plot title.
    output_path : Path, optional
        If provided, saves the plot as HTML.
    color_palette : List[str], optional
        Colors for clusters.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    clustering_by_k = results["clustering_by_k"]

    if k_range is None:
        k_range = sorted(clustering_by_k.keys())
    else:
        k_range = [k for k in k_range if k in clustering_by_k]

    if len(k_range) < 2:
        raise ValueError("Need at least 2 k values to create a flow diagram")

    if color_palette is None:
        color_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        ]

    def hex_to_rgba(hex_color: str, alpha: float = 0.5) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    node_labels = []
    node_colors = []
    node_index = {}

    for k in k_range:
        cluster_to_embryos = clustering_by_k[k]["assignments"]["cluster_to_embryos"]
        n_clusters = len(cluster_to_embryos)
        for c in range(n_clusters):
            idx = len(node_labels)
            node_index[(k, c)] = idx
            n_embryos = len(cluster_to_embryos.get(c, []))
            node_labels.append(f"k={k}: C{c} (n={n_embryos})")
            node_colors.append(color_palette[c % len(color_palette)])

    sources = []
    targets = []
    values = []
    link_colors = []

    for i in range(len(k_range) - 1):
        k_from = k_range[i]
        k_to = k_range[i + 1]

        assignments_from = clustering_by_k[k_from]["assignments"]["embryo_to_cluster"]
        assignments_to = clustering_by_k[k_to]["assignments"]["embryo_to_cluster"]

        flow_counts = {}

        for embryo_id, cluster_from in assignments_from.items():
            if embryo_id in assignments_to:
                cluster_to = assignments_to[embryo_id]
                key = (cluster_from, cluster_to)
                flow_counts[key] = flow_counts.get(key, 0) + 1

        for (cluster_from, cluster_to), count in flow_counts.items():
            sources.append(node_index[(k_from, cluster_from)])
            targets.append(node_index[(k_to, cluster_to)])
            values.append(count)
            link_colors.append(hex_to_rgba(color_palette[cluster_from % len(color_palette)], 0.5))

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        )
    )])

    fig.update_layout(
        title_text=title,
        font_size=12,
        height=400 + 50 * len(k_range),
        width=250 * len(k_range),
    )

    if output_path:
        fig.write_html(str(output_path))
        print(f"✓ Saved cluster flow diagram: {output_path}")

    return fig
