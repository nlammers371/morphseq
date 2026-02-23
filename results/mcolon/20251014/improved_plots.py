"""
Improved visualization functions focused on signed_margin metric
with clearer genotype differentiation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
except ImportError:
    linkage = None
    leaves_list = None
    pdist = None

try:
    from matplotlib import colormaps as mpl_colormaps
except ImportError:
    mpl_colormaps = None


def plot_signed_margin_heatmap(df_embryo_probs, df_penetrance, group1, group2, output_path=None):
    """
    Heatmap showing signed margin trajectories per embryo, split by genotype.

    Signed margin represents classifier confidence with directionality: positive
    values indicate correct predictions, negative indicate incorrect. Embryos are
    grouped by genotype with prominent labels and visual separators.
    """
    if df_embryo_probs.empty or df_penetrance.empty:
        print("  Skipping heatmap: no embryo data")
        return None

    if 'signed_margin' not in df_embryo_probs.columns:
        print("  Skipping heatmap: signed_margin column not found")
        return None

    embryo_to_label = df_penetrance.set_index('embryo_id')['true_label'].to_dict()
    label_order = [label for label in [group1, group2] if label in df_embryo_probs['true_label'].unique()]
    if not label_order:
        label_order = list(df_embryo_probs['true_label'].unique())

    time_bins = sorted(df_embryo_probs['time_bin'].unique())

    # Create pivot for signed margin
    pivot = df_embryo_probs.pivot_table(
        index='embryo_id',
        columns='time_bin',
        values='signed_margin',
        aggfunc='mean'
    )

    def _order_embryos_for_label(label, pivot_matrix):
        label_embryos = [eid for eid, lbl in embryo_to_label.items() if lbl == label]
        subset = pivot_matrix.loc[pivot_matrix.index.intersection(label_embryos)]
        if subset.empty:
            return []

        ordered_index = None
        if linkage and pdist and subset.shape[0] > 1:
            filled = subset.copy()
            col_means = filled.mean(axis=0)
            filled = filled.fillna(col_means)
            filled = filled.fillna(0.0)
            try:
                dist = pdist(filled.values, metric='euclidean')
                if np.any(dist > 0):
                    cluster = linkage(dist, method='ward')
                    ordered_index = list(filled.index[leaves_list(cluster)])
            except Exception as exc:
                print(f"  Ward clustering failed for {label}: {exc}")

        if ordered_index is None:
            ranking_metric = 'mean_signed_margin' if 'mean_signed_margin' in df_penetrance.columns else 'mean_confidence'
            ordered_index = list(
                df_penetrance[df_penetrance['embryo_id'].isin(subset.index)]
                .sort_values(ranking_metric, ascending=False)
                ['embryo_id']
            )

        return ordered_index

    row_sections = []
    for label in label_order:
        ordered_ids = _order_embryos_for_label(label, pivot)
        if ordered_ids:
            row_sections.append((label, ordered_ids))

    row_order = [eid for _, ids in row_sections for eid in ids]
    if not row_order:
        print("  Skipping heatmap: no embryos to plot")
        return None

    fig_height = max(8.0, len(row_order) * 0.2)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    # Reindex and prepare data
    pivot_plot = pivot.reindex(row_order).reindex(columns=time_bins)
    data = pivot_plot.values.astype(float)
    mask = np.isnan(data)

    # Get colormap (RdBu_r: red=positive/correct, blue=negative/incorrect)
    cmap_name = 'RdBu_r'
    if mpl_colormaps is not None:
        cmap_instance = mpl_colormaps[cmap_name].copy()
    else:
        cmap_instance = plt.get_cmap(cmap_name).copy()
    cmap_instance.set_bad(color='lightgray')
    data_masked = np.ma.masked_array(data, mask=mask)

    im = ax.imshow(data_masked, aspect='auto', cmap=cmap_instance, vmin=-0.5, vmax=0.5)

    # Set ticks
    ax.set_xticks(np.arange(len(time_bins)))
    ax.set_xticklabels(time_bins, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Developmental Time (hpf)', fontsize=13, fontweight='bold')

    # Y-axis: just embryo IDs
    row_labels = [str(eid)[:15] for eid in row_order]
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_ylabel('Embryo ID', fontsize=13, fontweight='bold')

    # Add genotype section labels and separators
    section_sizes = [len(ids) for _, ids in row_sections if ids]
    section_boundaries = np.cumsum(section_sizes)

    for idx, (label, ids) in enumerate(row_sections):
        if not ids:
            continue
        start_pos = section_boundaries[idx - 1] if idx > 0 else 0
        end_pos = section_boundaries[idx]
        center_pos = (start_pos + end_pos) / 2

        # Add genotype label on the right side
        ax.text(len(time_bins) + 0.5, center_pos, label.split('_')[-1].upper(),
                fontsize=12, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))

        # Add separator line
        if idx < len(row_sections) - 1:
            ax.axhline(end_pos - 0.5, color='black', linewidth=3, alpha=0.8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Signed Margin\n(RED: correct, BLUE: incorrect)',
                   rotation=270, labelpad=25, fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(f'Embryo Classification Trajectories: {group1.split("_")[-1].upper()} vs {group2.split("_")[-1].upper()}',
                fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    return fig
