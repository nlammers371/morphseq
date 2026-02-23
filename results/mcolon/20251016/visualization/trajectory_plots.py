"""
Signed margin trajectory and heatmap visualization.

This module provides functions for visualizing embryo-level signed margin
trajectories and heatmaps across developmental time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional

try:
    from matplotlib import colormaps as mpl_colormaps
except ImportError:
    mpl_colormaps = None

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
except ImportError:
    linkage = None
    leaves_list = None
    pdist = None


def plot_signed_margin_trajectories(
    df_embryo_probs: pd.DataFrame,
    df_penetrance: pd.DataFrame,
    group1: str,
    group2: str,
    output_path: Optional[str] = None,
    max_embryos: int = 30
) -> Optional[plt.Figure]:
    """
    Plot signed-margin trajectories per genotype highlighting decision-boundary crossings.

    Line colors encode the embryo's mean signed margin (blue = negative / wildtype-like,
    red = positive / phenotype).

    Parameters
    ----------
    df_embryo_probs : pd.DataFrame
        Per-embryo predictions with signed_margin column
    df_penetrance : pd.DataFrame
        Penetrance metrics with mean_signed_margin
    group1, group2 : str
        Genotype labels
    output_path : str or None
        Path to save figure
    max_embryos : int, default=30
        Maximum number of embryos to plot per genotype

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure, or None if insufficient data
    """
    if df_embryo_probs.empty or df_penetrance.empty:
        print("  Skipping trajectories: no embryo data")
        return None

    if 'signed_margin' not in df_embryo_probs.columns:
        print("  Skipping trajectories: signed_margin column not found")
        return None

    genotype_candidates = df_penetrance['true_label'].dropna().unique()
    genotypes = [g for g in [group1, group2] if g in genotype_candidates]
    if not genotypes:
        genotypes = sorted(genotype_candidates)
    if len(genotypes) == 0:
        print("  Skipping trajectories: no genotypes found")
        return None

    fig, axes = plt.subplots(1, len(genotypes), figsize=(8 * max(1, len(genotypes)), 6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, genotype in zip(axes, genotypes):
        genotype_penetrance = df_penetrance[df_penetrance['true_label'] == genotype].copy()
        if genotype_penetrance.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{genotype} (n=0)', fontsize=13, fontweight='bold')
            continue

        genotype_penetrance = genotype_penetrance.assign(
            abs_margin=np.abs(genotype_penetrance.get('mean_signed_margin', np.nan))
        )
        genotype_penetrance = genotype_penetrance.sort_values(
            by=['abs_margin', 'mean_signed_margin'], ascending=[False, False]
        ).head(max_embryos)
        top_embryos = genotype_penetrance['embryo_id'].values

        norm = Normalize(vmin=-0.5, vmax=0.5)
        cmap = plt.cm.RdBu_r
        alphas = np.linspace(0.35, 0.9, len(top_embryos)) if len(top_embryos) > 0 else []
        highlight_id = genotype_penetrance.iloc[0]['embryo_id'] if len(genotype_penetrance) > 0 else None
        penetrance_lookup = genotype_penetrance.set_index('embryo_id')

        for alpha, embryo_id in zip(alphas, top_embryos):
            embryo_curve = df_embryo_probs[df_embryo_probs['embryo_id'] == embryo_id].sort_values('time_bin')
            if embryo_curve.empty:
                continue

            mean_margin = np.nan
            if 'mean_signed_margin' in penetrance_lookup.columns:
                mean_margin = penetrance_lookup.at[embryo_id, 'mean_signed_margin']
            if np.isnan(mean_margin):
                mean_margin = embryo_curve['signed_margin'].mean()

            base_color = cmap(norm(mean_margin))
            color_rgba = (base_color[0], base_color[1], base_color[2], alpha)
            linewidth = 1.6
            marker_size = 3

            if embryo_id == highlight_id:
                linewidth = 2.8
                marker_size = 4
                color_rgba = cmap(norm(mean_margin))

            ax.plot(
                embryo_curve['time_bin'],
                embryo_curve['signed_margin'],
                color=color_rgba,
                linewidth=linewidth,
                marker='o',
                markersize=marker_size,
                alpha=0.95
            )

        ax.axhline(0.0, color='red', linestyle='--', linewidth=1.3, alpha=0.7,
                  label='Decision boundary')
        ax.set_xlabel('Time (hpf)', fontsize=12)
        ax.set_ylabel('Signed Margin vs 0.5', fontsize=12)
        ax.set_ylim([-0.5, 0.5])
        ax.grid(alpha=0.3)
        ax.set_title(f'{genotype} (n={len(top_embryos)})', fontsize=14, fontweight='bold')

        if len(top_embryos) > 0:
            ax.legend(loc='upper left', fontsize=9)

    fig.suptitle(
        f'Embryo Signed Margin Trajectories: {group1.split("_")[-1]} vs {group2.split("_")[-1]}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_signed_margin_heatmap(
    df_embryo_probs: pd.DataFrame,
    df_penetrance: pd.DataFrame,
    group1: str,
    group2: str,
    output_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Heatmap showing signed-margin trajectories per embryo, split by genotype.

    Signed margin measures distance from the 0.5 decision boundary with sign,
    making it a direct proxy for phenotype penetrance (positive = predicted
    genotype, negative = wildtype-like).

    Parameters
    ----------
    df_embryo_probs : pd.DataFrame
        Per-embryo predictions with signed_margin column
    df_penetrance : pd.DataFrame
        Penetrance metrics for ordering embryos
    group1, group2 : str
        Genotype labels
    output_path : str or None
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure, or None if insufficient data
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

    pivot_plot = pivot.reindex(row_order).reindex(columns=time_bins)
    data = pivot_plot.values.astype(float)
    mask = np.isnan(data)

    cmap_name = 'RdBu_r'
    if mpl_colormaps is not None:
        cmap_instance = mpl_colormaps[cmap_name].copy()
    else:
        cmap_instance = plt.get_cmap(cmap_name).copy()
    cmap_instance.set_bad(color='lightgray')
    data_masked = np.ma.masked_array(data, mask=mask)

    im = ax.imshow(data_masked, aspect='auto', cmap=cmap_instance, vmin=-0.5, vmax=0.5)

    ax.set_xticks(np.arange(len(time_bins)))
    ax.set_xticklabels(time_bins, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Developmental Time (hpf)', fontsize=13, fontweight='bold')

    row_labels = [str(eid)[:15] for eid in row_order]
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_ylabel('Embryo ID', fontsize=13, fontweight='bold')

    section_sizes = [len(ids) for _, ids in row_sections if ids]
    section_boundaries = np.cumsum(section_sizes)

    for idx, (label, ids) in enumerate(row_sections):
        if not ids:
            continue
        start_pos = section_boundaries[idx - 1] if idx > 0 else 0
        end_pos = section_boundaries[idx]
        center_pos = (start_pos + end_pos) / 2

        ax.text(len(time_bins) + 0.6, center_pos, label.split('_')[-1].upper(),
                fontsize=12, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='black', linewidth=2))

        if idx < len(row_sections) - 1:
            ax.axhline(end_pos - 0.5, color='black', linewidth=3, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Signed Margin\n(RED: predicted genotype, BLUE: wildtype-like)',
                   rotation=270, labelpad=25, fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        f'Embryo Signed-Margin Trajectories: {group1.split("_")[-1].upper()} vs {group2.split("_")[-1].upper()}',
        fontsize=16,
        fontweight='bold',
        pad=22
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig
