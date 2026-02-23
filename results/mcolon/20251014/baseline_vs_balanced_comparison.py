"""
Side-by-side comparison of Baseline vs Balanced (class_weight) methods
for embryo-level signed margin trajectories and heatmaps.

This script loads saved per-embryo prediction data and creates comparison
visualizations to assess how well class balancing corrects probability bias.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

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

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014"
data_dir = os.path.join(results_dir, "imbalance_methods", "data")
output_dir = os.path.join(results_dir, "imbalance_methods", "plots", "baseline_vs_balanced")
os.makedirs(output_dir, exist_ok=True)

print(f"Output directory: {output_dir}")

# Trajectory plotting configuration
MAX_TRAJECTORY_EMBRYOS = 30
TRAJECTORY_SELECTION_MODE = "shared"  # options: 'independent', 'shared'
SHARED_SELECTION_REFERENCE = "balanced"  # reference method when using shared selection

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_embryo_penetrance(df_embryo_probs, confidence_threshold=0.1):
    """Compute per-embryo penetrance metrics from prediction probabilities."""
    if df_embryo_probs.empty:
        return pd.DataFrame()

    penetrance_metrics = []

    for embryo_id, grp in df_embryo_probs.groupby('embryo_id'):
        grp = grp.sort_values('time_bin')

        mean_conf = grp['confidence'].mean()
        max_conf = grp['confidence'].max()
        n_bins = len(grp)
        mean_support_true = grp['support_true'].mean() if 'support_true' in grp.columns else np.nan
        min_support_true = grp['support_true'].min() if 'support_true' in grp.columns else np.nan
        mean_signed_margin = grp['signed_margin'].mean() if 'signed_margin' in grp.columns else np.nan
        min_signed_margin = grp['signed_margin'].min() if 'signed_margin' in grp.columns else np.nan

        correct = (grp['true_label'] == grp['predicted_label']).sum()
        temporal_consistency = correct / n_bins if n_bins > 0 else 0.0

        confident_bins = grp[grp['confidence'] > confidence_threshold]
        first_confident_time = confident_bins['time_bin'].min() if len(confident_bins) > 0 else np.nan

        true_label = grp['true_label'].iloc[0]

        penetrance_metrics.append({
            'embryo_id': embryo_id,
            'true_label': true_label,
            'mean_confidence': mean_conf,
            'mean_support_true': mean_support_true,
            'mean_signed_margin': mean_signed_margin,
            'temporal_consistency': temporal_consistency,
            'max_confidence': max_conf,
            'min_support_true': min_support_true,
            'min_signed_margin': min_signed_margin,
            'first_confident_time': first_confident_time,
            'n_time_bins': n_bins,
            'mean_pred_prob': grp['pred_proba'].mean()
        })

    return pd.DataFrame(penetrance_metrics)


def _select_embryos_by_penetrance(df_penetrance, genotype, max_embryos=None):
    """Return ordered embryo IDs for a genotype sorted by |mean_signed_margin|."""
    if df_penetrance is None or df_penetrance.empty:
        return []

    subset = df_penetrance[df_penetrance['true_label'] == genotype].copy()
    if subset.empty:
        return []

    subset = subset.assign(
        abs_margin=np.abs(subset.get('mean_signed_margin', np.nan))
    )
    subset = subset.sort_values(
        by=['abs_margin', 'mean_signed_margin'],
        ascending=[False, False]
    )
    embryo_ids = subset['embryo_id'].tolist()
    if max_embryos is not None:
        embryo_ids = embryo_ids[:max_embryos]
    return embryo_ids


def compute_shared_selection_map(genotypes, penetrance_map, max_embryos, reference_method):
    """
    Determine a shared embryo selection for all methods.

    Parameters
    ----------
    genotypes : list[str]
        Genotype labels being compared.
    penetrance_map : dict[str, pd.DataFrame]
        Mapping of method key to penetrance dataframe.
    max_embryos : int
        Maximum embryos to include per genotype.
    reference_method : str
        Which method to use as the initial ranking source.

    Returns
    -------
    tuple(dict, dict)
        shared_map: genotype -> ordered list of embryo IDs present in all methods.
        info_map: genotype -> diagnostics about selection process.
    """
    reference_key = reference_method.lower()
    available_keys = [k for k, df in penetrance_map.items() if df is not None and not df.empty]

    if reference_key not in available_keys:
        if not available_keys:
            return {}, {}
        reference_key = available_keys[0]

    shared_map = {}
    info_map = {}

    for genotype in genotypes:
        reference_df = penetrance_map.get(reference_key)
        if reference_df is None or reference_df.empty:
            shared_map[genotype] = []
            info_map[genotype] = {
                'reference_method': reference_key,
                'selected': [],
                'skipped_due_to_missing': [],
                'note': 'No penetrance data for reference method.'
            }
            continue

        ranked_ids = _select_embryos_by_penetrance(reference_df, genotype, max_embryos=None)
        if not ranked_ids:
            shared_map[genotype] = []
            info_map[genotype] = {
                'reference_method': reference_key,
                'selected': [],
                'skipped_due_to_missing': [],
                'note': 'Reference method had no embryos for genotype.'
            }
            continue

        selected = []
        skipped = []

        for embryo_id in ranked_ids:
            present_everywhere = True
            for method_key, pen_df in penetrance_map.items():
                if pen_df is None or pen_df.empty:
                    continue
                mask = (pen_df['true_label'] == genotype) & (pen_df['embryo_id'] == embryo_id)
                if not mask.any():
                    present_everywhere = False
                    break
            if present_everywhere:
                selected.append(embryo_id)
            else:
                skipped.append(embryo_id)
            if max_embryos is not None and len(selected) >= max_embryos:
                break

        shared_map[genotype] = selected
        note = None
        if max_embryos is not None and len(selected) < max_embryos:
            note = f"Only {len(selected)} embryos available across all methods (requested {max_embryos})."

        info_map[genotype] = {
            'reference_method': reference_key,
            'selected': selected,
            'skipped_due_to_missing': skipped,
            'note': note
        }

    return shared_map, info_map


def plot_signed_margin_trajectories_comparison(
    df_baseline, df_balanced,
    penetrance_baseline, penetrance_balanced,
    group1, group2,
    output_path=None,
    max_embryos=30,
    shared_selection_map=None,
    selection_reference_label=None
):
    """
    Side-by-side comparison of signed margin trajectories for baseline vs balanced.

    Shows how class balancing affects individual embryo prediction trajectories.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    methods = [
        ('Baseline', df_baseline, penetrance_baseline),
        ('Balanced (class_weight)', df_balanced, penetrance_balanced)
    ]

    genotypes = [group1, group2]

    for method_idx, (method_name, df_probs, df_pen) in enumerate(methods):
        if df_probs is None or df_probs.empty or df_pen is None or df_pen.empty:
            print(f"  Skipping {method_name}: no data")
            continue

        for geno_idx, genotype in enumerate(genotypes):
            ax = axes[geno_idx, method_idx]

            # Get embryos for this genotype
            genotype_penetrance = df_pen[df_pen['true_label'] == genotype].copy()
            if genotype_penetrance.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{method_name}\n{genotype} (n=0)',
                           fontsize=13, fontweight='bold')
                continue

            # Sort by absolute margin and select top embryos
            genotype_penetrance = genotype_penetrance.assign(
                abs_margin=np.abs(genotype_penetrance.get('mean_signed_margin', np.nan))
            )
            genotype_penetrance = genotype_penetrance.sort_values(
                by=['abs_margin', 'mean_signed_margin'], ascending=[False, False]
            )

            shared_list = None
            if shared_selection_map is not None:
                shared_list = shared_selection_map.get(genotype, [])

            if shared_list is not None:
                available_embryos = [eid for eid in shared_list if eid in genotype_penetrance['embryo_id'].values]
                if available_embryos:
                    genotype_penetrance = (
                        genotype_penetrance
                        .set_index('embryo_id')
                        .loc[available_embryos]
                        .reset_index()
                    )
                else:
                    genotype_penetrance = genotype_penetrance.iloc[0:0]
                top_embryos = available_embryos
            else:
                genotype_penetrance = genotype_penetrance.head(max_embryos)
                top_embryos = genotype_penetrance['embryo_id'].values

            # Color mapping based on mean signed margin
            norm = Normalize(vmin=-0.5, vmax=0.5)
            cmap = plt.cm.RdBu_r
            alphas = np.linspace(0.35, 0.9, len(top_embryos)) if len(top_embryos) > 0 else []
            highlight_id = genotype_penetrance.iloc[0]['embryo_id'] if len(genotype_penetrance) > 0 else None
            penetrance_lookup = genotype_penetrance.set_index('embryo_id')

            # Plot trajectories
            for alpha, embryo_id in zip(alphas, top_embryos):
                embryo_curve = df_probs[df_probs['embryo_id'] == embryo_id].sort_values('time_bin')
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

            # Add decision boundary
            ax.axhline(0.0, color='red', linestyle='--', linewidth=1.3, alpha=0.7,
                      label='Decision boundary')
            ax.set_xlabel('Time (hpf)', fontsize=11)
            ax.set_ylabel('Signed Margin vs 0.5', fontsize=11)
            ax.set_ylim([-0.5, 0.5])
            ax.grid(alpha=0.3)

            # Title showing method and genotype
            geno_short = genotype.split('_')[-1]
            selection_suffix = ''
            if shared_selection_map is not None:
                selection_suffix = ' [shared]'
            ax.set_title(f'{method_name}\n{geno_short} (n={len(top_embryos)}){selection_suffix}',
                        fontsize=12, fontweight='bold')

            if len(top_embryos) > 0:
                ax.legend(loc='upper left', fontsize=8)

    suptitle = f'Baseline vs Balanced: Signed Margin Trajectories\n{group1.split("_")[-1]} vs {group2.split("_")[-1]}'
    if selection_reference_label:
        suptitle += f'\n{selection_reference_label}'

    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_signed_margin_heatmap_comparison(
    df_baseline, df_balanced,
    penetrance_baseline, penetrance_balanced,
    group1, group2,
    output_path=None
):
    """
    Side-by-side heatmap comparison of baseline vs balanced methods.

    Shows how class balancing affects the signed margin patterns across embryos.
    """
    if df_baseline.empty or df_balanced.empty:
        print("  Skipping heatmap comparison: missing data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    methods_data = [
        ('Baseline', df_baseline, penetrance_baseline),
        ('Balanced (class_weight)', df_balanced, penetrance_balanced)
    ]

    for ax_idx, (method_name, df_probs, df_pen) in enumerate(methods_data):
        ax = axes[ax_idx]

        if df_probs is None or df_probs.empty or df_pen is None or df_pen.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(method_name, fontsize=14, fontweight='bold')
            continue

        # Build embryo to label mapping
        embryo_to_label = df_pen.set_index('embryo_id')['true_label'].to_dict()
        label_order = [label for label in [group1, group2] if label in df_probs['true_label'].unique()]
        if not label_order:
            label_order = list(df_probs['true_label'].unique())

        time_bins = sorted(df_probs['time_bin'].unique())

        # Create pivot table
        pivot = df_probs.pivot_table(
            index='embryo_id',
            columns='time_bin',
            values='signed_margin',
            aggfunc='mean'
        )

        # Order embryos by genotype and penetrance
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
                ranking_metric = 'mean_signed_margin' if 'mean_signed_margin' in df_pen.columns else 'mean_confidence'
                ordered_index = list(
                    df_pen[df_pen['embryo_id'].isin(subset.index)]
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
            ax.text(0.5, 0.5, 'No embryos to plot', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue

        # Plot heatmap
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

        # X-axis (time bins)
        ax.set_xticks(np.arange(len(time_bins)))
        ax.set_xticklabels(time_bins, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Time (hpf)', fontsize=12, fontweight='bold')

        # Y-axis (embryo IDs)
        row_labels = [str(eid)[:12] for eid in row_order]
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_ylabel('Embryo ID', fontsize=12, fontweight='bold')

        # Add genotype section boundaries
        section_sizes = [len(ids) for _, ids in row_sections if ids]
        section_boundaries = np.cumsum(section_sizes)

        for idx, (label, ids) in enumerate(row_sections):
            if not ids:
                continue
            start_pos = section_boundaries[idx - 1] if idx > 0 else 0
            end_pos = section_boundaries[idx]
            center_pos = (start_pos + end_pos) / 2

            # Add genotype label on the right
            ax.text(len(time_bins) + 0.6, center_pos, label.split('_')[-1].upper(),
                   fontsize=11, fontweight='bold', va='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='black', linewidth=1.5))

            # Add dividing line
            if idx < len(row_sections) - 1:
                ax.axhline(end_pos - 0.5, color='black', linewidth=2.5, alpha=0.85)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Signed Margin', rotation=270, labelpad=20,
                      fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)

        # Title
        ax.set_title(method_name, fontsize=14, fontweight='bold', pad=15)

    fig.suptitle(
        f'Baseline vs Balanced: Embryo Signed Margin Heatmaps\n{group1.split("_")[-1]} vs {group2.split("_")[-1]}',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


# ============================================================================
# MAIN COMPARISON ANALYSIS
# ============================================================================

COMPARISONS = {
    "b9d2": [
        ("b9d2_wildtype", "b9d2_heterozygous"),
        ("b9d2_wildtype", "b9d2_homozygous"),
        ("b9d2_heterozygous", "b9d2_homozygous")
    ],
    "cep290": [
        ("cep290_wildtype", "cep290_heterozygous"),
        ("cep290_wildtype", "cep290_homozygous"),
        ("cep290_heterozygous", "cep290_homozygous")
    ],
    "tmem67": [
        ("tmem67_wildtype", "tmem67_heterozygote"),
        ("tmem67_wildtype", "tmem67_homozygous"),
        ("tmem67_heterozygote", "tmem67_homozygous")
    ]
}

print("\n" + "="*80)
print("BASELINE VS BALANCED COMPARISON")
print("="*80)

for gene, comparisons in COMPARISONS.items():
    print(f"\n{'='*80}")
    print(f"GENE: {gene.upper()}")
    print(f"{'='*80}")

    gene_data_dir = os.path.join(data_dir, gene)
    gene_output_dir = os.path.join(output_dir, gene)
    os.makedirs(gene_output_dir, exist_ok=True)

    for group1, group2 in comparisons:
        print(f"\n  Comparison: {group1} vs {group2}")

        # Generate safe filename
        safe_name = f"{group1.split('_')[-1]}_vs_{group2.split('_')[-1]}"

        # Load baseline data
        baseline_path = os.path.join(gene_data_dir, f"embryo_probs_baseline_{safe_name}.csv")
        balanced_path = os.path.join(gene_data_dir, f"embryo_probs_class_weight_{safe_name}.csv")

        if not os.path.exists(baseline_path):
            print(f"    Missing baseline data: {baseline_path}")
            continue
        if not os.path.exists(balanced_path):
            print(f"    Missing balanced data: {balanced_path}")
            continue

        # Load data
        df_baseline = pd.read_csv(baseline_path)
        df_balanced = pd.read_csv(balanced_path)

        print(f"    Loaded baseline: {len(df_baseline)} predictions, {df_baseline['embryo_id'].nunique()} embryos")
        print(f"    Loaded balanced: {len(df_balanced)} predictions, {df_balanced['embryo_id'].nunique()} embryos")

        # Compute penetrance metrics
        penetrance_baseline = compute_embryo_penetrance(df_baseline)
        penetrance_balanced = compute_embryo_penetrance(df_balanced)

        # Determine shared selection if requested
        shared_selection_map = None
        selection_label = None
        if TRAJECTORY_SELECTION_MODE.lower() == "shared":
            penetrance_map = {
                'baseline': penetrance_baseline,
                'balanced': penetrance_balanced
            }
            shared_selection_map, selection_info = compute_shared_selection_map(
                [group1, group2],
                penetrance_map,
                max_embryos=MAX_TRAJECTORY_EMBRYOS,
                reference_method=SHARED_SELECTION_REFERENCE
            )
            ref_name = SHARED_SELECTION_REFERENCE.replace('_', ' ').title()
            selection_label = f"Shared embryo selection (reference: {ref_name})"
            for genotype in [group1, group2]:
                selected = shared_selection_map.get(genotype, [])
                short_label = genotype.split('_')[-1]
                print(f"    Shared embryos for {short_label}: {len(selected)}")
                if selected:
                    preview = ', '.join(selected[:5])
                    if len(selected) > 5:
                        preview += ", ..."
                    print(f"      First embryos: {preview}")
                info = selection_info.get(genotype, {})
                skipped = info.get('skipped_due_to_missing') or []
                if skipped:
                    print(f"      Skipped due to missing coverage: {len(skipped)}")
                note = info.get('note')
                if note:
                    print(f"      Note: {note}")

        # Generate trajectory comparison
        print(f"    Generating trajectory comparison...")
        plot_signed_margin_trajectories_comparison(
            df_baseline, df_balanced,
            penetrance_baseline, penetrance_balanced,
            group1, group2,
            output_path=os.path.join(gene_output_dir, f'trajectories_baseline_vs_balanced_{safe_name}.png'),
            max_embryos=MAX_TRAJECTORY_EMBRYOS,
            shared_selection_map=shared_selection_map,
            selection_reference_label=selection_label
        )

        # Generate heatmap comparison
        print(f"    Generating heatmap comparison...")
        plot_signed_margin_heatmap_comparison(
            df_baseline, df_balanced,
            penetrance_baseline, penetrance_balanced,
            group1, group2,
            output_path=os.path.join(gene_output_dir, f'heatmap_baseline_vs_balanced_{safe_name}.png')
        )

        # Print summary statistics
        print(f"\n    Summary Statistics:")
        print(f"    {'Method':<20} {'Mean |Margin|':<15} {'Mean Support':<15}")
        print(f"    {'-'*50}")

        for method_name, df_pen in [('Baseline', penetrance_baseline),
                                     ('Balanced', penetrance_balanced)]:
            if df_pen is not None and not df_pen.empty:
                mean_abs_margin = df_pen['mean_signed_margin'].abs().mean()
                mean_support = df_pen['mean_support_true'].mean()
                print(f"    {method_name:<20} {mean_abs_margin:<15.3f} {mean_support:<15.3f}")

print("\n" + "="*80)
print("BASELINE VS BALANCED COMPARISON COMPLETE")
print("="*80)
print(f"\nPlots saved to: {output_dir}")
