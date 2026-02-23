"""
Diagnostic script to investigate why baseline and balanced methods
appear to have different numbers of data points in trajectory plots.

This script performs detailed comparisons at each processing step:
1. Raw embryo prediction data
2. Penetrance metrics computation
3. Embryo selection for plotting
4. Final plotted trajectories
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014"
data_dir = os.path.join(results_dir, "imbalance_methods", "data")
output_dir = os.path.join(results_dir, "imbalance_methods", "diagnostics")
os.makedirs(output_dir, exist_ok=True)

print(f"Diagnostic output directory: {output_dir}\n")

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def compute_embryo_penetrance(df_embryo_probs, confidence_threshold=0.1):
    """Compute per-embryo penetrance metrics."""
    if df_embryo_probs.empty:
        return pd.DataFrame()

    penetrance_metrics = []

    for embryo_id, grp in df_embryo_probs.groupby('embryo_id'):
        grp = grp.sort_values('time_bin')

        mean_conf = grp['confidence'].mean()
        max_conf = grp['confidence'].max()
        n_bins = len(grp)
        mean_support_true = grp['support_true'].mean() if 'support_true' in grp.columns else np.nan
        mean_signed_margin = grp['signed_margin'].mean() if 'signed_margin' in grp.columns else np.nan

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
            'first_confident_time': first_confident_time,
            'n_time_bins': n_bins,
            'mean_pred_prob': grp['pred_proba'].mean()
        })

    return pd.DataFrame(penetrance_metrics)


def diagnose_comparison(gene, group1, group2, max_embryos=30):
    """
    Detailed diagnostic comparison of baseline vs balanced for one pairwise comparison.
    """
    print("="*80)
    print(f"DIAGNOSTIC: {gene.upper()} - {group1} vs {group2}")
    print("="*80)

    safe_name = f"{group1.split('_')[-1]}_vs_{group2.split('_')[-1]}"
    gene_data_dir = os.path.join(data_dir, gene)

    # Load data files
    baseline_path = os.path.join(gene_data_dir, f"embryo_probs_baseline_{safe_name}.csv")
    balanced_path = os.path.join(gene_data_dir, f"embryo_probs_class_weight_{safe_name}.csv")

    if not os.path.exists(baseline_path) or not os.path.exists(balanced_path):
        print(f"  MISSING FILES - skipping")
        return None

    df_baseline = pd.read_csv(baseline_path)
    df_balanced = pd.read_csv(balanced_path)

    # ========================================================================
    # STEP 1: Raw data comparison
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: RAW DATA COMPARISON")
    print("-"*80)

    print(f"\nBaseline:")
    print(f"  Total rows: {len(df_baseline)}")
    print(f"  Unique embryos: {df_baseline['embryo_id'].nunique()}")
    print(f"  Unique time bins: {df_baseline['time_bin'].nunique()}")
    print(f"  Time bin range: {df_baseline['time_bin'].min()} - {df_baseline['time_bin'].max()}")

    print(f"\nBalanced:")
    print(f"  Total rows: {len(df_balanced)}")
    print(f"  Unique embryos: {df_balanced['embryo_id'].nunique()}")
    print(f"  Unique time bins: {df_balanced['time_bin'].nunique()}")
    print(f"  Time bin range: {df_balanced['time_bin'].min()} - {df_balanced['time_bin'].max()}")

    # Check for missing values
    print(f"\nMissing values:")
    print(f"  Baseline - signed_margin NaNs: {df_baseline['signed_margin'].isna().sum()}")
    print(f"  Balanced - signed_margin NaNs: {df_balanced['signed_margin'].isna().sum()}")

    # Check embryo overlap
    baseline_embryos = set(df_baseline['embryo_id'].unique())
    balanced_embryos = set(df_balanced['embryo_id'].unique())
    embryo_overlap = baseline_embryos & balanced_embryos
    embryo_baseline_only = baseline_embryos - balanced_embryos
    embryo_balanced_only = balanced_embryos - baseline_embryos

    print(f"\nEmbryo overlap:")
    print(f"  Both methods: {len(embryo_overlap)}")
    print(f"  Baseline only: {len(embryo_baseline_only)}")
    if embryo_baseline_only:
        print(f"    IDs: {list(embryo_baseline_only)[:5]}...")
    print(f"  Balanced only: {len(embryo_balanced_only)}")
    if embryo_balanced_only:
        print(f"    IDs: {list(embryo_balanced_only)[:5]}...")

    # ========================================================================
    # STEP 2: Per-genotype breakdown
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: PER-GENOTYPE BREAKDOWN")
    print("-"*80)

    for genotype in [group1, group2]:
        print(f"\n{genotype}:")

        baseline_geno = df_baseline[df_baseline['true_label'] == genotype]
        balanced_geno = df_balanced[df_balanced['true_label'] == genotype]

        print(f"  Baseline: {baseline_geno['embryo_id'].nunique()} embryos, {len(baseline_geno)} predictions")
        print(f"  Balanced: {balanced_geno['embryo_id'].nunique()} embryos, {len(balanced_geno)} predictions")

        # Check predictions per embryo
        baseline_preds_per_embryo = baseline_geno.groupby('embryo_id').size()
        balanced_preds_per_embryo = balanced_geno.groupby('embryo_id').size()

        print(f"  Baseline - predictions/embryo: mean={baseline_preds_per_embryo.mean():.1f}, "
              f"min={baseline_preds_per_embryo.min()}, max={baseline_preds_per_embryo.max()}")
        print(f"  Balanced - predictions/embryo: mean={balanced_preds_per_embryo.mean():.1f}, "
              f"min={balanced_preds_per_embryo.min()}, max={balanced_preds_per_embryo.max()}")

    # ========================================================================
    # STEP 3: Signed margin distributions
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: SIGNED MARGIN DISTRIBUTIONS")
    print("-"*80)

    for genotype in [group1, group2]:
        print(f"\n{genotype}:")

        baseline_geno = df_baseline[df_baseline['true_label'] == genotype]
        balanced_geno = df_balanced[df_balanced['true_label'] == genotype]

        baseline_margins = baseline_geno['signed_margin'].dropna()
        balanced_margins = balanced_geno['signed_margin'].dropna()

        print(f"  Baseline signed margin:")
        print(f"    Mean: {baseline_margins.mean():.3f}, Std: {baseline_margins.std():.3f}")
        print(f"    Range: [{baseline_margins.min():.3f}, {baseline_margins.max():.3f}]")
        print(f"    Mean |margin|: {baseline_margins.abs().mean():.3f}")

        print(f"  Balanced signed margin:")
        print(f"    Mean: {balanced_margins.mean():.3f}, Std: {balanced_margins.std():.3f}")
        print(f"    Range: [{balanced_margins.min():.3f}, {balanced_margins.max():.3f}]")
        print(f"    Mean |margin|: {balanced_margins.abs().mean():.3f}")

    # ========================================================================
    # STEP 4: Penetrance metrics
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: PENETRANCE METRICS COMPUTATION")
    print("-"*80)

    penetrance_baseline = compute_embryo_penetrance(df_baseline)
    penetrance_balanced = compute_embryo_penetrance(df_balanced)

    print(f"\nPenetrance dataframes:")
    print(f"  Baseline: {len(penetrance_baseline)} embryos")
    print(f"  Balanced: {len(penetrance_balanced)} embryos")

    # ========================================================================
    # STEP 5: Embryo selection for plotting
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 5: EMBRYO SELECTION FOR PLOTTING")
    print("-"*80)

    for genotype in [group1, group2]:
        print(f"\n{genotype}:")

        # Baseline selection
        baseline_pen = penetrance_baseline[penetrance_baseline['true_label'] == genotype].copy()
        baseline_pen['abs_margin'] = baseline_pen['mean_signed_margin'].abs()
        baseline_pen_sorted = baseline_pen.sort_values(
            by=['abs_margin', 'mean_signed_margin'], ascending=[False, False]
        )
        baseline_selected = baseline_pen_sorted.head(max_embryos)

        print(f"  Baseline:")
        print(f"    Total embryos: {len(baseline_pen)}")
        print(f"    Selected (top {max_embryos}): {len(baseline_selected)}")
        if len(baseline_selected) > 0:
            print(f"    Mean |margin| of selected: {baseline_selected['abs_margin'].mean():.3f}")
            print(f"    Range of |margin|: [{baseline_selected['abs_margin'].min():.3f}, "
                  f"{baseline_selected['abs_margin'].max():.3f}]")

        # Balanced selection
        balanced_pen = penetrance_balanced[penetrance_balanced['true_label'] == genotype].copy()
        balanced_pen['abs_margin'] = balanced_pen['mean_signed_margin'].abs()
        balanced_pen_sorted = balanced_pen.sort_values(
            by=['abs_margin', 'mean_signed_margin'], ascending=[False, False]
        )
        balanced_selected = balanced_pen_sorted.head(max_embryos)

        print(f"  Balanced:")
        print(f"    Total embryos: {len(balanced_pen)}")
        print(f"    Selected (top {max_embryos}): {len(balanced_selected)}")
        if len(balanced_selected) > 0:
            print(f"    Mean |margin| of selected: {balanced_selected['abs_margin'].mean():.3f}")
            print(f"    Range of |margin|: [{balanced_selected['abs_margin'].min():.3f}, "
                  f"{balanced_selected['abs_margin'].max():.3f}]")

        # Check overlap in selected embryos
        baseline_selected_ids = set(baseline_selected['embryo_id'])
        balanced_selected_ids = set(balanced_selected['embryo_id'])
        overlap_ids = baseline_selected_ids & balanced_selected_ids

        print(f"  Overlap in selected embryos: {len(overlap_ids)}/{max_embryos}")
        print(f"  Baseline-only: {len(baseline_selected_ids - balanced_selected_ids)}")
        print(f"  Balanced-only: {len(balanced_selected_ids - baseline_selected_ids)}")

    # ========================================================================
    # STEP 6: Final plotted data points
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 6: FINAL PLOTTED DATA POINTS")
    print("-"*80)

    for genotype in [group1, group2]:
        print(f"\n{genotype}:")

        # Get selected embryo IDs
        baseline_pen = penetrance_baseline[penetrance_baseline['true_label'] == genotype].copy()
        baseline_pen['abs_margin'] = baseline_pen['mean_signed_margin'].abs()
        baseline_selected_ids = baseline_pen.sort_values(
            by=['abs_margin', 'mean_signed_margin'], ascending=[False, False]
        ).head(max_embryos)['embryo_id'].values

        balanced_pen = penetrance_balanced[penetrance_balanced['true_label'] == genotype].copy()
        balanced_pen['abs_margin'] = balanced_pen['mean_signed_margin'].abs()
        balanced_selected_ids = balanced_pen.sort_values(
            by=['abs_margin', 'mean_signed_margin'], ascending=[False, False]
        ).head(max_embryos)['embryo_id'].values

        # Count actual data points that would be plotted
        baseline_plot_data = df_baseline[
            (df_baseline['true_label'] == genotype) &
            (df_baseline['embryo_id'].isin(baseline_selected_ids))
        ]
        balanced_plot_data = df_balanced[
            (df_balanced['true_label'] == genotype) &
            (df_balanced['embryo_id'].isin(balanced_selected_ids))
        ]

        print(f"  Baseline:")
        print(f"    Embryos selected: {len(baseline_selected_ids)}")
        print(f"    Total data points to plot: {len(baseline_plot_data)}")
        print(f"    Points per embryo: {len(baseline_plot_data) / len(baseline_selected_ids):.1f}")

        print(f"  Balanced:")
        print(f"    Embryos selected: {len(balanced_selected_ids)}")
        print(f"    Total data points to plot: {len(balanced_plot_data)}")
        print(f"    Points per embryo: {len(balanced_plot_data) / len(balanced_selected_ids):.1f}")

        # Check for time bin differences per embryo
        baseline_bins_per_embryo = baseline_plot_data.groupby('embryo_id')['time_bin'].nunique()
        balanced_bins_per_embryo = balanced_plot_data.groupby('embryo_id')['time_bin'].nunique()

        print(f"  Time bins per embryo:")
        print(f"    Baseline: mean={baseline_bins_per_embryo.mean():.1f}, "
              f"min={baseline_bins_per_embryo.min()}, max={baseline_bins_per_embryo.max()}")
        print(f"    Balanced: mean={balanced_bins_per_embryo.mean():.1f}, "
              f"min={balanced_bins_per_embryo.min()}, max={balanced_bins_per_embryo.max()}")

    # ========================================================================
    # STEP 7: Create visual comparison
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 7: CREATING VISUAL DIAGNOSTIC PLOTS")
    print("-"*80)

    # Distribution comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for geno_idx, genotype in enumerate([group1, group2]):
        baseline_geno = df_baseline[df_baseline['true_label'] == genotype]['signed_margin'].dropna()
        balanced_geno = df_balanced[df_balanced['true_label'] == genotype]['signed_margin'].dropna()

        # Histogram comparison
        ax = axes[geno_idx, 0]
        ax.hist(baseline_geno, bins=30, alpha=0.6, label='Baseline', color='steelblue', density=True)
        ax.hist(balanced_geno, bins=30, alpha=0.6, label='Balanced', color='coral', density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Signed Margin', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{genotype.split("_")[-1]} - Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Box plot comparison
        ax = axes[geno_idx, 1]
        data_to_plot = [baseline_geno, balanced_geno]
        bp = ax.boxplot(data_to_plot, labels=['Baseline', 'Balanced'], patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_ylabel('Signed Margin', fontsize=11)
        ax.set_title(f'{genotype.split("_")[-1]} - Box Plot', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

    fig.suptitle(f'Signed Margin Distribution Comparison\n{gene.upper()}: {safe_name}',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'distribution_comparison_{gene}_{safe_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved distribution plot: {plot_path}")
    plt.close()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    summary = {
        'gene': gene,
        'comparison': safe_name,
        'baseline_total_predictions': len(df_baseline),
        'balanced_total_predictions': len(df_balanced),
        'baseline_embryos': df_baseline['embryo_id'].nunique(),
        'balanced_embryos': df_balanced['embryo_id'].nunique(),
        'embryo_overlap': len(embryo_overlap),
        'baseline_mean_abs_margin': df_baseline['signed_margin'].abs().mean(),
        'balanced_mean_abs_margin': df_balanced['signed_margin'].abs().mean()
    }

    return summary


# ============================================================================
# RUN DIAGNOSTICS
# ============================================================================

COMPARISONS = {
    "cep290": [
        ("cep290_wildtype", "cep290_heterozygous"),
        ("cep290_wildtype", "cep290_homozygous"),
        ("cep290_heterozygous", "cep290_homozygous")
    ],
    "b9d2": [
        ("b9d2_wildtype", "b9d2_heterozygous"),
        ("b9d2_wildtype", "b9d2_homozygous"),
        ("b9d2_heterozygous", "b9d2_homozygous")
    ],
    "tmem67": [
        ("tmem67_wildtype", "tmem67_heterozygote"),
        ("tmem67_wildtype", "tmem67_homozygous"),
        ("tmem67_heterozygote", "tmem67_homozygous")
    ]
}

print("\n" + "="*80)
print("DATA DIFFERENCE INVESTIGATION")
print("="*80)

all_summaries = []

for gene, comparisons in COMPARISONS.items():
    for group1, group2 in comparisons:
        summary = diagnose_comparison(gene, group1, group2, max_embryos=30)
        if summary:
            all_summaries.append(summary)
        print("\n")

# Save summary table
if all_summaries:
    df_summary = pd.DataFrame(all_summaries)
    summary_path = os.path.join(output_dir, 'data_differences_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Summary table saved: {summary_path}")
    print(f"{'='*80}")

    # Print summary table
    print("\nQUICK REFERENCE TABLE:")
    print(df_summary.to_string(index=False))

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print(f"\nDiagnostic outputs saved to: {output_dir}")
