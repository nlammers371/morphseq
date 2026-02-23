"""
Replot probability shift analysis by loading directly from saved CSV files.

This bypasses any in-memory issues and ensures all methods with saved data are plotted.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014"
data_dir_base = os.path.join(results_dir, "imbalance_methods", "data")
plot_dir_base = os.path.join(results_dir, "imbalance_methods", "plots")
replot_dir = os.path.join(plot_dir_base, "replotted")
os.makedirs(replot_dir, exist_ok=True)

print(f"Replotting probability shift analysis")
print(f"Output directory: {replot_dir}\n")

# Methods to load
METHODS = [
    'baseline',
    'class_weight',
    'embryo_weight',
    'combined_weight',
    'calibrated_class_weight',
    'calibrated_combined_weight',
    'balanced_bootstrap'
]

# ============================================================================
# PLOTTING FUNCTION (modified to load from CSVs)
# ============================================================================

def plot_probability_shift_from_csvs(gene, comparison, group1, group2, output_path=None):
    """
    Plot probability shift analysis by loading data directly from CSV files.
    """
    data_dir = os.path.join(data_dir_base, gene)

    # Load data for each method
    results_by_method = {}

    for method_name in METHODS:
        csv_path = os.path.join(data_dir, f"embryo_probs_{method_name}_{comparison}.csv")

        if not os.path.exists(csv_path):
            print(f"  ⚠ Skipping {method_name}: file not found")
            continue

        try:
            df = pd.read_csv(csv_path)

            if df.empty:
                print(f"  ⚠ Skipping {method_name}: empty dataframe")
                continue

            if 'true_label' not in df.columns or 'pred_proba' not in df.columns or 'time_bin' not in df.columns:
                print(f"  ⚠ Skipping {method_name}: missing required columns")
                continue

            results_by_method[method_name] = df
            print(f"  ✓ Loaded {method_name}: {len(df)} rows")

        except Exception as e:
            print(f"  ✗ Error loading {method_name}: {e}")
            continue

    if not results_by_method:
        print(f"  No data loaded - cannot generate plot")
        return None

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_method)))

    ax1 = axes[0]
    ax2 = axes[1]

    # Determine classes
    first_df = list(results_by_method.values())[0]
    classes = sorted(first_df['true_label'].unique())

    if len(classes) != 2:
        print(f"  Error: Expected 2 classes, got {len(classes)}")
        return None

    print(f"\n  Plotting {len(results_by_method)} methods...")

    for method_idx, (method_name, df_probs) in enumerate(results_by_method.items()):
        print(f"    {method_name}:")

        # For each class, compute mean predicted probability over time
        for cls_idx, cls in enumerate(classes):
            df_cls = df_probs[df_probs['true_label'] == cls]

            if df_cls.empty:
                print(f"      ⚠ No data for class {cls}")
                continue

            mean_probs = df_cls.groupby('time_bin')['pred_proba'].mean()

            if len(mean_probs) == 0:
                print(f"      ⚠ No time bins for class {cls}")
                continue

            ax = ax1 if cls_idx == 0 else ax2
            ax.plot(mean_probs.index, mean_probs.values,
                   label=method_name, color=colors[method_idx],
                   linewidth=2, marker='o', markersize=4)

            print(f"      ✓ Plotted class {cls}: {len(mean_probs)} time bins")

    # Formatting
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision boundary')
    ax1.set_xlabel('Time Bin (hpf)', fontsize=12)
    ax1.set_ylabel('Mean Predicted Probability', fontsize=12)
    ax1.set_title(f'True Class: {classes[0]}', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(alpha=0.3)

    ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision boundary')
    ax2.set_xlabel('Time Bin (hpf)', fontsize=12)
    ax2.set_ylabel('Mean Predicted Probability', fontsize=12)
    ax2.set_title(f'True Class: {classes[1]}', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(alpha=0.3)

    fig.suptitle(f'Probability Shift Analysis (Replotted from CSVs)\n{group1} vs {group2}',
                fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Saved: {output_path}")

    return fig


# ============================================================================
# REPLOT ALL COMPARISONS
# ============================================================================

COMPARISONS = {
    "cep290": [
        ("cep290_wildtype", "cep290_heterozygous", "wildtype_vs_heterozygous"),
        ("cep290_wildtype", "cep290_homozygous", "wildtype_vs_homozygous"),
        ("cep290_heterozygous", "cep290_homozygous", "heterozygous_vs_homozygous")
    ],
    "b9d2": [
        ("b9d2_wildtype", "b9d2_heterozygous", "wildtype_vs_heterozygous"),
        ("b9d2_wildtype", "b9d2_homozygous", "wildtype_vs_homozygous"),
        ("b9d2_heterozygous", "b9d2_homozygous", "heterozygous_vs_homozygous")
    ],
    "tmem67": [
        ("tmem67_wildtype", "tmem67_heterozygote", "wildtype_vs_heterozygote"),
        ("tmem67_wildtype", "tmem67_homozygous", "wildtype_vs_homozygous"),
        ("tmem67_heterozygote", "tmem67_homozygous", "heterozygote_vs_homozygous")
    ]
}

print("="*80)
print("REPLOTTING PROBABILITY SHIFT ANALYSIS FROM CSVs")
print("="*80)

for gene, comparisons in COMPARISONS.items():
    print(f"\n{gene.upper()}:")

    gene_replot_dir = os.path.join(replot_dir, gene)
    os.makedirs(gene_replot_dir, exist_ok=True)

    for group1, group2, comparison in comparisons:
        print(f"\n  {comparison}:")

        output_path = os.path.join(gene_replot_dir, f'probability_shift_replotted_{comparison}.png')

        plot_probability_shift_from_csvs(gene, comparison, group1, group2, output_path)

print("\n" + "="*80)
print("REPLOTTING COMPLETE")
print("="*80)
print(f"\nReplotted figures saved to: {replot_dir}")
