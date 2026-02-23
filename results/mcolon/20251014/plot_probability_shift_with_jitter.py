"""
Plot probability shift analysis with vertical jitter for overlapping lines.

Adds small vertical offsets to methods that produce nearly identical predictions,
making all lines visible even when they overlap.
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
jitter_dir = os.path.join(plot_dir_base, "with_jitter")
os.makedirs(jitter_dir, exist_ok=True)

print(f"Plotting with jitter for overlapping lines")
print(f"Output directory: {jitter_dir}\n")

METHODS = [
    'baseline',
    'class_weight',
    'embryo_weight',
    'combined_weight',
    'calibrated_class_weight',
    'calibrated_combined_weight',
    'balanced_bootstrap'
]

# Jitter amounts (vertical offsets in probability units)
# These will be added to the y-values to separate overlapping lines
JITTER_OFFSETS = {
    'baseline': 0.005,           # Slight offset up
    'class_weight': -0.005,      # Slight offset down
    'embryo_weight': 0.010,      # Larger offset up
    'combined_weight': -0.010,   # Larger offset down
    'calibrated_class_weight': 0.015,
    'calibrated_combined_weight': -0.015,
    'balanced_bootstrap': 0.0    # No offset (reference)
}

# Distinct visual styles
STYLES = {
    'baseline': {'color': 'red', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'o', 'markersize': 6, 'alpha': 0.9},
    'class_weight': {'color': 'blue', 'linestyle': '--', 'linewidth': 2.5, 'marker': 's', 'markersize': 6, 'alpha': 0.9},
    'embryo_weight': {'color': 'green', 'linestyle': '-.', 'linewidth': 2, 'marker': '^', 'markersize': 5, 'alpha': 0.8},
    'combined_weight': {'color': 'purple', 'linestyle': ':', 'linewidth': 2, 'marker': 'v', 'markersize': 5, 'alpha': 0.8},
    'calibrated_class_weight': {'color': 'orange', 'linestyle': '-', 'linewidth': 1.5, 'marker': 'd', 'markersize': 4, 'alpha': 0.7},
    'calibrated_combined_weight': {'color': 'brown', 'linestyle': '--', 'linewidth': 1.5, 'marker': 'p', 'markersize': 4, 'alpha': 0.7},
    'balanced_bootstrap': {'color': 'cyan', 'linestyle': '-.', 'linewidth': 2, 'marker': '*', 'markersize': 6, 'alpha': 0.8}
}

# ============================================================================
# PLOTTING FUNCTION WITH JITTER
# ============================================================================

def plot_probability_shift_with_jitter(gene, comparison, group1, group2, output_path=None):
    """
    Plot probability shift analysis with vertical jitter to separate overlapping lines.
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
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    ax2 = axes[1]

    # Determine classes
    first_df = list(results_by_method.values())[0]
    classes = sorted(first_df['true_label'].unique())

    if len(classes) != 2:
        print(f"  Error: Expected 2 classes, got {len(classes)}")
        return None

    print(f"\n  Plotting {len(results_by_method)} methods with jitter...")

    for method_name, df_probs in results_by_method.items():
        style = STYLES.get(method_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5})
        jitter = JITTER_OFFSETS.get(method_name, 0.0)

        print(f"    {method_name}: jitter={jitter:+.3f}, color={style['color']}, style={style['linestyle']}")

        # For each class, compute mean predicted probability over time
        for cls_idx, cls in enumerate(classes):
            df_cls = df_probs[df_probs['true_label'] == cls]

            if df_cls.empty:
                continue

            mean_probs = df_cls.groupby('time_bin')['pred_proba'].mean()

            if len(mean_probs) == 0:
                continue

            # Apply jitter (vertical offset)
            mean_probs_jittered = mean_probs + jitter

            ax = ax1 if cls_idx == 0 else ax2
            ax.plot(mean_probs.index, mean_probs_jittered.values,
                   label=f"{method_name} (offset={jitter:+.3f})",
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   marker=style['marker'],
                   markersize=style['markersize'],
                   alpha=style['alpha'],
                   zorder=10 if method_name in ['baseline', 'class_weight'] else 5)

    # Formatting
    ax1.axhline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Decision boundary', zorder=1)
    ax1.set_xlabel('Time Bin (hpf)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean Predicted Probability (with jitter)', fontsize=13, fontweight='bold')
    ax1.set_title(f'True Class: {classes[0]}', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=8, loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3, zorder=0)

    ax2.axhline(0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Decision boundary', zorder=1)
    ax2.set_xlabel('Time Bin (hpf)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Predicted Probability (with jitter)', fontsize=13, fontweight='bold')
    ax2.set_title(f'True Class: {classes[1]}', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=8, loc='best', framealpha=0.9)
    ax2.grid(alpha=0.3, zorder=0)

    fig.suptitle(f'Probability Shift Analysis (with jitter for visibility)\n{group1} vs {group2}',
                fontsize=16, fontweight='bold', y=0.995)

    # Add note about jitter
    fig.text(0.5, 0.02,
             'Note: Small vertical offsets added to separate overlapping lines. Jitter amounts shown in legend.',
             ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Saved: {output_path}")
        plt.close()

    return fig


# ============================================================================
# PLOT ALL COMPARISONS
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
print("PLOTTING PROBABILITY SHIFT WITH JITTER")
print("="*80)

for gene, comparisons in COMPARISONS.items():
    print(f"\n{gene.upper()}:")

    gene_jitter_dir = os.path.join(jitter_dir, gene)
    os.makedirs(gene_jitter_dir, exist_ok=True)

    for group1, group2, comparison in comparisons:
        print(f"\n  {comparison}:")

        output_path = os.path.join(gene_jitter_dir, f'probability_shift_jittered_{comparison}.png')

        plot_probability_shift_with_jitter(gene, comparison, group1, group2, output_path)

print("\n" + "="*80)
print("PLOTTING COMPLETE")
print("="*80)
print(f"\nJittered plots saved to: {jitter_dir}")
print("\nNOTE: Lines are artificially separated by small vertical offsets.")
print("Check legend for offset amounts. True values differ by these offsets.")
