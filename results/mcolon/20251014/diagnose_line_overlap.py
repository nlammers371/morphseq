"""
Diagnose why baseline and class_weight lines aren't visible despite being plotted.

Tests:
1. Are predicted probabilities nearly identical between methods?
2. Visual separation with distinct styles per method
3. Plot methods individually to confirm they exist
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
output_dir = os.path.join(results_dir, "imbalance_methods", "diagnostics", "line_overlap")
os.makedirs(output_dir, exist_ok=True)

print(f"Line overlap diagnostic\n")

METHODS = ['baseline', 'class_weight', 'embryo_weight', 'combined_weight',
           'calibrated_class_weight', 'calibrated_combined_weight', 'balanced_bootstrap']

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def check_probability_similarity(gene, comparison, class_label):
    """
    Check if predicted probabilities are nearly identical across methods.
    """
    data_dir = os.path.join(data_dir_base, gene)

    print(f"\n{'='*80}")
    print(f"PROBABILITY SIMILARITY CHECK")
    print(f"{gene} - {comparison} - {class_label}")
    print(f"{'='*80}")

    # Load all methods
    method_data = {}
    for method_name in METHODS:
        csv_path = os.path.join(data_dir, f"embryo_probs_{method_name}_{comparison}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_cls = df[df['true_label'] == class_label]
            if not df_cls.empty:
                mean_probs = df_cls.groupby('time_bin')['pred_proba'].mean()
                method_data[method_name] = mean_probs

    if len(method_data) < 2:
        print("  Insufficient methods to compare")
        return

    # Compare baseline and class_weight to others
    if 'baseline' in method_data and 'class_weight' in method_data:
        baseline_vals = method_data['baseline']
        classweight_vals = method_data['class_weight']

        # Find common time bins
        common_bins = baseline_vals.index.intersection(classweight_vals.index)

        if len(common_bins) > 0:
            baseline_common = baseline_vals.loc[common_bins]
            classweight_common = classweight_vals.loc[common_bins]

            # Compute difference
            diff = np.abs(baseline_common - classweight_common)

            print(f"\nBaseline vs Class_weight:")
            print(f"  Time bins compared: {len(common_bins)}")
            print(f"  Mean absolute difference: {diff.mean():.6f}")
            print(f"  Max absolute difference: {diff.max():.6f}")
            print(f"  Min absolute difference: {diff.min():.6f}")

            if diff.max() < 0.01:
                print(f"  ⚠ WARNING: Predictions are NEARLY IDENTICAL (max diff < 0.01)")
                print(f"     Lines will be visually indistinguishable!")

        # Compare against other methods
        print(f"\nBaseline vs other methods:")
        for other_method, other_vals in method_data.items():
            if other_method in ['baseline', 'class_weight']:
                continue

            common_bins = baseline_vals.index.intersection(other_vals.index)
            if len(common_bins) > 0:
                baseline_common = baseline_vals.loc[common_bins]
                other_common = other_vals.loc[common_bins]
                diff = np.abs(baseline_common - other_common)

                print(f"  vs {other_method}:")
                print(f"    Mean diff: {diff.mean():.6f}, Max diff: {diff.max():.6f}")


def plot_with_distinct_styles(gene, comparison, group1, group2, output_path=None):
    """
    Plot with highly distinct styles to ensure visibility.
    """
    data_dir = os.path.join(data_dir_base, gene)

    # Load data
    results_by_method = {}
    for method_name in METHODS:
        csv_path = os.path.join(data_dir, f"embryo_probs_{method_name}_{comparison}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                results_by_method[method_name] = df

    if not results_by_method:
        print("  No data to plot")
        return

    # Define VERY distinct styles
    styles = {
        'baseline': {'color': 'red', 'linestyle': '-', 'linewidth': 3, 'marker': 'o', 'markersize': 8, 'alpha': 1.0},
        'class_weight': {'color': 'blue', 'linestyle': '--', 'linewidth': 3, 'marker': 's', 'markersize': 8, 'alpha': 1.0},
        'embryo_weight': {'color': 'green', 'linestyle': '-.', 'linewidth': 2, 'marker': '^', 'markersize': 6, 'alpha': 0.7},
        'combined_weight': {'color': 'purple', 'linestyle': ':', 'linewidth': 2, 'marker': 'v', 'markersize': 6, 'alpha': 0.7},
        'calibrated_class_weight': {'color': 'orange', 'linestyle': '-', 'linewidth': 1.5, 'marker': 'd', 'markersize': 5, 'alpha': 0.6},
        'calibrated_combined_weight': {'color': 'brown', 'linestyle': '--', 'linewidth': 1.5, 'marker': 'p', 'markersize': 5, 'alpha': 0.6},
        'balanced_bootstrap': {'color': 'pink', 'linestyle': '-.', 'linewidth': 1.5, 'marker': '*', 'markersize': 7, 'alpha': 0.6}
    }

    # Get classes
    first_df = list(results_by_method.values())[0]
    classes = sorted(first_df['true_label'].unique())

    if len(classes) != 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax1, ax2 = axes

    print(f"\n  Plotting with distinct styles:")

    for method_name, df_probs in results_by_method.items():
        style = styles.get(method_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1})

        for cls_idx, cls in enumerate(classes):
            df_cls = df_probs[df_probs['true_label'] == cls]
            if df_cls.empty:
                continue

            mean_probs = df_cls.groupby('time_bin')['pred_proba'].mean()
            if len(mean_probs) == 0:
                continue

            ax = ax1 if cls_idx == 0 else ax2

            # Plot with explicit style
            ax.plot(mean_probs.index, mean_probs.values,
                   label=method_name,
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['linewidth'],
                   marker=style['marker'],
                   markersize=style['markersize'],
                   alpha=style['alpha'],
                   zorder=10 if method_name in ['baseline', 'class_weight'] else 5)  # Bring to front

            print(f"    {method_name} ({cls}): {style['color']} {style['linestyle']}")

    # Formatting
    for ax, cls in zip([ax1, ax2], classes):
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Decision boundary', zorder=1)
        ax.set_xlabel('Time Bin (hpf)', fontsize=12)
        ax.set_ylabel('Mean Predicted Probability', fontsize=12)
        ax.set_title(f'True Class: {cls}', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3, zorder=0)

    fig.suptitle(f'Probability Shift (Distinct Styles)\n{group1} vs {group2}',
                fontsize=15, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n  Saved: {output_path}")

    return fig


def plot_methods_individually(gene, comparison, group1, group2, output_dir):
    """
    Plot each method on its own subplot to confirm data exists.
    """
    data_dir = os.path.join(data_dir_base, gene)

    # Load data
    results_by_method = {}
    for method_name in METHODS:
        csv_path = os.path.join(data_dir, f"embryo_probs_{method_name}_{comparison}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                results_by_method[method_name] = df

    if not results_by_method:
        return

    # Get classes
    first_df = list(results_by_method.values())[0]
    classes = sorted(first_df['true_label'].unique())

    if len(classes) != 2:
        return

    # Create grid of subplots
    n_methods = len(results_by_method)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()

    print(f"\n  Plotting methods individually:")

    for method_idx, (method_name, df_probs) in enumerate(results_by_method.items()):
        ax = axes[method_idx]

        for cls_idx, cls in enumerate(classes):
            df_cls = df_probs[df_probs['true_label'] == cls]
            if df_cls.empty:
                continue

            mean_probs = df_cls.groupby('time_bin')['pred_proba'].mean()
            if len(mean_probs) == 0:
                continue

            color = 'steelblue' if cls_idx == 0 else 'coral'
            ax.plot(mean_probs.index, mean_probs.values,
                   label=cls, color=color, linewidth=2, marker='o', markersize=4)

            print(f"    {method_name}: {len(mean_probs)} time bins for {cls}")

        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time Bin (hpf)', fontsize=10)
        ax.set_ylabel('Mean Pred Prob', fontsize=10)
        ax.set_title(method_name, fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Individual Method Plots\n{group1} vs {group2}',
                fontsize=15, fontweight='bold')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'individual_methods_{gene}_{comparison}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {plot_path}")

    return fig


# ============================================================================
# RUN DIAGNOSTICS
# ============================================================================

print("="*80)
print("LINE OVERLAP / VISIBILITY DIAGNOSTIC")
print("="*80)

# Focus on tmem67 heterozygote vs homozygous (where you're seeing the issue)
GENE = "tmem67"
COMPARISON = "heterozygote_vs_homozygous"
GROUP1 = "tmem67_heterozygote"
GROUP2 = "tmem67_homozygous"

# Check probability similarity
for class_label in [GROUP1, GROUP2]:
    check_probability_similarity(GENE, COMPARISON, class_label)

# Plot with distinct styles
print(f"\n{'='*80}")
print(f"PLOTTING WITH DISTINCT STYLES")
print(f"{'='*80}")

output_path = os.path.join(output_dir, f'distinct_styles_{GENE}_{COMPARISON}.png')
plot_with_distinct_styles(GENE, COMPARISON, GROUP1, GROUP2, output_path)

# Plot methods individually
print(f"\n{'='*80}")
print(f"PLOTTING METHODS INDIVIDUALLY")
print(f"{'='*80}")

plot_methods_individually(GENE, COMPARISON, GROUP1, GROUP2, output_dir)

# Also check cep290 for comparison
print(f"\n\n{'='*80}")
print("BONUS: CHECK CEP290 (for comparison)")
print(f"{'='*80}")

check_probability_similarity('cep290', 'wildtype_vs_heterozygous', 'cep290_wildtype')
check_probability_similarity('cep290', 'wildtype_vs_heterozygous', 'cep290_heterozygous')

print(f"\n{'='*80}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*80}")
print(f"\nOutputs saved to: {output_dir}")
