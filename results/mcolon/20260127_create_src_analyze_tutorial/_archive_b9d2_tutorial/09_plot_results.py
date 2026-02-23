"""
Tutorial 09: Plot Difference Detection Results

Demonstrates visualization of classification test results.

Creates AUROC curves over time showing when phenotypes diverge.

Key patterns:
- Time-resolved AUROC visualization
- Significance thresholds and p-value overlays
- Modular plotting for different comparison types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup directories
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# ============================================================================
# Load comparison results
# ============================================================================
print("Loading comparison results...")

results_ovr = pd.read_csv(RESULTS_DIR / "comparison_ovr.csv")
results_vs_np = pd.read_csv(RESULTS_DIR / "comparison_phenotypes_vs_not_penetrant.csv")
results_het_vs_wt = pd.read_csv(RESULTS_DIR / "comparison_het_vs_wt_in_not_penetrant.csv")

print(f"✓ Loaded comparison results from {RESULTS_DIR}")

# ============================================================================
# Plot 1: One-vs-Rest AUROC curves
# ============================================================================
print("\n1. Plotting One-vs-Rest AUROC curves...")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each cluster
for group in results_ovr['group'].unique():
    group_data = results_ovr[results_ovr['group'] == group]

    # Plot AUROC with error bands
    ax.plot(
        group_data['time_bin'],
        group_data['auroc_mean'],
        label=group,
        linewidth=2,
    )
    ax.fill_between(
        group_data['time_bin'],
        group_data['auroc_mean'] - group_data['auroc_std'],
        group_data['auroc_mean'] + group_data['auroc_std'],
        alpha=0.2,
    )

# Reference line at AUROC = 0.5 (random chance)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random chance')

# Styling
ax.set_xlabel('Time (hpf)', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('One-vs-Rest Classification Performance', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 1.0])

plt.tight_layout()
fig.savefig(FIGURES_DIR / "26_auroc_one_vs_rest.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: {FIGURES_DIR / '26_auroc_one_vs_rest.png'}")

# ============================================================================
# Plot 2: Phenotypes vs Not Penetrant
# ============================================================================
print("\n2. Plotting Phenotypes vs Not Penetrant AUROC curves...")

fig, ax = plt.subplots(figsize=(10, 6))

for group in results_vs_np['group'].unique():
    group_data = results_vs_np[results_vs_np['group'] == group]

    ax.plot(
        group_data['time_bin'],
        group_data['auroc_mean'],
        label=f"{group} vs Not Penetrant",
        linewidth=2,
    )
    ax.fill_between(
        group_data['time_bin'],
        group_data['auroc_mean'] - group_data['auroc_std'],
        group_data['auroc_mean'] + group_data['auroc_std'],
        alpha=0.2,
    )

ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random chance')

ax.set_xlabel('Time (hpf)', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Phenotype Clusters vs Not Penetrant', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 1.0])

plt.tight_layout()
fig.savefig(FIGURES_DIR / "27_auroc_phenotypes_vs_not_penetrant.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: {FIGURES_DIR / '27_auroc_phenotypes_vs_not_penetrant.png'}")

# ============================================================================
# Plot 3: Het vs WT (within Not Penetrant)
# ============================================================================
print("\n3. Plotting Het vs WT AUROC curve...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    results_het_vs_wt['time_bin'],
    results_het_vs_wt['auroc_mean'],
    label='Heterozygous vs Wildtype',
    linewidth=2,
    color='#ff7f0e',
)
ax.fill_between(
    results_het_vs_wt['time_bin'],
    results_het_vs_wt['auroc_mean'] - results_het_vs_wt['auroc_std'],
    results_het_vs_wt['auroc_mean'] + results_het_vs_wt['auroc_std'],
    alpha=0.2,
    color='#ff7f0e',
)

ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random chance')

ax.set_xlabel('Time (hpf)', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Heterozygous vs Wildtype (Not Penetrant only)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 1.0])

plt.tight_layout()
fig.savefig(FIGURES_DIR / "28_auroc_het_vs_wt.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: {FIGURES_DIR / '28_auroc_het_vs_wt.png'}")

# ============================================================================
# Plot 4: Combined plot with significance markers
# ============================================================================
print("\n4. Creating combined plot with significance markers...")

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Panel A: One-vs-Rest
ax = axes[0]
for group in results_ovr['group'].unique():
    group_data = results_ovr[results_ovr['group'] == group]
    ax.plot(group_data['time_bin'], group_data['auroc_mean'], label=group, linewidth=2)

    # Mark significant time bins (p < 0.05)
    if 'p_value' in group_data.columns:
        sig_bins = group_data[group_data['p_value'] < 0.05]
        ax.scatter(sig_bins['time_bin'], sig_bins['auroc_mean'], marker='*', s=100, zorder=10)

ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('A. One-vs-Rest', fontsize=12, fontweight='bold', loc='left')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 1.0])

# Panel B: Phenotypes vs Not Penetrant
ax = axes[1]
for group in results_vs_np['group'].unique():
    group_data = results_vs_np[results_vs_np['group'] == group]
    ax.plot(group_data['time_bin'], group_data['auroc_mean'], label=f"{group}", linewidth=2)

    if 'p_value' in group_data.columns:
        sig_bins = group_data[group_data['p_value'] < 0.05]
        ax.scatter(sig_bins['time_bin'], sig_bins['auroc_mean'], marker='*', s=100, zorder=10)

ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('B. Phenotypes vs Not Penetrant', fontsize=12, fontweight='bold', loc='left')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 1.0])

# Panel C: Het vs WT
ax = axes[2]
ax.plot(results_het_vs_wt['time_bin'], results_het_vs_wt['auroc_mean'], linewidth=2, color='#ff7f0e')

if 'p_value' in results_het_vs_wt.columns:
    sig_bins = results_het_vs_wt[results_het_vs_wt['p_value'] < 0.05]
    ax.scatter(sig_bins['time_bin'], sig_bins['auroc_mean'], marker='*', s=100, zorder=10, color='#ff7f0e')

ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Time (hpf)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('C. Het vs WT (Not Penetrant only)', fontsize=12, fontweight='bold', loc='left')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 1.0])

plt.tight_layout()
fig.savefig(FIGURES_DIR / "29_auroc_combined.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: {FIGURES_DIR / '29_auroc_combined.png'}")

# ============================================================================
# Summary table: Peak AUROC per comparison
# ============================================================================
print("\n5. Creating summary table...")

summary_rows = []

# One-vs-Rest
for group in results_ovr['group'].unique():
    group_data = results_ovr[results_ovr['group'] == group]
    peak_idx = group_data['auroc_mean'].idxmax()
    peak_row = group_data.loc[peak_idx]

    summary_rows.append({
        'comparison_type': 'One-vs-Rest',
        'group': group,
        'reference': 'rest',
        'peak_auroc': peak_row['auroc_mean'],
        'peak_time_hpf': peak_row['time_bin'],
    })

# Phenotypes vs Not Penetrant
for group in results_vs_np['group'].unique():
    group_data = results_vs_np[results_vs_np['group'] == group]
    peak_idx = group_data['auroc_mean'].idxmax()
    peak_row = group_data.loc[peak_idx]

    summary_rows.append({
        'comparison_type': 'Phenotype vs NP',
        'group': group,
        'reference': 'Not Penetrant',
        'peak_auroc': peak_row['auroc_mean'],
        'peak_time_hpf': peak_row['time_bin'],
    })

# Het vs WT
peak_idx = results_het_vs_wt['auroc_mean'].idxmax()
peak_row = results_het_vs_wt.loc[peak_idx]
summary_rows.append({
    'comparison_type': 'Het vs WT',
    'group': 'b9d2_heterozygous',
    'reference': 'b9d2_wildtype',
    'peak_auroc': peak_row['auroc_mean'],
    'peak_time_hpf': peak_row['time_bin'],
})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "auroc_peak_summary.csv", index=False)

print("\nPeak AUROC Summary:")
print(summary_df.to_string(index=False))
print(f"\nSaved to: {RESULTS_DIR / 'auroc_peak_summary.csv'}")

# ============================================================================
# Modular plotting function (for notebook use)
# ============================================================================
print("\n" + "="*80)
print("MODULAR PLOTTING PATTERN")
print("="*80)
print("""
For Jupyter notebooks, you can create reusable plotting functions:

def plot_auroc_curve(results_df, title, output_path=None):
    '''Plot AUROC curves from classification results.'''
    fig, ax = plt.subplots(figsize=(10, 6))

    for group in results_df['group'].unique():
        group_data = results_df[results_df['group'] == group]
        ax.plot(group_data['time_bin'], group_data['auroc_mean'],
                label=group, linewidth=2)
        ax.fill_between(group_data['time_bin'],
                        group_data['auroc_mean'] - group_data['auroc_std'],
                        group_data['auroc_mean'] + group_data['auroc_std'],
                        alpha=0.2)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('AUROC')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig

# Usage:
# fig = plot_auroc_curve(results_ovr, 'One-vs-Rest AUROC')
# fig.show()
""")

print("\n✓ Tutorial 09 complete!")
print(f"  All figures saved to: {FIGURES_DIR}")
print(f"  Summary saved to: {RESULTS_DIR}")
