"""
Apply two-sided SA outlier detection with tuned thresholds.

Uses global reference curves (p5/p95 vs stage) to flag embryos that are:
- Too large: SA > k_upper × p95 (segmentation artifacts)
- Too small: SA < k_lower × p5 (incomplete masks, dead embryos)

Based on tuning analysis:
- k_lower = 0.7 catches F03, F06, H07 (too small)
- k_upper = 1.4 (reasonable for upper bound)

Usage:
    conda activate segmentations_grounded_sam
    python apply_two_sided_qc.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BUILD04_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output")
OUTPUT_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/tests/sa_outlier_analysis/outputs")
REF_CURVES = OUTPUT_DIR / "sa_reference_curves.csv"

# Tuned thresholds
K_UPPER = 1.2  # Upper threshold = 1.4 × p95
K_LOWER = 0.9  # Lower threshold = 0.7 × p5

# Test on this experiment
TEST_EXPERIMENT = '20250711'
TEST_EMBRYOS = ['20250711_F03_e01', '20250711_F06_e01', '20250711_H07_e01']

print("=" * 80)
print("TWO-SIDED SA OUTLIER DETECTION")
print("=" * 80)
print(f"\nThresholds:")
print(f"  Upper: SA > {K_UPPER} × p95 (too large)")
print(f"  Lower: SA < {K_LOWER} × p5 (too small)")

# Load reference curves
print(f"\nLoading reference curves: {REF_CURVES}")
ref_df = pd.read_csv(REF_CURVES)
print(f"✓ Loaded {len(ref_df)} stage bins")

# Load test data
print(f"\nLoading test experiment: {TEST_EXPERIMENT}")
test_file = BUILD04_DIR / f"qc_staged_{TEST_EXPERIMENT}.csv"
df = pd.read_csv(test_file)
print(f"✓ Loaded {len(df)} rows, {df['embryo_id'].nunique()} embryos")

# Interpolate reference values at each embryo's stage
print("\n" + "=" * 80)
print("APPLYING TWO-SIDED OUTLIER DETECTION")
print("=" * 80)

# Simple interpolation (reference curves already filled/extrapolated during generation)
df['ref_p5'] = np.interp(df['predicted_stage_hpf'], ref_df['stage_hpf'], ref_df['p5'])
df['ref_p95'] = np.interp(df['predicted_stage_hpf'], ref_df['stage_hpf'], ref_df['p95'])

# Calculate thresholds
df['upper_threshold'] = K_UPPER * df['ref_p95']
df['lower_threshold'] = K_LOWER * df['ref_p5']

# Flag outliers (two-sided)
df['sa_outlier_upper'] = df['surface_area_um'] > df['upper_threshold']
df['sa_outlier_lower'] = df['surface_area_um'] < df['lower_threshold']
df['sa_outlier_flag_new'] = df['sa_outlier_upper'] | df['sa_outlier_lower']

# Compare to old flags
if 'sa_outlier_flag' in df.columns:
    old_flags = df['sa_outlier_flag'].sum()
    print(f"\nOld (one-sided) flags: {old_flags} frames")
else:
    old_flags = 0
    print(f"\nOld flags: N/A (column not found)")

new_flags = df['sa_outlier_flag_new'].sum()
upper_flags = df['sa_outlier_upper'].sum()
lower_flags = df['sa_outlier_lower'].sum()

print(f"New (two-sided) flags: {new_flags} frames")
print(f"  - Too large (> {K_UPPER}×p95): {upper_flags}")
print(f"  - Too small (< {K_LOWER}×p5): {lower_flags}")

# Embryo-level summary
embryo_flagged_old = df.groupby('embryo_id')['sa_outlier_flag'].max() if 'sa_outlier_flag' in df.columns else pd.Series(dtype=bool)
embryo_flagged_new = df.groupby('embryo_id')['sa_outlier_flag_new'].max()

n_embryos = df['embryo_id'].nunique()
n_flagged_new = embryo_flagged_new.sum()
pct_flagged = 100 * n_flagged_new / n_embryos

print(f"\nEmbryos flagged (any frame): {n_flagged_new} / {n_embryos} ({pct_flagged:.1f}%)")

# Check test embryos
print("\n" + "=" * 80)
print("TEST EMBRYO VALIDATION")
print("=" * 80)

for emb_id in TEST_EMBRYOS:
    if emb_id in embryo_flagged_new.index:
        is_flagged = embryo_flagged_new[emb_id]
        emb_data = df[df['embryo_id'] == emb_id]
        n_frames_flagged = emb_data['sa_outlier_flag_new'].sum()
        n_upper = emb_data['sa_outlier_upper'].sum()
        n_lower = emb_data['sa_outlier_lower'].sum()

        status = "✓ CAUGHT" if is_flagged else "✗ MISSED"
        print(f"\n{emb_id}: {status}")
        print(f"  Frames flagged: {n_frames_flagged} / {len(emb_data)}")
        print(f"    - Too large: {n_upper}")
        print(f"    - Too small: {n_lower}")
    else:
        print(f"\n{emb_id}: Not found in dataset")

# Visualization: Show flagged embryos
print("\n" + "=" * 80)
print("GENERATING VALIDATION PLOT")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top panel: All data with reference bands and flagged points
ax = axes[0]

# Reference curves
ax.plot(ref_df['stage_hpf'], ref_df['p50'] / 1e6, 'b-', linewidth=2, label='p50 (reference)', zorder=1)
ax.fill_between(ref_df['stage_hpf'], ref_df['p5'] / 1e6, ref_df['p95'] / 1e6,
                alpha=0.2, color='blue', label='p5-p95 range', zorder=1)

# Threshold bands
ax.plot(ref_df['stage_hpf'], K_UPPER * ref_df['p95'] / 1e6, 'r--', linewidth=1,
        label=f'Upper threshold ({K_UPPER}×p95)', alpha=0.7, zorder=2)
ax.plot(ref_df['stage_hpf'], K_LOWER * ref_df['p5'] / 1e6, 'r--', linewidth=1,
        label=f'Lower threshold ({K_LOWER}×p5)', alpha=0.7, zorder=2)

# Non-flagged points (background)
df_normal = df[~df['sa_outlier_flag_new']]
ax.scatter(df_normal['predicted_stage_hpf'], df_normal['surface_area_um'] / 1e6,
           c='gray', s=1, alpha=0.1, label='Normal embryos', zorder=3)

# Flagged points (highlighted)
df_flagged = df[df['sa_outlier_flag_new']]
ax.scatter(df_flagged['predicted_stage_hpf'], df_flagged['surface_area_um'] / 1e6,
           c='red', s=5, alpha=0.5, label=f'Flagged ({n_flagged_new} embryos)', zorder=4)

# Highlight test embryos
colors_test = ['purple', 'red', 'orange']
for i, emb_id in enumerate(TEST_EMBRYOS):
    emb_data = df[df['embryo_id'] == emb_id].sort_values('predicted_stage_hpf')
    if len(emb_data) > 0:
        ax.plot(emb_data['predicted_stage_hpf'], emb_data['surface_area_um'] / 1e6,
                color=colors_test[i], linewidth=2, marker='o', markersize=3,
                label=emb_id, zorder=5, alpha=0.8)

ax.set_xlabel('Predicted Stage (hpf)', fontsize=12)
ax.set_ylabel('Surface Area (mm²)', fontsize=12)
ax.set_title(f'Two-Sided SA Outlier Detection - {TEST_EXPERIMENT}', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 2.5)

# Bottom panel: Flagging breakdown by stage
ax = axes[1]

# Bin by stage
stage_bins = np.arange(0, 130, 5)
df['stage_bin'] = pd.cut(df['predicted_stage_hpf'], bins=stage_bins)

# Count flags per bin
bin_stats = df.groupby('stage_bin').agg({
    'sa_outlier_flag_new': 'sum',
    'sa_outlier_upper': 'sum',
    'sa_outlier_lower': 'sum',
    'embryo_id': 'count'
}).rename(columns={'embryo_id': 'total'})

bin_stats['pct_flagged'] = 100 * bin_stats['sa_outlier_flag_new'] / bin_stats['total']

# Plot
bin_centers = [(b.left + b.right) / 2 for b in bin_stats.index]
ax.bar(bin_centers, bin_stats['sa_outlier_upper'], width=4, alpha=0.6, color='orange', label='Too large')
ax.bar(bin_centers, bin_stats['sa_outlier_lower'], width=4, alpha=0.6, color='purple',
       bottom=bin_stats['sa_outlier_upper'], label='Too small')

ax.set_xlabel('Predicted Stage (hpf)', fontsize=12)
ax.set_ylabel('Flagged Frames', fontsize=12)
ax.set_title('Flagged Frames by Stage and Type', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()

output_plot = OUTPUT_DIR / "two_sided_qc_validation.png"
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {output_plot}")

# Save flagged embryos list
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# List of all flagged embryos
flagged_embryos = embryo_flagged_new[embryo_flagged_new].index.tolist()

output_csv = OUTPUT_DIR / "flagged_embryos_list.csv"
flagged_df = pd.DataFrame({'embryo_id': flagged_embryos})
flagged_df.to_csv(output_csv, index=False)
print(f"✓ Saved flagged embryo list to: {output_csv}")
print(f"  {len(flagged_embryos)} embryos flagged")

# Summary stats
summary = {
    'experiment': TEST_EXPERIMENT,
    'k_upper': K_UPPER,
    'k_lower': K_LOWER,
    'total_embryos': n_embryos,
    'flagged_embryos': n_flagged_new,
    'pct_flagged': pct_flagged,
    'frames_too_large': upper_flags,
    'frames_too_small': lower_flags,
    'F03_caught': embryo_flagged_new.get('20250711_F03_e01', False),
    'F06_caught': embryo_flagged_new.get('20250711_F06_e01', False),
    'H07_caught': embryo_flagged_new.get('20250711_H07_e01', False),
}

summary_df = pd.DataFrame([summary])
output_summary = OUTPUT_DIR / "qc_validation_summary.csv"
summary_df.to_csv(output_summary, index=False)
print(f"✓ Saved summary to: {output_summary}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nThresholds: k_upper={K_UPPER}, k_lower={K_LOWER}")
print(f"Flagged: {n_flagged_new}/{n_embryos} embryos ({pct_flagged:.1f}%)")
print(f"Test embryos caught: F03={'✓' if summary['F03_caught'] else '✗'}, F06={'✓' if summary['F06_caught'] else '✗'}, H07={'✓' if summary['H07_caught'] else '✗'}")
print("\n✓ Done!")
