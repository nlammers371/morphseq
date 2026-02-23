"""
Tune SA outlier detection thresholds using F06/H07 as test cases.

Tests different upper/lower multipliers to find optimal balance between
catching true outliers and minimizing false positives.

Usage:
    conda activate segmentations_grounded_sam
    python tune_thresholds.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BUILD04_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output")
OUTPUT_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/tests/sa_outlier_analysis/outputs")
REF_CURVES = OUTPUT_DIR / "sa_reference_curves.csv"

# Test cases - known problematic embryos (segmentation issues)
TEST_EMBRYOS = ['20250711_F03_e01', '20250711_F06_e01', '20250711_H07_e01']
TEST_EXPERIMENT = '20250711'

print("=" * 80)
print("TUNING SA OUTLIER DETECTION THRESHOLDS")
print("=" * 80)

# Load reference curves
print(f"\nLoading reference curves from: {REF_CURVES}")
ref_df = pd.read_csv(REF_CURVES)
print(f"✓ Loaded {len(ref_df)} stage bins (range: {ref_df['stage_hpf'].min():.1f} - {ref_df['stage_hpf'].max():.1f} hpf)")

# Load test data
print(f"\nLoading test experiment: {TEST_EXPERIMENT}")
test_file = BUILD04_DIR / f"qc_staged_{TEST_EXPERIMENT}.csv"
df_test = pd.read_csv(test_file)
print(f"✓ Loaded {len(df_test)} rows, {df_test['embryo_id'].nunique()} embryos")

# Extract test embryos
df_test_embs = df_test[df_test['embryo_id'].isin(TEST_EMBRYOS)].copy()
print(f"\nTest embryos found: {df_test_embs['embryo_id'].nunique()}")
for emb_id in TEST_EMBRYOS:
    n = (df_test_embs['embryo_id'] == emb_id).sum()
    print(f"  {emb_id}: {n} frames")

# Interpolate reference curves at test embryo stages
print("\n" + "=" * 80)
print("INTERPOLATING REFERENCE VALUES")
print("=" * 80)

# For each test embryo frame, get expected p5/p50/p95
df_test_embs['ref_p5'] = np.interp(
    df_test_embs['predicted_stage_hpf'],
    ref_df['stage_hpf'],
    ref_df['p5']
)
df_test_embs['ref_p50'] = np.interp(
    df_test_embs['predicted_stage_hpf'],
    ref_df['stage_hpf'],
    ref_df['p50']
)
df_test_embs['ref_p95'] = np.interp(
    df_test_embs['predicted_stage_hpf'],
    ref_df['stage_hpf'],
    ref_df['p95']
)

# Calculate ratios
df_test_embs['ratio_to_p50'] = df_test_embs['surface_area_um'] / df_test_embs['ref_p50']
df_test_embs['ratio_to_p5'] = df_test_embs['surface_area_um'] / df_test_embs['ref_p5']
df_test_embs['ratio_to_p95'] = df_test_embs['surface_area_um'] / df_test_embs['ref_p95']

print("\nRatio statistics for test embryos:")
for emb_id in TEST_EMBRYOS:
    emb_data = df_test_embs[df_test_embs['embryo_id'] == emb_id]
    print(f"\n{emb_id}:")
    print(f"  SA / p50: {emb_data['ratio_to_p50'].min():.2f} - {emb_data['ratio_to_p50'].max():.2f} (mean: {emb_data['ratio_to_p50'].mean():.2f})")
    print(f"  SA / p5:  {emb_data['ratio_to_p5'].min():.2f} - {emb_data['ratio_to_p5'].max():.2f} (mean: {emb_data['ratio_to_p5'].mean():.2f})")
    print(f"  SA / p95: {emb_data['ratio_to_p95'].min():.2f} - {emb_data['ratio_to_p95'].max():.2f} (mean: {emb_data['ratio_to_p95'].mean():.2f})")

# Test different threshold combinations
print("\n" + "=" * 80)
print("TESTING THRESHOLD COMBINATIONS")
print("=" * 80)

# Grid of k values to test
k_upper_values = [1.2, 1.4, 1.6, 2.0]
k_lower_values = [0.5, 0.6, 0.7, 0.8]

# For each embryo in full dataset
df_full = df_test.copy()
df_full['ref_p5'] = np.interp(df_full['predicted_stage_hpf'], ref_df['stage_hpf'], ref_df['p5'])
df_full['ref_p95'] = np.interp(df_full['predicted_stage_hpf'], ref_df['stage_hpf'], ref_df['p95'])

results = []

for k_upper in k_upper_values:
    for k_lower in k_lower_values:
        # Apply thresholds
        upper_threshold = k_upper * df_full['ref_p95']
        lower_threshold = k_lower * df_full['ref_p5']

        # Flag outliers
        df_full['outlier_flag_test'] = (
            (df_full['surface_area_um'] > upper_threshold) |
            (df_full['surface_area_um'] < lower_threshold)
        )

        # Count embryos flagged (any frame flagged = embryo flagged)
        embryo_flagged = df_full.groupby('embryo_id')['outlier_flag_test'].max()
        n_flagged = embryo_flagged.sum()
        pct_flagged = 100 * n_flagged / len(embryo_flagged)

        # Check if test embryos are caught
        test_caught = []
        for emb_id in TEST_EMBRYOS:
            if emb_id in embryo_flagged.index:
                test_caught.append(embryo_flagged[emb_id])
            else:
                test_caught.append(False)

        results.append({
            'k_upper': k_upper,
            'k_lower': k_lower,
            'n_flagged': n_flagged,
            'pct_flagged': pct_flagged,
            'F03_caught': test_caught[0] if len(test_caught) > 0 else False,
            'F06_caught': test_caught[1] if len(test_caught) > 1 else False,
            'H07_caught': test_caught[2] if len(test_caught) > 2 else False,
            'all_caught': all(test_caught)
        })

results_df = pd.DataFrame(results)

print("\nThreshold tuning results:")
print(results_df.to_string(index=False))

# Highlight best options (catches all test embryos, <10% flagged)
best = results_df[(results_df['all_caught']) & (results_df['pct_flagged'] < 10)]
print(f"\n✓ Best options (catches F03+F06+H07, <10% flagged):")
print(best.to_string(index=False))

# Save results
output_csv = OUTPUT_DIR / "threshold_tuning_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\n✓ Saved results to: {output_csv}")

# Visualization 1: Test embryo trajectories vs reference
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top panel: Trajectories with reference bands
ax = axes[0]

# Reference curves (convert to mm²)
ax.plot(ref_df['stage_hpf'], ref_df['p50'] / 1e6, 'b-', linewidth=2, label='p50 (reference)', zorder=1)
ax.fill_between(ref_df['stage_hpf'], ref_df['p5'] / 1e6, ref_df['p95'] / 1e6,
                alpha=0.3, color='blue', label='p5-p95 range', zorder=1)

# Test embryo trajectories
colors = ['purple', 'red', 'orange']
for i, emb_id in enumerate(TEST_EMBRYOS):
    emb_data = df_test_embs[df_test_embs['embryo_id'] == emb_id].sort_values('predicted_stage_hpf')
    ax.plot(emb_data['predicted_stage_hpf'], emb_data['surface_area_um'] / 1e6,
            color=colors[i], linewidth=1.5, marker='o', markersize=2,
            label=emb_id, zorder=3, alpha=0.7)

ax.set_xlabel('Predicted Stage (hpf)', fontsize=12)
ax.set_ylabel('Surface Area (mm²)', fontsize=12)
ax.set_title('Test Embryos vs Reference Curves', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

# Bottom panel: Ratio to p50 over time
ax = axes[1]

colors_bottom = ['purple', 'red', 'orange']
for i, emb_id in enumerate(TEST_EMBRYOS):
    emb_data = df_test_embs[df_test_embs['embryo_id'] == emb_id].sort_values('predicted_stage_hpf')
    ax.plot(emb_data['predicted_stage_hpf'], emb_data['ratio_to_p50'],
            color=colors_bottom[i], linewidth=1.5, marker='o', markersize=2,
            label=emb_id, alpha=0.7)

# Add threshold lines
ax.axhline(y=1.0, color='blue', linestyle='-', linewidth=1, label='p50 reference', alpha=0.5)
ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1, label='Lower threshold (0.7x p5)', alpha=0.5)
ax.axhline(y=1.4, color='green', linestyle='--', linewidth=1, label='Upper threshold (1.4x p95)', alpha=0.5)

ax.set_xlabel('Predicted Stage (hpf)', fontsize=12)
ax.set_ylabel('SA / p50 Ratio', fontsize=12)
ax.set_title('Normalized SA Trajectories (relative to reference p50)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 2)

plt.tight_layout()

output_plot1 = OUTPUT_DIR / "test_embryos_vs_reference.png"
plt.savefig(output_plot1, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {output_plot1}")

# Visualization 2: Flagging rate heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Pivot for heatmap
heatmap_data = results_df.pivot(index='k_lower', columns='k_upper', values='pct_flagged')

# Create heatmap
im = ax.imshow(heatmap_data.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=20)

# Set ticks
ax.set_xticks(np.arange(len(k_upper_values)))
ax.set_yticks(np.arange(len(k_lower_values)))
ax.set_xticklabels(k_upper_values)
ax.set_yticklabels(k_lower_values)

# Labels
ax.set_xlabel('k_upper (multiplier for p95 upper threshold)', fontsize=12)
ax.set_ylabel('k_lower (multiplier for p5 lower threshold)', fontsize=12)
ax.set_title('Embryo Flagging Rate (%) by Threshold Combination', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(k_lower_values)):
    for j in range(len(k_upper_values)):
        pct = heatmap_data.values[i, j]
        k_u = k_upper_values[j]
        k_l = k_lower_values[i]

        # Check if catches test embryos
        row = results_df[(results_df['k_upper'] == k_u) & (results_df['k_lower'] == k_l)]
        all_caught = row['all_caught'].values[0]

        text_color = 'white' if pct > 10 else 'black'
        marker = '✓' if all_caught else '✗'

        ax.text(j, i, f'{pct:.1f}%\n{marker}',
                ha='center', va='center', color=text_color, fontsize=10)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('% Embryos Flagged', rotation=270, labelpad=20)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='white', edgecolor='black', label='✓ = Catches F03 + F06 + H07'),
    Patch(facecolor='white', edgecolor='black', label='✗ = Misses at least one')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))

plt.tight_layout()

output_plot2 = OUTPUT_DIR / "flagging_rate_heatmap.png"
plt.savefig(output_plot2, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {output_plot2}")

# Summary recommendation
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if len(best) > 0:
    # Choose option with lowest flagging rate
    recommended = best.loc[best['pct_flagged'].idxmin()]
    print(f"\n✅ RECOMMENDED THRESHOLDS:")
    print(f"   k_upper = {recommended['k_upper']} (upper threshold = {recommended['k_upper']}x p95)")
    print(f"   k_lower = {recommended['k_lower']} (lower threshold = {recommended['k_lower']}x p5)")
    print(f"\n   Performance:")
    print(f"   - Catches F03_e01: {'Yes ✓' if recommended['F03_caught'] else 'No ✗'}")
    print(f"   - Catches F06_e01: {'Yes ✓' if recommended['F06_caught'] else 'No ✗'}")
    print(f"   - Catches H07_e01: {'Yes ✓' if recommended['H07_caught'] else 'No ✗'}")
    print(f"   - Flagging rate: {recommended['pct_flagged']:.1f}% of embryos ({recommended['n_flagged']} / {len(embryo_flagged)})")
else:
    print("\n⚠️  No threshold combination catches all test embryos with <10% flagging rate")
    print("    Consider relaxing constraints or investigating test embryos further")

print("\n✓ Done!")
