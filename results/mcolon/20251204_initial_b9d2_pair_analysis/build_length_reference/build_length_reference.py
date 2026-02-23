#!/usr/bin/env python3
"""
Build Total Length reference curves for wildtype embryos.

Aggregates total_length_um distributions from specified experiments to establish
normal ranges (p5, p50, p95) vs predicted_stage_hpf.

Adapted from SA reference builder (tests/20251010_sa_outlier_analysis/build_sa_reference.py)

Experiments: 20250912, 20251106-20251119
Filters: wildtype genotype, no chem perturbation, wik/ab genetic background

Usage:
    python build_length_reference.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.signal
import sys

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')
from src.analyze.trajectory_analysis.data_loading import _load_df03_format

# Configuration
EXPERIMENT_IDS = [
    '20250912',
    '20251106', '20251107', '20251108', '20251109', '20251110',
    '20251111', '20251112', '20251113', '20251114', '20251115',
    '20251116', '20251117', '20251118', '20251119'
]

# Output
OUTPUT_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251204_initial_b9d2_pair_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BUILDING TOTAL LENGTH REFERENCE CURVES FROM WILDTYPE CONTROLS")
print("=" * 80)
print(f"\nExperiments: {len(EXPERIMENT_IDS)}")
for exp_id in EXPERIMENT_IDS:
    print(f"  - {exp_id}")

# Load and concatenate data
all_data = []
for exp_id in EXPERIMENT_IDS:
    print(f"\nLoading {exp_id}...", end=" ")

    try:
        df = _load_df03_format(exp_id)

        # Handle total_length_um column collision
        if 'total_length_um_y' in df.columns:
            df['total_length_um'] = df['total_length_um_y']

        df['experiment_id_loaded'] = exp_id
        all_data.append(df)
        print(f"✓ ({len(df)} rows)")
    except Exception as e:
        print(f"✗ Error: {e}")

if not all_data:
    raise ValueError("No data loaded! Check experiment IDs and file paths.")

df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal loaded: {len(df_all)} rows from {len(all_data)} experiments")

# Filter for true wild-type controls
print("\n" + "=" * 80)
print("FILTERING FOR TRUE WILD-TYPE CONTROLS")
print("=" * 80)

# Wildtype genotype filter
wt_genotypes = ['wildtype', 'wt', 'WT', 'Wildtype']
genotype_mask = pd.Series([False] * len(df_all))

if 'genotype' in df_all.columns:
    # Check if genotype contains "wildtype" (case-insensitive)
    genotype_contains_wt = df_all['genotype'].astype(str).str.lower().str.contains('wildtype', na=False)
    # Also check exact matches for wt variations
    genotype_exact_wt = df_all['genotype'].astype(str).isin(wt_genotypes)
    genotype_mask = genotype_contains_wt | genotype_exact_wt
    print(f"Wildtype genotype filter: {genotype_mask.sum()} / {len(df_all)} rows")
else:
    print("⚠️  'genotype' column not found!")

# Genetic perturbation filter (wik, ab, wik-ab)
genetic_backgrounds = ['wik', 'ab', 'wik-ab', 'AB', 'WIK']
genetic_mask = pd.Series([True] * len(df_all))  # Default to True if column missing

if 'genetic_perturbation' in df_all.columns:
    genetic_mask = df_all['genetic_perturbation'].astype(str).isin(genetic_backgrounds)
    print(f"Genetic background filter (wik/ab): {genetic_mask.sum()} / {len(df_all)} rows")
else:
    print("⚠️  'genetic_perturbation' column not found, skipping filter")

# No chemical perturbation
chem_mask = pd.Series([True] * len(df_all))

if 'chem_perturbation' in df_all.columns:
    chem_mask = (df_all['chem_perturbation'].astype(str) == 'None') | df_all['chem_perturbation'].isna()
    print(f"No chem perturbation: {chem_mask.sum()} / {len(df_all)} rows")
else:
    print("⚠️  'chem_perturbation' column not found, skipping filter")

# Use embryo flag
use_mask = pd.Series([True] * len(df_all))

if 'use_embryo_flag' in df_all.columns:
    use_mask = df_all['use_embryo_flag'].astype(bool)
    print(f"use_embryo_flag=True: {use_mask.sum()} / {len(df_all)} rows")
else:
    print("⚠️  'use_embryo_flag' column not found, assuming all usable")

# Combined filter
ref_mask = genotype_mask & genetic_mask & chem_mask & use_mask
print(f"\n✓ Final reference set: {ref_mask.sum()} / {len(df_all)} rows ({100*ref_mask.sum()/len(df_all):.1f}%)")

df_ref = df_all[ref_mask].copy()

if len(df_ref) == 0:
    raise ValueError("No data passed filters! Check filter criteria.")

# Check required columns
if 'predicted_stage_hpf' not in df_ref.columns:
    raise ValueError("Missing 'predicted_stage_hpf' column!")
if 'total_length_um' not in df_ref.columns:
    raise ValueError("Missing 'total_length_um' column!")

# Drop NaN values
print(f"\nDropping NaN values...")
initial_len = len(df_ref)
df_ref = df_ref.dropna(subset=['predicted_stage_hpf', 'total_length_um'])
print(f"  {initial_len} → {len(df_ref)} rows ({initial_len - len(df_ref)} removed)")

# Calculate stage-binned statistics
print("\n" + "=" * 80)
print("CALCULATING STAGE-BINNED STATISTICS")
print("=" * 80)

# Create bins (0.5 hpf increments)
hpf_bins = np.arange(0, 130, 0.5)
hpf_centers = (hpf_bins[:-1] + hpf_bins[1:]) / 2

# Initialize arrays
p5_array = np.full(len(hpf_centers), np.nan)
p50_array = np.full(len(hpf_centers), np.nan)
p95_array = np.full(len(hpf_centers), np.nan)
n_array = np.zeros(len(hpf_centers), dtype=int)

# Calculate percentiles for each bin
print(f"Processing {len(hpf_centers)} bins (0.5 hpf increments)...")
for i, hpf_center in enumerate(hpf_centers):
    # Use ±0.25 hpf window for each bin
    bin_mask = (
        (df_ref['predicted_stage_hpf'] >= hpf_center - 0.25) &
        (df_ref['predicted_stage_hpf'] < hpf_center + 0.25)
    )

    n = bin_mask.sum()
    n_array[i] = n

    if n >= 5:  # Minimum sample size for stable percentiles
        length_vals = df_ref.loc[bin_mask, 'total_length_um']
        p5_array[i] = length_vals.quantile(0.05)
        p50_array[i] = length_vals.quantile(0.50)
        p95_array[i] = length_vals.quantile(0.95)

# Summary
n_valid_bins = (~np.isnan(p50_array)).sum()
print(f"\n✓ Valid bins: {n_valid_bins} / {len(hpf_centers)} ({100*n_valid_bins/len(hpf_centers):.1f}%)")
if n_valid_bins > 0:
    print(f"  Stage range: {hpf_centers[~np.isnan(p50_array)].min():.1f} - {hpf_centers[~np.isnan(p50_array)].max():.1f} hpf")

# Fill gaps with interpolation, then edges with extrapolation
print("\nFilling gaps and edge bins...")
valid_mask = ~np.isnan(p50_array)
if valid_mask.sum() > 0:
    first_valid = np.where(valid_mask)[0][0]
    last_valid = np.where(valid_mask)[0][-1]

    # Get valid data points for interpolation
    valid_indices = np.where(valid_mask)[0]
    valid_hpf = hpf_centers[valid_indices]

    # Interpolate missing values in the middle (linear)
    p5_array = np.interp(hpf_centers, valid_hpf, p5_array[valid_indices])
    p50_array = np.interp(hpf_centers, valid_hpf, p50_array[valid_indices])
    p95_array = np.interp(hpf_centers, valid_hpf, p95_array[valid_indices])

    n_before = first_valid
    n_after = len(hpf_centers) - last_valid - 1
    n_middle_gaps = len(hpf_centers) - valid_mask.sum() - n_before - n_after

    print(f"  Filled {n_before} bins at start (forward fill)")
    print(f"  Filled {n_middle_gaps} gaps in middle (linear interpolation)")
    print(f"  Filled {n_after} bins at end (backward fill)")

# Smooth with Savitzky-Golay filter
print("\nSmoothing with Savitzky-Golay filter (window=5, poly=2)...")
window = 5
poly = 2

try:
    p5_smooth = scipy.signal.savgol_filter(p5_array, window, poly)
    p50_smooth = scipy.signal.savgol_filter(p50_array, window, poly)
    p95_smooth = scipy.signal.savgol_filter(p95_array, window, poly)
    print("  ✓ Smoothing successful")
except Exception as e:
    print(f"  ⚠️  Smoothing failed: {e}")
    print("  Using unsmoothed data")
    p5_smooth = p5_array
    p50_smooth = p50_array
    p95_smooth = p95_array

# Save reference curves
print("\n" + "=" * 80)
print("SAVING REFERENCE CURVES")
print("=" * 80)

ref_curves = pd.DataFrame({
    'stage_hpf': hpf_centers,
    'p5': p5_smooth,
    'p50': p50_smooth,
    'p95': p95_smooth,
    'n': n_array
})

output_csv = OUTPUT_DIR / "length_reference_curves.csv"
ref_curves.to_csv(output_csv, index=False)
print(f"✓ Saved to: {output_csv}")
print(f"  Shape: {ref_curves.shape}")
print(f"\nFirst 10 rows:")
print(ref_curves.head(10).to_string(index=False))

# Plot reference curves
print("\n" + "=" * 80)
print("GENERATING REFERENCE PLOT")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Top panel: Reference curves with shaded band
ax = axes[0]
ax.plot(hpf_centers, p50_smooth, 'b-', linewidth=2, label='p50 (median)')
ax.fill_between(hpf_centers, p5_smooth, p95_smooth, alpha=0.3, color='blue', label='p5-p95 range')
ax.plot(hpf_centers, p5_smooth, 'b--', linewidth=1, alpha=0.5)
ax.plot(hpf_centers, p95_smooth, 'b--', linewidth=1, alpha=0.5)
ax.set_xlabel('Predicted Stage (hpf)', fontsize=12)
ax.set_ylabel('Total Length (µm)', fontsize=12)
ax.set_title('Total Length Reference Curves from Wildtype Controls', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Bottom panel: Sample size per bin
ax = axes[1]
ax.bar(hpf_centers, n_array, width=0.4, alpha=0.6, color='gray')
ax.set_xlabel('Predicted Stage (hpf)', fontsize=12)
ax.set_ylabel('Sample Size (n)', fontsize=12)
ax.set_title('Sample Size per 0.5 hpf Bin', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()

output_plot = OUTPUT_DIR / "length_reference_plot.png"
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {output_plot}")

# Plot individual trajectories with reference bands
print("\nGenerating individual trajectories plot...")
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# Plot individual wildtype trajectories (faded)
if 'embryo_id' in df_ref.columns:
    embryo_ids = df_ref['embryo_id'].unique()[:100]  # Limit to first 100 for visibility
    for embryo_id in embryo_ids:
        embryo_data = df_ref[df_ref['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')
        if len(embryo_data) > 1:
            ax.plot(embryo_data['predicted_stage_hpf'], embryo_data['total_length_um'],
                   alpha=0.1, linewidth=0.5, color='gray')

# Plot reference bands
ax.fill_between(hpf_centers, p5_smooth, p95_smooth, alpha=0.3, color='blue', label='p5-p95 range (reference)')
ax.plot(hpf_centers, p50_smooth, 'b-', linewidth=2.5, label='p50 (median)')
ax.plot(hpf_centers, p5_smooth, 'b--', linewidth=1.5, alpha=0.7, label='p5')
ax.plot(hpf_centers, p95_smooth, 'b--', linewidth=1.5, alpha=0.7, label='p95')

ax.set_xlabel('Predicted Stage (hpf)', fontsize=12)
ax.set_ylabel('Total Length (µm)', fontsize=12)
ax.set_title('Wildtype Total Length Trajectories with Reference Bounds', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

output_traj_plot = OUTPUT_DIR / "length_reference_with_trajectories.png"
plt.savefig(output_traj_plot, dpi=150, bbox_inches='tight')
print(f"✓ Saved trajectories plot to: {output_traj_plot}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nReference dataset:")
if 'embryo_id' in df_ref.columns:
    print(f"  Total embryos: {df_ref['embryo_id'].nunique()}")
print(f"  Total frames: {len(df_ref)}")
print(f"  Stage range: {df_ref['predicted_stage_hpf'].min():.1f} - {df_ref['predicted_stage_hpf'].max():.1f} hpf")
print(f"  Length range: {df_ref['total_length_um'].min():.1f} - {df_ref['total_length_um'].max():.1f} µm")

print(f"\nReference curves:")
print(f"  Valid bins: {n_valid_bins} / {len(hpf_centers)}")
print(f"  p5 range: {np.nanmin(p5_smooth):.1f} - {np.nanmax(p5_smooth):.1f} µm")
print(f"  p50 range: {np.nanmin(p50_smooth):.1f} - {np.nanmax(p50_smooth):.1f} µm")
print(f"  p95 range: {np.nanmin(p95_smooth):.1f} - {np.nanmax(p95_smooth):.1f} µm")

print("\n" + "=" * 80)
print("✓ DONE!")
print("=" * 80)
print(f"\nOutputs:")
print(f"  - {output_csv}")
print(f"  - {output_plot}")
print(f"  - {output_traj_plot}")
