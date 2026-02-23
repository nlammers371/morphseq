"""
Build SA reference curves from all build04 data.

Aggregates surface area distributions across all experiments to establish
normal ranges (p5, p50, p95) vs predicted_stage_hpf.

Usage:
    conda activate segmentations_grounded_sam
    python build_sa_reference.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import scipy.signal

# Paths
BUILD04_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output")
# Output to metadata/ for production use
OUTPUT_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BUILDING SA REFERENCE CURVES FROM ALL BUILD04 DATA")
print("=" * 80)

# Find all build04 CSV files
csv_files = sorted(glob(str(BUILD04_DIR / "qc_staged_*.csv")))
print(f"\nFound {len(csv_files)} build04 CSV files")

# Load and concatenate all data
all_data = []
for csv_file in csv_files:
    exp_id = Path(csv_file).stem.replace("qc_staged_", "")
    print(f"Loading {exp_id}...", end=" ")

    try:
        df = pd.read_csv(csv_file)
        df['experiment_id_loaded'] = exp_id
        all_data.append(df)
        print(f"✓ ({len(df)} rows)")
    except Exception as e:
        print(f"✗ Error: {e}")

df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal loaded: {len(df_all)} rows from {df_all['experiment_id_loaded'].nunique()} experiments")

# Filter for true wild-type controls
print("\n" + "=" * 80)
print("FILTERING FOR TRUE WILD-TYPE CONTROLS")
print("=" * 80)

# Check what columns are available
print(f"\nAvailable columns: {list(df_all.columns)[:20]}...")  # Show first 20

# Build filter
wt_genotypes = ['wik', 'ab', 'wik-ab', 'wik/ab', 'AB', 'WIK']
if 'genotype' in df_all.columns:
    genotype_mask = df_all['genotype'].astype(str).isin(wt_genotypes)
    print(f"Genotype filter: {genotype_mask.sum()} / {len(df_all)} rows")
else:
    print("⚠️  'genotype' column not found, checking 'phenotype'...")
    genotype_mask = pd.Series([False] * len(df_all))

# Check phenotype as alternative
if 'phenotype' in df_all.columns:
    phenotype_mask = df_all['phenotype'].astype(str).str.lower() == 'wt'
    print(f"Phenotype='wt' filter: {phenotype_mask.sum()} / {len(df_all)} rows")
    genotype_mask = genotype_mask | phenotype_mask

# No chemical perturbation
if 'chem_perturbation' in df_all.columns:
    chem_mask = (df_all['chem_perturbation'].astype(str) == 'None') | df_all['chem_perturbation'].isna()
    print(f"No chem perturbation: {chem_mask.sum()} / {len(df_all)} rows")
else:
    print("⚠️  'chem_perturbation' column not found, skipping filter")
    chem_mask = pd.Series([True] * len(df_all))

# Use embryo flag
if 'use_embryo_flag' in df_all.columns:
    use_mask = df_all['use_embryo_flag'].astype(bool)
    print(f"use_embryo_flag=True: {use_mask.sum()} / {len(df_all)} rows")
else:
    print("⚠️  'use_embryo_flag' column not found, assuming all usable")
    use_mask = pd.Series([True] * len(df_all))

# Control flag alternative
if 'control_flag' in df_all.columns:
    control_mask = df_all['control_flag'].astype(bool)
    print(f"control_flag=True: {control_mask.sum()} / {len(df_all)} rows")
    genotype_mask = genotype_mask | control_mask

# Combined filter
ref_mask = genotype_mask & chem_mask & use_mask
print(f"\n✓ Final reference set: {ref_mask.sum()} / {len(df_all)} rows ({100*ref_mask.sum()/len(df_all):.1f}%)")

df_ref = df_all[ref_mask].copy()

# Check required columns
if 'predicted_stage_hpf' not in df_ref.columns:
    raise ValueError("Missing 'predicted_stage_hpf' column!")
if 'surface_area_um' not in df_ref.columns:
    raise ValueError("Missing 'surface_area_um' column!")

# Drop NaN values
print(f"\nDropping NaN values...")
initial_len = len(df_ref)
df_ref = df_ref.dropna(subset=['predicted_stage_hpf', 'surface_area_um'])
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
        sa_vals = df_ref.loc[bin_mask, 'surface_area_um']
        p5_array[i] = sa_vals.quantile(0.05)
        p50_array[i] = sa_vals.quantile(0.50)
        p95_array[i] = sa_vals.quantile(0.95)

# Summary
n_valid_bins = (~np.isnan(p50_array)).sum()
print(f"\n✓ Valid bins: {n_valid_bins} / {len(hpf_centers)} ({100*n_valid_bins/len(hpf_centers):.1f}%)")
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

output_csv = OUTPUT_DIR / "sa_reference_curves.csv"
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
ax.plot(hpf_centers, p50_smooth / 1e6, 'b-', linewidth=2, label='p50 (median)')
ax.fill_between(hpf_centers, p5_smooth / 1e6, p95_smooth / 1e6, alpha=0.3, color='blue', label='p5-p95 range')
ax.plot(hpf_centers, p5_smooth / 1e6, 'b--', linewidth=1, alpha=0.5)
ax.plot(hpf_centers, p95_smooth / 1e6, 'b--', linewidth=1, alpha=0.5)
ax.set_xlabel('Predicted Stage (hpf)', fontsize=12)
ax.set_ylabel('Surface Area (mm²)', fontsize=12)
ax.set_title('SA Reference Curves from All Build04 Data (Wild-Type Controls)', fontsize=14, fontweight='bold')
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

output_plot = OUTPUT_DIR / "reference_plot.png"
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {output_plot}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nReference dataset:")
print(f"  Total embryos: {df_ref['embryo_id'].nunique() if 'embryo_id' in df_ref.columns else 'N/A'}")
print(f"  Total frames: {len(df_ref)}")
print(f"  Stage range: {df_ref['predicted_stage_hpf'].min():.1f} - {df_ref['predicted_stage_hpf'].max():.1f} hpf")
print(f"  SA range: {df_ref['surface_area_um'].min():.0f} - {df_ref['surface_area_um'].max():.0f} µm")

print(f"\nReference curves:")
print(f"  Valid bins: {n_valid_bins} / {len(hpf_centers)}")
print(f"  p5 range: {np.nanmin(p5_smooth):.0f} - {np.nanmax(p5_smooth):.0f} µm")
print(f"  p50 range: {np.nanmin(p50_smooth):.0f} - {np.nanmax(p50_smooth):.0f} µm")
print(f"  p95 range: {np.nanmin(p95_smooth):.0f} - {np.nanmax(p95_smooth):.0f} µm")

print("\n✓ Done!")
