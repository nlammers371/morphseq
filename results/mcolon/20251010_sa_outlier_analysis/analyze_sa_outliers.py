"""
Analyze surface area outlier detection issues in build04.

Investigates why embryos like 20250711_F06_e01 and 20250711_H07_e01
were not flagged as SA outliers despite having abnormal surface areas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output/qc_staged_20250711.csv"
df = pd.read_csv(data_path)

print("=" * 80)
print("SURFACE AREA OUTLIER ANALYSIS")
print("=" * 80)

# Filter for reference embryos (controls)
ref_mask = (
    ((df['phenotype'].astype(str).str.lower() == 'wt') |
     (df['control_flag'].astype(bool))) &
    (df['use_embryo_flag'].astype(bool))
)

print(f"\nTotal rows: {len(df)}")
print(f"Reference embryos (wt or control & usable): {ref_mask.sum()}")
print(f"SA outlier flags: {df['sa_outlier_flag'].sum()}")

# Analyze SA distribution by predicted_stage_hpf bins
print("\n" + "=" * 80)
print("SURFACE AREA BY PREDICTED STAGE (HPF) - REFERENCE EMBRYOS")
print("=" * 80)

# Create bins
hpf_bins = np.arange(0, 130, 5)  # 5 hpf bins
df_ref = df[ref_mask].copy()
df_ref['stage_bin'] = pd.cut(df_ref['predicted_stage_hpf'], bins=hpf_bins)

# Calculate percentiles for each bin
bin_stats = df_ref.groupby('stage_bin')['surface_area_um'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('p5', lambda x: x.quantile(0.05)),
    ('p95', lambda x: x.quantile(0.95)),
    ('max', 'max')
]).round(1)

print("\nReference SA statistics by 5 hpf bins:")
print(bin_stats.to_string())

# Now check our problem embryos against these thresholds
print("\n" + "=" * 80)
print("PROBLEM EMBRYO ANALYSIS")
print("=" * 80)

problem_embryos = ['20250711_F06_e01', '20250711_H07_e01']

for emb_id in problem_embryos:
    emb_data = df[df['embryo_id'] == emb_id].copy()

    if len(emb_data) == 0:
        continue

    print(f"\n{emb_id}:")
    print(f"  Frames: {len(emb_data)}")
    print(f"  SA range: {emb_data['surface_area_um'].min():.0f} - {emb_data['surface_area_um'].max():.0f} µm")
    print(f"  Predicted stage range: {emb_data['predicted_stage_hpf'].min():.1f} - {emb_data['predicted_stage_hpf'].max():.1f} hpf")
    print(f"  SA outlier flags: {emb_data['sa_outlier_flag'].sum()} / {len(emb_data)}")

    # Compare to reference at matching stages
    emb_data['stage_bin'] = pd.cut(emb_data['predicted_stage_hpf'], bins=hpf_bins)

    print(f"\n  Sample frames compared to reference p95:")
    for idx, row in emb_data.iloc[::20].iterrows():  # Sample every 20th frame
        stage = row['predicted_stage_hpf']
        sa = row['surface_area_um']

        # Get reference p95 for this stage
        stage_ref = df_ref[
            (df_ref['predicted_stage_hpf'] >= stage - 2.5) &
            (df_ref['predicted_stage_hpf'] <= stage + 2.5)
        ]

        if len(stage_ref) > 0:
            ref_p95 = stage_ref['surface_area_um'].quantile(0.95)
            ratio = sa / ref_p95
            print(f"    Frame {row['frame_index']:3d}, stage {stage:5.1f} hpf: SA={sa:10.0f}, ref_p95={ref_p95:10.0f}, ratio={ratio:.2f}x")

# Analyze how many embryos should have been flagged
print("\n" + "=" * 80)
print("OUTLIER DETECTION SIMULATION")
print("=" * 80)

# Simulate what SHOULD happen with proper thresholds
margin_k_values = [1.2, 1.4, 1.6, 2.0]

for margin_k in margin_k_values:
    flagged_count = 0

    # For each embryo, check if ANY frame exceeds margin_k * p95
    for emb_id in df['embryo_id'].unique():
        emb_data = df[df['embryo_id'] == emb_id]

        should_flag = False
        for _, row in emb_data.iterrows():
            stage = row['predicted_stage_hpf']
            sa = row['surface_area_um']

            # Get reference p95 for this stage
            stage_ref = df_ref[
                (df_ref['predicted_stage_hpf'] >= stage - 0.75) &
                (df_ref['predicted_stage_hpf'] <= stage + 0.75)
            ]

            if len(stage_ref) >= 2:
                ref_p95 = stage_ref['surface_area_um'].quantile(0.95)
                threshold = margin_k * ref_p95

                if sa > threshold:
                    should_flag = True
                    break

        if should_flag:
            flagged_count += 1

    print(f"\nmargin_k = {margin_k}:")
    print(f"  Embryos that would be flagged: {flagged_count} / {df['embryo_id'].nunique()}")
    print(f"  Percentage: {100 * flagged_count / df['embryo_id'].nunique():.1f}%")

# Save summary
output_file = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/tests/sa_outlier_analysis/summary.txt")
with open(output_file, 'w') as f:
    f.write("SURFACE AREA OUTLIER ANALYSIS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset: 20250711\n")
    f.write(f"Total embryos: {df['embryo_id'].nunique()}\n")
    f.write(f"Total frames: {len(df)}\n")
    f.write(f"Current SA outlier flags: {df['sa_outlier_flag'].sum()}\n\n")

    f.write("Problem embryos:\n")
    for emb_id in problem_embryos:
        emb_data = df[df['embryo_id'] == emb_id]
        if len(emb_data) > 0:
            f.write(f"  {emb_id}: SA range {emb_data['surface_area_um'].min():.0f} - {emb_data['surface_area_um'].max():.0f} µm\n")

print(f"\n\nSummary saved to: {output_file}")
