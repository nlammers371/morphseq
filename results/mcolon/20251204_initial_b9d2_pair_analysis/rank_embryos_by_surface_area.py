#!/usr/bin/env python3
"""
Rank embryos by average surface area between 30-40 hpf.
"""

import pandas as pd
import sys

sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')
from src.analyze.trajectory_analysis.data_loading import _load_df03_format

# Load data from experiment 20251119
df = _load_df03_format('20251119')

# Handle column collision
if 'surface_area_um_y' in df.columns:
    df['surface_area_um'] = df['surface_area_um_y']

# Filter for 30-40 hpf range
df_filtered = df[(df['predicted_stage_hpf'] >= 20) & (df['predicted_stage_hpf'] <= 30)].copy()

# Drop rows with missing surface area
df_filtered = df_filtered.dropna(subset=['surface_area_um', 'embryo_id'])

# Calculate average surface area per embryo
embryo_avg = df_filtered.groupby('embryo_id')['surface_area_um'].mean().reset_index()
embryo_avg.columns = ['embryo_id', 'avg_surface_area_um']

# Sort by average surface area (descending)
embryo_avg = embryo_avg.sort_values('avg_surface_area_um', ascending=False).reset_index(drop=True)

# Add rank
embryo_avg.insert(0, 'rank', range(1, len(embryo_avg) + 1))

print("\n" + "=" * 80)
print("EMBRYOS RANKED BY AVERAGE SURFACE AREA (20-30 hpf)")
print("=" * 80)
print(f"\nTotal embryos: {len(embryo_avg)}")
print(f"Average surface area range: {embryo_avg['avg_surface_area_um'].min():.1f} - {embryo_avg['avg_surface_area_um'].max():.1f} µm²\n")

print(embryo_avg.to_string(index=False))

# Save to CSV
output_path = '/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251204_initial_b9d2_pair_analysis/embryo_rankings_surface_area_20_30hpf.csv'
embryo_avg.to_csv(output_path, index=False)
print(f"\n✓ Saved to: {output_path}")
