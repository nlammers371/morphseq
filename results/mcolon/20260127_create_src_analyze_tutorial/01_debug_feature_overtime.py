"""
Debug: Trend line not extending back to earliest timepoint.

This script compares the earliest time in trajectories vs the earliest
trend bin center to diagnose binning/centering gaps.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.analyze.utils.data_processing import get_trajectories_for_group
from src.analyze.utils.stats import compute_trend_line
from src.analyze.trajectory_analysis.viz.styling import get_color_for_genotype
from src.analyze.viz.plotting import plot_feature_over_time

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading experiment data...")
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build06_output'

df1 = pd.read_csv(meta_dir / 'df03_final_output_with_latents_20251121.csv')
df2 = pd.read_csv(meta_dir / 'df03_final_output_with_latents_20251125.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Filter to valid embryos
df = df[df['use_embryo_flag']].copy()

# Choose target genotype
all_genotypes = list(df['genotype'].dropna().unique())
unknown_candidates = [g for g in all_genotypes if 'unknown' in str(g).lower()]
if unknown_candidates:
    target_genotype = unknown_candidates[0]
else:
    target_genotype = all_genotypes[0] if all_genotypes else None

if target_genotype is None:
    raise RuntimeError("No genotypes found in dataframe")

print(f"Target genotype: {target_genotype}")

feature = 'baseline_deviation_normalized'
time_col = 'predicted_stage_hpf'
id_col = 'embryo_id'
bin_width = 0.5

# Filter to target genotype for clarity
sub_df = df[df['genotype'] == target_genotype].copy()

# Raw data diagnostics
raw_min_time = float(sub_df[time_col].min())
raw_non_nan = sub_df[[time_col, feature]].dropna()
raw_non_nan_min_time = float(raw_non_nan[time_col].min()) if len(raw_non_nan) else None
print(f"Raw min time (any): {raw_min_time}")
print(f"Raw min time (non-NaN feature): {raw_non_nan_min_time}")
print(f"NaNs in {feature}: {sub_df[feature].isna().sum()} / {len(sub_df)}")
print(f"NaNs in {time_col}: {sub_df[time_col].isna().sum()} / {len(sub_df)}")

# Bin occupancy diagnostics (raw, non-NaN)
if len(raw_non_nan):
    t_min = np.floor(raw_non_nan[time_col].min())
    t_max = np.ceil(raw_non_nan[time_col].max())
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    first_non_empty = None
    first_non_nan_stat = None
    for i in range(len(bins) - 1):
        mask = (raw_non_nan[time_col] >= bins[i]) & (raw_non_nan[time_col] < bins[i + 1])
        vals = raw_non_nan.loc[mask, feature].to_numpy()
        if vals.size > 0 and first_non_empty is None:
            first_non_empty = (bins[i], bins[i + 1])
        if vals.size > 0 and np.isfinite(np.nanmedian(vals)):
            first_non_nan_stat = (bins[i], bins[i + 1])
            break
    print(f"First non-empty raw bin: {first_non_empty}")
    print(f"First bin with finite stat (raw): {first_non_nan_stat}")
trajectories, _, _ = get_trajectories_for_group(
    sub_df, {},
    time_col=time_col, metric_col=feature, embryo_id_col=id_col,
    smooth_method='gaussian', smooth_params={'sigma': 1.5},
)
if trajectories:
    all_times = np.concatenate([t['times'] for t in trajectories])
    all_metrics = np.concatenate([t['metrics'] for t in trajectories])
    trend_t_old, _ = compute_trend_line(all_times, all_metrics, bin_width=bin_width)
    traj_min_time = float(all_times.min())
    trend_old_min = float(min(trend_t_old)) if trend_t_old else None
    bin_start = np.floor(traj_min_time)
    bin_center = bin_start + (bin_width / 2)
    traj_nan_count = int(np.isnan(all_metrics).sum())
    traj_non_nan_min_time = float(all_times[~np.isnan(all_metrics)].min()) if np.any(~np.isnan(all_metrics)) else None
else:
    traj_min_time = None
    trend_old_min = None
    bin_start = None
    bin_center = None
    traj_nan_count = None
    traj_non_nan_min_time = None

print(f"Traj min time: {traj_min_time}")
print(f"Traj min time (non-NaN): {traj_non_nan_min_time}")
print(f"NaNs in trajectory metrics: {traj_nan_count}")
print(f"First bin start (floor): {bin_start}")
print(f"First bin center: {bin_center}")
print(f"Trend min time: {trend_old_min}")

# Render a plot for visual inspection
color_lookup = {gt: get_color_for_genotype(gt) for gt in all_genotypes}
figs = plot_feature_over_time(
    sub_df,
    feature=feature,
    color_by='genotype',
    color_lookup=color_lookup,
    backend='both',
    show_individual=True,
    show_error_band=False,
)

figs['plotly'].write_html(FIGURES_DIR / "01_debug_feature_overtime.html")
figs['matplotlib'].savefig(FIGURES_DIR / "01_debug_feature_overtime.png", dpi=300, bbox_inches='tight')

print(f"Saved debug figures to: {FIGURES_DIR}")
