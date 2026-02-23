#!/usr/bin/env python3
"""
Test script for SIMPLIFIED faceted plotting functionality.

Tests different trend_method options: 'mean', 'median', 'dba', None
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze.utils.plotting_faceted import plot_embryos_metric_over_time_faceted

# Configuration
EXPERIMENTS = ['20251017', '20250305', '20250417']
CURV_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "body_axis" / "summary"
META_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
OUTPUT_DIR = Path(__file__).parent / "output_test_trend_methods"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TESTING TREND METHODS")
print("=" * 80)

# Load experiment data (simplified version)
def load_experiment_data(experiment_id):
    """Load and merge curvature metrics with metadata."""
    curv_file = CURV_DIR / f"curvature_metrics_{experiment_id}.csv"
    meta_file = META_DIR / f"df03_final_output_with_latents_{experiment_id}.csv"
    
    if not curv_file.exists() or not meta_file.exists():
        return None
    
    curv_df = pd.read_csv(curv_file)
    meta_df = pd.read_csv(meta_file)
    
    required_cols = ['snip_id', 'embryo_id', 'genotype', 'predicted_stage_hpf', 'baseline_deviation_normalized']
    meta_subset = meta_df[required_cols].copy()
    
    df = curv_df.merge(meta_subset, on='snip_id', how='inner')
    df['normalized_baseline_deviation'] = df['baseline_deviation_normalized']
    df['source_experiment'] = experiment_id
    
    return df

# Load all experiments
print("\nLoading data...")
dfs = []
for exp_id in EXPERIMENTS:
    print(f"  Loading {exp_id}...", end=" ")
    df = load_experiment_data(exp_id)
    if df is not None:
        dfs.append(df)
        print(f"✓ ({len(df)} rows)")
    else:
        print("✗ (missing files)")

if not dfs:
    print("\nERROR: No data loaded!")
    sys.exit(1)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df_all)} rows from {len(dfs)} experiments")

# =============================================================================
# TEST 1: Mean trend (no smoothing) - SIMPLEST
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Mean Trend (no smoothing)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,
        trend_method='mean',  # ← Simple mean
        smooth_window=None,   # ← No smoothing
        facet_ncols=3,
        save_path=OUTPUT_DIR / "test1_mean_no_smooth.png"
    )
    print("✓ Test 1 PASSED")
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 2: Median trend (robust to outliers)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Median Trend (robust to outliers)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,
        trend_method='median',  # ← Median
        smooth_window=None,
        facet_ncols=3,
        save_path=OUTPUT_DIR / "test2_median_no_smooth.png"
    )
    print("✓ Test 2 PASSED")
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 3: DBA trend (time-warped)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: DBA Trend (time-warped averaging)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,
        trend_method='dba',    # ← DTW Barycenter Averaging
        smooth_window=None,    # ← Let DBA handle it
        facet_ncols=3,
        save_path=OUTPUT_DIR / "test3_dba_no_smooth.png"
    )
    print("✓ Test 3 PASSED")
except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 4: NO trend line (only individual trajectories)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: No Trend Line (only individuals)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,
        trend_method=None,     # ← No trend!
        facet_ncols=3,
        save_path=OUTPUT_DIR / "test4_no_trend.png"
    )
    print("✓ Test 4 PASSED")
except Exception as e:
    print(f"✗ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 5: Mean with light smoothing
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: Mean with Light Smoothing (window=3)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,
        trend_method='mean',
        smooth_window=3,       # ← Light smoothing
        facet_ncols=3,
        save_path=OUTPUT_DIR / "test5_mean_smooth3.png"
    )
    print("✓ Test 5 PASSED")
except Exception as e:
    print(f"✗ Test 5 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 6: Mean with SD bands
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: Mean with SD Bands")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=False,  # ← Hide individuals for clarity
        trend_method='mean',
        show_sd_band=True,      # ← Show uncertainty
        smooth_window=None,
        facet_ncols=3,
        save_path=OUTPUT_DIR / "test6_mean_with_sd.png"
    )
    print("✓ Test 6 PASSED")
except Exception as e:
    print(f"✗ Test 6 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)
print(f"\nOutput saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for png_file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  - {png_file.name}")

print("\n" + "=" * 80)
print("TREND METHOD COMPARISON:")
print("=" * 80)
print("test1_mean_no_smooth.png   - Standard arithmetic mean (recommended)")
print("test2_median_no_smooth.png - Median (robust to outliers)")
print("test3_dba_no_smooth.png    - DTW Barycenter (time-warped, slow)")
print("test4_no_trend.png         - No trend line (raw data only)")
print("test5_mean_smooth3.png     - Mean with rolling window smoothing")
print("test6_mean_with_sd.png     - Mean with uncertainty bands")
