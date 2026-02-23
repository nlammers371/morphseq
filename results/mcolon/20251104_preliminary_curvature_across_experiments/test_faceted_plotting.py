#!/usr/bin/env python3
"""
Test script for faceted plotting functionality.

Tests the new plot_embryos_metric_over_time_faceted() function with
multi-experiment data from the hierarchical clustering analysis.
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
OUTPUT_DIR = Path(__file__).parent / "output_test_faceted"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TESTING FACETED PLOTTING")
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
print(f"Genotypes: {df_all['genotype'].value_counts().to_dict()}")

# =============================================================================
# TEST 1: Basic faceted plot (3 experiments, color by genotype)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Basic Faceted Plot (3 panels, color by genotype)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        metric='normalized_baseline_deviation',
        time_col='predicted_stage_hpf',
        embryo_col='embryo_id',
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,
        trend_method='mean',  # Options: 'mean', 'median', 'dba', or None
        show_sd_band=False,
        smooth_window=None,  # No smoothing to see raw trend
        alpha_individual=0.2,
        alpha_trend=0.9,
        figsize_per_panel=(6, 5),
        facet_ncols=3,
        save_path=OUTPUT_DIR / "test1_basic_faceted.png"
    )
    print("✓ Test 1 PASSED")
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 2: Vertical layout (1 column)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Vertical Layout (1 column, stacked)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        metric='normalized_baseline_deviation',
        time_col='predicted_stage_hpf',
        embryo_col='embryo_id',
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=False,  # Only show trend for clarity
        trend_method='median',  # Using median for robustness
        show_sd_band=True,
        alpha_trend=0.9,
        figsize_per_panel=(10, 4),
        facet_ncols=1,
        save_path=OUTPUT_DIR / "test2_vertical_layout.png"
    )
    print("✓ Test 2 PASSED")
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 3: Auto grid layout (let function decide columns)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Auto Grid Layout (automatic column selection)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        metric='normalized_baseline_deviation',
        time_col='predicted_stage_hpf',
        embryo_col='embryo_id',
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,
        trend_method='dba',  # Using DBA for time-warped average
        alpha_individual=0.15,
        alpha_trend=0.95,
        smooth_window=None,
        figsize_per_panel=(5, 4),
        facet_ncols=None,  # Auto-select
        title="Multi-Experiment Genotype Comparison (DBA trend)",
        save_path=OUTPUT_DIR / "test3_auto_layout.png"
    )
    print("✓ Test 3 PASSED")
except Exception as e:
    print(f"✗ Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 4: Shared axes OFF (independent scales per panel)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: Independent Axes (facet_sharex=False, facet_sharey=False)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        metric='normalized_baseline_deviation',
        time_col='predicted_stage_hpf',
        embryo_col='embryo_id',
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,
        trend_method='mean',
        facet_ncols=3,
        facet_sharex=False,  # Independent x-axis
        facet_sharey=False,  # Independent y-axis
        save_path=OUTPUT_DIR / "test4_independent_axes.png"
    )
    print("✓ Test 4 PASSED")
except Exception as e:
    print(f"✗ Test 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 5: No smoothing (raw DBA mean)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: No Smoothing (smooth_window=None)")
print("=" * 80)

try:
    fig = plot_embryos_metric_over_time_faceted(
        df_all,
        metric='normalized_baseline_deviation',
        time_col='predicted_stage_hpf',
        embryo_col='embryo_id',
        color_by='genotype',
        facet_by='source_experiment',
        show_individual=True,  # Show individuals only
        trend_method=None,  # NO TREND LINE
        show_sd_band=False,
        smooth_window=None,
        alpha_individual=0.3,
        alpha_trend=0.9,
        figsize_per_panel=(6, 5),
        facet_ncols=3,
        title="Multi-Experiment Comparison (No Smoothing)",
        save_path=OUTPUT_DIR / "test5_no_smoothing.png"
    )
    print("✓ Test 5 PASSED")
except Exception as e:
    print(f"✗ Test 5 FAILED: {e}")
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
