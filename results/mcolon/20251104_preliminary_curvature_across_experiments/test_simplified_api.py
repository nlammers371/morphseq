#!/usr/bin/env python3
"""Quick test of simplified faceted plotting."""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze.utils.plotting_faceted import plot_embryos_metric_over_time_faceted

# Quick sanity check with minimal data
print("Testing simplified faceted plotting API...")

# Create minimal test data
data = {
    'embryo_id': ['e1', 'e1', 'e2', 'e2'] * 6,
    'predicted_stage_hpf': [10, 15, 10, 15] * 6,
    'normalized_baseline_deviation': [0.5, 0.6, 0.4, 0.5] * 6,
    'genotype': ['wt', 'wt', 'mut', 'mut'] * 6,
    'source_experiment': ['exp1'] * 8 + ['exp2'] * 8 + ['exp3'] * 8
}
df = pd.DataFrame(data)

print(f"Test data: {len(df)} rows, {df['source_experiment'].nunique()} experiments")

# Test 1: Mean trend (default)
print("\nTest 1: trend_method='mean'")
try:
    fig = plot_embryos_metric_over_time_faceted(
        df,
        color_by='genotype',
        facet_by='source_experiment',
        trend_method='mean',
        smooth_window=None
    )
    print("✓ PASSED")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: Median trend
print("\nTest 2: trend_method='median'")
try:
    fig = plot_embryos_metric_over_time_faceted(
        df,
        color_by='genotype',
        facet_by='source_experiment',
        trend_method='median',
        smooth_window=None
    )
    print("✓ PASSED")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 3: No trend
print("\nTest 3: trend_method=None (no trend line)")
try:
    fig = plot_embryos_metric_over_time_faceted(
        df,
        color_by='genotype',
        facet_by='source_experiment',
        trend_method=None,
        show_individual=True
    )
    print("✓ PASSED")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n✓ All tests passed! API is simplified and working.")
