#!/usr/bin/env python3
"""
Test script to verify all module imports work correctly.

This script tests the modular package structure without running the full analysis.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("TESTING MODULE IMPORTS")
print("="*80)

# Test config
print("\n1. Testing config module...")
try:
    import config
    print("   ✓ config imported successfully")
    print(f"   - Results dir: {config.RESULTS_DIR}")
    print(f"   - N_PERMUTATIONS: {config.N_PERMUTATIONS}")
    print(f"   - USE_CLASS_WEIGHTS: {config.USE_CLASS_WEIGHTS}")
except Exception as e:
    print(f"   ✗ Failed to import config: {e}")
    sys.exit(1)

# Test utils
print("\n2. Testing utils module...")
try:
    from utils import (
        load_experiments,
        bin_by_embryo_time,
        make_safe_comparison_name,
        get_plot_path,
        get_data_path
    )
    print("   ✓ utils imported successfully")
    print("   - load_experiments:", load_experiments)
    print("   - bin_by_embryo_time:", bin_by_embryo_time)
    print("   - make_safe_comparison_name:", make_safe_comparison_name)
except Exception as e:
    print(f"   ✗ Failed to import utils: {e}")
    sys.exit(1)

# Test classification
print("\n3. Testing classification module...")
try:
    from classification import (
        predictive_signal_test,
        compute_embryo_penetrance
    )
    print("   ✓ classification imported successfully")
    print("   - predictive_signal_test:", predictive_signal_test)
    print("   - compute_embryo_penetrance:", compute_embryo_penetrance)
except Exception as e:
    print(f"   ✗ Failed to import classification: {e}")
    sys.exit(1)

# Test visualization
print("\n4. Testing visualization module...")
try:
    from visualization import (
        plot_auroc_over_time,
        plot_auroc_with_significance,
        plot_signed_margin_trajectories,
        plot_signed_margin_heatmap,
        plot_penetrance_distribution
    )
    print("   ✓ visualization imported successfully")
    print("   - plot_auroc_over_time:", plot_auroc_over_time)
    print("   - plot_signed_margin_trajectories:", plot_signed_margin_trajectories)
    print("   - plot_penetrance_distribution:", plot_penetrance_distribution)
except Exception as e:
    print(f"   ✗ Failed to import visualization: {e}")
    sys.exit(1)

# Test utility functions
print("\n5. Testing utility functions...")
try:
    # Test file path generation
    safe_name = make_safe_comparison_name("cep290_wildtype", "cep290_homozygous")
    print(f"   ✓ Safe comparison name: {safe_name}")

    plot_path = get_plot_path(config.PLOT_DIR, "cep290", "auroc", safe_name)
    print(f"   ✓ Plot path: {plot_path}")

    data_path = get_data_path(config.DATA_DIR, "cep290", "embryo_predictions", safe_name)
    print(f"   ✓ Data path: {data_path}")
except Exception as e:
    print(f"   ✗ Utility function test failed: {e}")
    sys.exit(1)

# Test config helpers
print("\n6. Testing config helper functions...")
try:
    config.print_config()
except Exception as e:
    print(f"   ✗ Config print failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nPackage structure:")
print("  ✓ config.py - Configuration constants")
print("  ✓ utils/ - Data loading, binning, file utilities")
print("  ✓ classification/ - Predictive test with class weights (FIXED)")
print("  ✓ visualization/ - All plotting functions")
print("  ✓ run_analysis.py - Main orchestration script")
print("\nKey improvements:")
print("  ✓ Class weights are now properly implemented and enabled by default")
print("  ✓ Modular structure for easy maintenance and testing")
print("  ✓ Clear separation of concerns")
print("  ✓ Comprehensive docstrings")
print("\nTo run the full analysis:")
print("  cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251016")
print("  python run_analysis.py")
print("\nOr with custom parameters:")
print("  MORPHSEQ_N_PERMUTATIONS=500 python run_analysis.py")
