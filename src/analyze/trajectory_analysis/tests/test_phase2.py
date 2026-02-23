#!/usr/bin/env python3
"""
Phase 2 Import Tests

Tests that all Phase 2 subpackages (distance, utilities, io) import correctly.
Includes runtime import checks to catch lazy/deferred imports that might fail.
"""

import sys
import os
from pathlib import Path

# Determine repo root: prefer env var, fallback to __file__-based derivation
# This file is at: <repo>/src/analyze/trajectory_analysis/tests/test_phase2.py
REPO_ROOT = os.environ.get('MORPHSEQ_REPO_ROOT')
if not REPO_ROOT:
    REPO_ROOT = str(Path(__file__).resolve().parents[4])

SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def test_distance_imports():
    """Test distance subpackage imports"""
    print("Testing distance imports...")
    from analyze.trajectory_analysis.distance import (
        compute_dtw_distance,
        compute_dtw_distance_matrix,
        prepare_multivariate_array,
        compute_md_dtw_distance_matrix,
        compute_trajectory_distances,
        dba,
    )
    print("✓ Distance imports OK")
    return True

def test_utilities_imports():
    """Test utilities subpackage imports"""
    print("Testing utilities imports...")
    from analyze.trajectory_analysis.utilities import (
        extract_trajectories_df,
        interpolate_to_common_grid_df,
        interpolate_to_common_grid_multi_df,
        compute_trend_line,
        fit_pca_on_embeddings,
        transform_embeddings_to_pca,
        test_anticorrelation,
    )
    print("✓ Utilities imports OK")
    return True

def test_io_imports():
    """Test io subpackage imports"""
    print("Testing io imports...")
    from analyze.trajectory_analysis.io import (
        load_experiment_dataframe,
        extract_trajectory_dataframe,
        compute_dtw_distance_from_df,
        load_phenotype_file,
        save_phenotype_file,
    )
    print("✓ I/O imports OK")
    return True

def test_cross_imports():
    """Test that io correctly imports from distance"""
    print("Testing cross-subpackage imports...")
    # This import chain: io.data_loading -> distance
    from analyze.trajectory_analysis.io.data_loading import compute_dtw_distance_from_df
    print("✓ Cross-subpackage imports OK")
    return True

def test_runtime_imports():
    """Test runtime/deferred imports that could fail at call time"""
    print("Testing runtime imports (deferred imports in functions)...")

    # Test prepare_multivariate_array's deferred import
    # It imports from ..utilities.trajectory_utils inside the function
    import numpy as np
    import pandas as pd
    from analyze.trajectory_analysis.distance import prepare_multivariate_array

    # Create minimal test data
    test_df = pd.DataFrame({
        'embryo_id': ['e1', 'e1', 'e2', 'e2'],
        'hpf': [1.0, 2.0, 1.0, 2.0],
        'metric1': [0.1, 0.2, 0.3, 0.4],
        'metric2': [0.5, 0.6, 0.7, 0.8],
    })

    # This will trigger the deferred import
    try:
        arr, ids, grid = prepare_multivariate_array(
            test_df,
            metrics=['metric1', 'metric2'],
            embryo_id_col='embryo_id',
            time_col='hpf',
            verbose=False
        )
        print("  ✓ prepare_multivariate_array runtime imports OK")
    except ImportError as e:
        print(f"  ✗ prepare_multivariate_array import failed: {e}")
        return False

    # Test pair_analysis.data_utils compute_binned_mean deferred import
    import warnings
    from analyze.trajectory_analysis.pair_analysis.data_utils import compute_binned_mean

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = compute_binned_mean(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.1, 0.2, 0.3]),
                bin_width=1.0
            )
        print("  ✓ compute_binned_mean runtime imports OK")
    except ImportError as e:
        print(f"  ✗ compute_binned_mean import failed: {e}")
        return False

    print("✓ Runtime imports OK")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 2 Import Tests")
    print("=" * 60)

    results = []
    results.append(("distance", test_distance_imports()))
    results.append(("utilities", test_utilities_imports()))
    results.append(("io", test_io_imports()))
    results.append(("cross-imports", test_cross_imports()))
    results.append(("runtime-imports", test_runtime_imports()))

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All Phase 2 tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)
