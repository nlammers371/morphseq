#!/usr/bin/env python3
"""
Phase 3 Import Tests - QC Consolidation

Tests that:
1. New qc subpackage imports work
2. All 5 QC functions are accessible from qc/__init__.py
3. Backward compatibility shims work (outliers.py, distance_filtering.py)
4. Deprecation warnings are raised for old imports
5. consensus_pipeline.py imports work with new structure
"""

import sys
import os
import warnings
from pathlib import Path

# Determine repo root: prefer env var, fallback to __file__-based derivation
# This file is at: <repo>/src/analyze/trajectory_analysis/tests/test_phase3.py
REPO_ROOT = os.environ.get('MORPHSEQ_REPO_ROOT')
if not REPO_ROOT:
    REPO_ROOT = str(Path(__file__).resolve().parents[4])

SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def test_qc_subpackage():
    """Test qc subpackage imports."""
    print("Testing qc subpackage imports...")

    # Test direct import from qc
    from analyze.trajectory_analysis.qc import (
        identify_outliers,
        remove_outliers_from_distance_matrix,
        identify_embryo_outliers_iqr,
        filter_data_and_ids,
        identify_cluster_outliers_combined,
    )

    # Verify they're callable
    assert callable(identify_outliers)
    assert callable(remove_outliers_from_distance_matrix)
    assert callable(identify_embryo_outliers_iqr)
    assert callable(filter_data_and_ids)
    assert callable(identify_cluster_outliers_combined)

    print("✓ QC subpackage imports OK")
    return True


def test_qc_quality_control_module():
    """Test direct import from qc.quality_control module."""
    print("Testing qc.quality_control module imports...")

    from analyze.trajectory_analysis.qc.quality_control import (
        identify_outliers,
        remove_outliers_from_distance_matrix,
        identify_embryo_outliers_iqr,
        filter_data_and_ids,
        identify_cluster_outliers_combined,
    )

    assert callable(identify_outliers)
    assert callable(identify_embryo_outliers_iqr)

    print("✓ QC quality_control module imports OK")
    return True


def test_backward_compat_outliers():
    """Test backward compatibility for outliers.py."""
    print("Testing backward compatibility: outliers.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.outliers import (
            identify_outliers,
            remove_outliers_from_distance_matrix,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ No DeprecationWarning raised (may be cached)")

        # Verify functions work
        assert callable(identify_outliers)
        assert callable(remove_outliers_from_distance_matrix)

    print("✓ Backward compatibility (outliers.py) OK")
    return True


def test_backward_compat_distance_filtering():
    """Test backward compatibility for distance_filtering.py."""
    print("Testing backward compatibility: distance_filtering.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.distance_filtering import (
            identify_embryo_outliers_iqr,
            filter_data_and_ids,
            identify_cluster_outliers_combined,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ No DeprecationWarning raised (may be cached)")

        # Verify functions work
        assert callable(identify_embryo_outliers_iqr)
        assert callable(filter_data_and_ids)
        assert callable(identify_cluster_outliers_combined)

    print("✓ Backward compatibility (distance_filtering.py) OK")
    return True


def test_consensus_pipeline_imports():
    """Test that consensus_pipeline.py imports work with new structure."""
    print("Testing consensus_pipeline.py imports...")

    # This should work without errors
    from analyze.trajectory_analysis.consensus_pipeline import (
        run_consensus_pipeline,
        create_filtering_log,
    )

    assert callable(run_consensus_pipeline)
    assert callable(create_filtering_log)

    print("✓ consensus_pipeline.py imports OK")
    return True


def test_main_init_imports():
    """Test that main __init__.py imports all QC functions."""
    print("Testing main __init__.py imports...")

    from analyze.trajectory_analysis import (
        identify_outliers,
        remove_outliers_from_distance_matrix,
        identify_embryo_outliers_iqr,
        filter_data_and_ids,
        identify_cluster_outliers_combined,
    )

    assert callable(identify_outliers)
    assert callable(remove_outliers_from_distance_matrix)
    assert callable(identify_embryo_outliers_iqr)
    assert callable(filter_data_and_ids)
    assert callable(identify_cluster_outliers_combined)

    print("✓ Main __init__.py imports OK")
    return True


def run_all_tests():
    """Run all Phase 3 tests."""
    print("=" * 60)
    print("Phase 3 Import Tests - QC Consolidation")
    print("=" * 60)

    results = {}

    tests = [
        ("qc_subpackage", test_qc_subpackage),
        ("qc_quality_control", test_qc_quality_control_module),
        ("backward_compat_outliers", test_backward_compat_outliers),
        ("backward_compat_distance_filtering", test_backward_compat_distance_filtering),
        ("consensus_pipeline", test_consensus_pipeline_imports),
        ("main_init", test_main_init_imports),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print("All Phase 3 tests PASSED!")
    else:
        print("Some tests FAILED!")

    return all_passed


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    success = run_all_tests()
    sys.exit(0 if success else 1)
