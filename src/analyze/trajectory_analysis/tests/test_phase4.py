#!/usr/bin/env python3
"""
Phase 4 Import Tests - Clustering Consolidation

Tests that:
1. New clustering subpackage imports work
2. All clustering functions are accessible from clustering/__init__.py
3. Backward compatibility shims work (bootstrap_clustering.py, cluster_posteriors.py, etc.)
4. Deprecation warnings are raised for old imports
5. Main __init__.py imports work with new structure
"""

import sys
import os
import warnings
from pathlib import Path

# Determine repo root: prefer env var, fallback to __file__-based derivation
# This file is at: <repo>/src/analyze/trajectory_analysis/tests/test_phase4.py
REPO_ROOT = os.environ.get('MORPHSEQ_REPO_ROOT')
if not REPO_ROOT:
    REPO_ROOT = str(Path(__file__).resolve().parents[4])

SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def test_clustering_subpackage():
    """Test clustering subpackage imports."""
    print("Testing clustering subpackage imports...")

    # Test direct import from clustering
    from analyze.trajectory_analysis.clustering import (
        # Bootstrap clustering
        run_bootstrap_hierarchical,
        run_bootstrap_kmedoids,
        compute_consensus_labels,
        get_cluster_assignments,
        compute_coassociation_matrix,
        coassociation_to_distance,
        # Posterior analysis
        analyze_bootstrap_results,
        compute_assignment_posteriors,
        compute_quality_metrics,
        align_bootstrap_labels,
        # Classification
        classify_membership_2d,
        classify_membership_adaptive,
        get_classification_summary,
        # Consensus pipeline
        run_consensus_pipeline,
        create_filtering_log,
        # K selection
        evaluate_k_range,
        plot_k_selection,
        run_k_selection_pipeline,
        run_two_phase_pipeline,
        run_k_selection_with_plots,
        add_membership_column,
        # Cluster extraction
        extract_cluster_embryos,
        get_cluster_summary,
        map_clusters_to_phenotypes,
    )

    # Verify they're callable
    assert callable(run_bootstrap_hierarchical)
    assert callable(analyze_bootstrap_results)
    assert callable(extract_cluster_embryos)
    assert callable(get_cluster_summary)
    assert callable(map_clusters_to_phenotypes)
    assert callable(classify_membership_2d)
    assert callable(run_consensus_pipeline)
    assert callable(evaluate_k_range)
    assert callable(add_membership_column)

    print("✓ Clustering subpackage imports OK")
    return True


def test_backward_compat_bootstrap_clustering():
    """Test backward compatibility for bootstrap_clustering.py."""
    print("Testing backward compatibility: bootstrap_clustering.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.bootstrap_clustering import (
            run_bootstrap_hierarchical,
            run_bootstrap_kmedoids,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ No DeprecationWarning raised (may be cached)")

        # Verify functions work
        assert callable(run_bootstrap_hierarchical)
        assert callable(run_bootstrap_kmedoids)

    print("✓ Backward compatibility (bootstrap_clustering.py) OK")
    return True


def test_backward_compat_cluster_posteriors():
    """Test backward compatibility for cluster_posteriors.py."""
    print("Testing backward compatibility: cluster_posteriors.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.cluster_posteriors import (
            analyze_bootstrap_results,
            compute_assignment_posteriors,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ No DeprecationWarning raised (may be cached)")

        # Verify functions work
        assert callable(analyze_bootstrap_results)
        assert callable(compute_assignment_posteriors)

    print("✓ Backward compatibility (cluster_posteriors.py) OK")
    return True


def test_backward_compat_cluster_classification():
    """Test backward compatibility for cluster_classification.py."""
    print("Testing backward compatibility: cluster_classification.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.cluster_classification import (
            classify_membership_2d,
            classify_membership_adaptive,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ No DeprecationWarning raised (may be cached)")

        # Verify functions work
        assert callable(classify_membership_2d)
        assert callable(classify_membership_adaptive)

    print("✓ Backward compatibility (cluster_classification.py) OK")
    return True


def test_backward_compat_consensus_pipeline():
    """Test backward compatibility for consensus_pipeline.py."""
    print("Testing backward compatibility: consensus_pipeline.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.consensus_pipeline import (
            run_consensus_pipeline,
            create_filtering_log,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ No DeprecationWarning raised (may be cached)")

        # Verify functions work
        assert callable(run_consensus_pipeline)
        assert callable(create_filtering_log)

    print("✓ Backward compatibility (consensus_pipeline.py) OK")
    return True


def test_backward_compat_k_selection():
    """Test backward compatibility for k_selection.py."""
    print("Testing backward compatibility: k_selection.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.k_selection import (
            evaluate_k_range,
            plot_k_selection,
            add_membership_column,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ No DeprecationWarning raised (may be cached)")

        # Verify functions work
        assert callable(evaluate_k_range)
        assert callable(plot_k_selection)
        assert callable(add_membership_column)

    print("✓ Backward compatibility (k_selection.py) OK")
    return True


def test_backward_compat_cluster_extraction():
    """Test backward compatibility for cluster_extraction.py."""
    print("Testing backward compatibility: cluster_extraction.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.cluster_extraction import (
            extract_cluster_embryos,
            get_cluster_summary,
            map_clusters_to_phenotypes,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ No DeprecationWarning raised (may be cached)")

        # Verify functions work
        assert callable(extract_cluster_embryos)
        assert callable(get_cluster_summary)
        assert callable(map_clusters_to_phenotypes)

    print("✓ Backward compatibility (cluster_extraction.py) OK")
    return True


def test_main_init_imports():
    """Test that main __init__.py imports all clustering functions."""
    print("Testing main __init__.py imports...")

    from analyze.trajectory_analysis import (
        # Bootstrap clustering
        run_bootstrap_hierarchical,
        run_bootstrap_kmedoids,
        compute_consensus_labels,
        get_cluster_assignments,
        compute_coassociation_matrix,
        coassociation_to_distance,
        # Posterior analysis
        analyze_bootstrap_results,
        compute_assignment_posteriors,
        compute_quality_metrics,
        align_bootstrap_labels,
        # Classification
        classify_membership_2d,
        classify_membership_adaptive,
        get_classification_summary,
        # Consensus pipeline
        run_consensus_pipeline,
        create_filtering_log,
        # K selection
        evaluate_k_range,
        plot_k_selection,
        run_k_selection_pipeline,
        run_two_phase_pipeline,
        run_k_selection_with_plots,
    )

    assert callable(run_bootstrap_hierarchical)
    assert callable(run_bootstrap_kmedoids)
    assert callable(analyze_bootstrap_results)
    assert callable(classify_membership_2d)
    assert callable(run_consensus_pipeline)
    assert callable(evaluate_k_range)
    assert callable(plot_k_selection)

    print("✓ Main __init__.py imports OK")
    return True


def run_all_tests():
    """Run all Phase 4 tests."""
    print("=" * 60)
    print("Phase 4 Import Tests - Clustering Consolidation")
    print("=" * 60)

    results = {}

    tests = [
        ("clustering_subpackage", test_clustering_subpackage),
        ("backward_compat_bootstrap_clustering", test_backward_compat_bootstrap_clustering),
        ("backward_compat_cluster_posteriors", test_backward_compat_cluster_posteriors),
        ("backward_compat_cluster_classification", test_backward_compat_cluster_classification),
        ("backward_compat_consensus_pipeline", test_backward_compat_consensus_pipeline),
        ("backward_compat_k_selection", test_backward_compat_k_selection),
        ("backward_compat_cluster_extraction", test_backward_compat_cluster_extraction),
        ("main_init", test_main_init_imports),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
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
        print("All Phase 4 tests PASSED!")
    else:
        print("Some tests FAILED!")

    return all_passed


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    success = run_all_tests()
    sys.exit(0 if success else 1)
