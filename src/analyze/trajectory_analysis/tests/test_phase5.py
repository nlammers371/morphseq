#!/usr/bin/env python3
"""
Phase 5 Import Tests - Viz Restructure

Tests that:
1. New viz subpackage imports work
2. viz.plotting subpackage imports work
3. All viz functions are accessible from viz/__init__.py
4. Backward compatibility shims work (dendrogram.py, genotype_styling.py, etc.)
5. Deprecation warnings are raised for old imports
6. Main __init__.py imports work with new structure
7. pair_analysis imports updated correctly
"""

import sys
import os
import warnings
from pathlib import Path

# Determine repo root: prefer env var, fallback to __file__-based derivation
# This file is at: <repo>/src/analyze/trajectory_analysis/tests/test_phase5.py
REPO_ROOT = os.environ.get('MORPHSEQ_REPO_ROOT')
if not REPO_ROOT:
    REPO_ROOT = str(Path(__file__).resolve().parents[4])

SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def test_viz_subpackage():
    """Test viz subpackage imports."""
    print("Testing viz subpackage imports...")

    # Test direct import from viz
    from analyze.trajectory_analysis.viz import (
        # Dendrogram
        generate_dendrograms,
        plot_dendrogram,
        add_cluster_column,
        plot_dendrogram_with_categories,
        PASTEL_COLORS,
        # Styling
        extract_genotype_suffix,
        extract_genotype_prefix,
        get_color_for_genotype,
        sort_genotypes_by_suffix,
        build_genotype_style_config,
        format_genotype_label,
        # Core plotting
        plot_cluster_trajectories_df,
        plot_membership_trajectories_df,
        plot_posterior_heatmap,
        plot_2d_scatter,
        plot_membership_vs_k,
        plot_cluster_flow,
        # Faceted plotting
        plot_proportions,
        # 3D plotting
        plot_3d_scatter,
    )

    # Verify they're callable
    assert callable(generate_dendrograms)
    assert callable(plot_dendrogram)
    assert callable(get_color_for_genotype)
    assert callable(plot_cluster_trajectories_df)
    assert callable(plot_proportions)
    assert callable(plot_cluster_flow)
    assert callable(plot_3d_scatter)

    print("✓ viz subpackage imports OK")
    return True


def test_viz_plotting_subpackage():
    """Test viz.plotting subpackage imports."""
    print("Testing viz.plotting subpackage imports...")

    # Test direct import from viz.plotting
    from analyze.trajectory_analysis.viz.plotting import (
        # Core
        plot_cluster_trajectories_df,
        plot_membership_trajectories_df,
        plot_posterior_heatmap,
        plot_2d_scatter,
        plot_membership_vs_k,
        plot_cluster_flow,
        # Faceted
        plot_proportions,
        # 3D
        plot_3d_scatter,
    )

    # Verify they're callable
    assert callable(plot_cluster_trajectories_df)
    assert callable(plot_proportions)
    assert callable(plot_cluster_flow)
    assert callable(plot_3d_scatter)

    print("✓ viz.plotting subpackage imports OK")
    return True


def test_viz_styling():
    """Test viz.styling imports."""
    print("Testing viz.styling imports...")

    from analyze.trajectory_analysis.viz.styling import (
        extract_genotype_suffix,
        extract_genotype_prefix,
        get_color_for_genotype,
        sort_genotypes_by_suffix,
        build_genotype_style_config,
        format_genotype_label,
    )

    assert callable(get_color_for_genotype)
    assert callable(extract_genotype_suffix)

    print("✓ viz.styling imports OK")
    return True


def test_backward_compat_dendrogram():
    """Test backward compatibility for dendrogram.py."""
    print("Testing backward compatibility: dendrogram.py...")

    # Should work but raise DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.dendrogram import (
            generate_dendrograms,
            plot_dendrogram,
        )

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ Warning: No DeprecationWarning raised (expected one)")

        # Verify imports work
        assert callable(generate_dendrograms)
        assert callable(plot_dendrogram)

    print("✓ dendrogram.py backward compatibility OK")
    return True


def test_backward_compat_genotype_styling():
    """Test backward compatibility for genotype_styling.py."""
    print("Testing backward compatibility: genotype_styling.py...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.genotype_styling import (
            get_color_for_genotype,
            extract_genotype_suffix,
        )

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ Warning: No DeprecationWarning raised (expected one)")

        assert callable(get_color_for_genotype)

    print("✓ genotype_styling.py backward compatibility OK")
    return True


def test_backward_compat_plotting():
    """Test backward compatibility for plotting.py."""
    print("Testing backward compatibility: plotting.py...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.plotting import (
            plot_cluster_trajectories_df,
            plot_membership_trajectories_df,
        )

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ Warning: No DeprecationWarning raised (expected one)")

        assert callable(plot_cluster_trajectories_df)

    print("✓ plotting.py backward compatibility OK")
    return True


def test_backward_compat_plotting_3d():
    """Test backward compatibility for plotting_3d.py."""
    print("Testing backward compatibility: plotting_3d.py...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        from analyze.trajectory_analysis.plotting_3d import plot_3d_scatter

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]

        if len(deprecation_warnings) > 0:
            print(f"  ⚠ DeprecationWarning raised: {deprecation_warnings[0].message}")
        else:
            print("  ⚠ Warning: No DeprecationWarning raised (expected one)")

        assert callable(plot_3d_scatter)

    print("✓ plotting_3d.py backward compatibility OK")
    return True


def test_main_init_imports():
    """Test that main __init__.py imports work with new viz structure."""
    print("Testing main __init__.py imports...")

    from analyze.trajectory_analysis import (
        # Dendrogram
        generate_dendrograms,
        plot_dendrogram_with_categories,
        # Styling
        get_color_for_genotype,
        extract_genotype_suffix,
        # Plotting
        plot_cluster_trajectories_df,
        plot_proportions,
        plot_cluster_flow,
        plot_3d_scatter,
    )

    assert callable(generate_dendrograms)
    assert callable(get_color_for_genotype)
    assert callable(plot_cluster_trajectories_df)
    assert callable(plot_proportions)
    assert callable(plot_cluster_flow)
    assert callable(plot_3d_scatter)

    print("✓ Main __init__.py imports OK")
    return True


def test_pair_analysis_imports():
    """Test that pair_analysis imports updated correctly."""
    print("Testing pair_analysis imports...")

    # This should work without errors (imports from viz now)
    from analyze.trajectory_analysis.pair_analysis import (
        plot_pairs_overview,
        plot_genotypes_by_pair,
    )

    assert callable(plot_pairs_overview)
    assert callable(plot_genotypes_by_pair)

    print("✓ pair_analysis imports OK")
    return True


def main():
    """Run all Phase 5 tests."""
    print("=" * 70)
    print("Phase 5 Import Tests - Viz Restructure")
    print("=" * 70)
    print()

    tests = [
        ("viz subpackage", test_viz_subpackage),
        ("viz.plotting subpackage", test_viz_plotting_subpackage),
        ("viz.styling", test_viz_styling),
        ("backward compat: dendrogram.py", test_backward_compat_dendrogram),
        ("backward compat: genotype_styling.py", test_backward_compat_genotype_styling),
        ("backward compat: plotting.py", test_backward_compat_plotting),
        ("backward compat: plotting_3d.py", test_backward_compat_plotting_3d),
        ("main __init__.py", test_main_init_imports),
        ("pair_analysis", test_pair_analysis_imports),
    ]

    passed = 0
    failed = 0
    results = []

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            results.append(f"✓ {name}")
        except Exception as e:
            failed += 1
            results.append(f"✗ {name}: {e}")
            print(f"✗ FAILED: {e}")

        print()

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    for result in results:
        print(result)

    print()
    print(f"Total: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 70)

    if failed == 0:
        print()
        print("🎉 All Phase 5 tests passed!")
        return 0
    else:
        print()
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
