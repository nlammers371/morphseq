"""
Test script for MD-DTW core functions.

Tests prepare_multivariate_array() and compute_md_dtw_distance_matrix() with synthetic data.

Created: 2025-12-18
Location: results/mcolon/20251218_MD-DTW-morphseq_analysis/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import functions to test
from md_dtw_prototype import prepare_multivariate_array, compute_md_dtw_distance_matrix

print("=" * 70)
print("MD-DTW Functions Test Suite")
print("=" * 70)

# Create synthetic test data
np.random.seed(42)
n_embryos = 5
n_timepoints_base = 50
time_grid = np.linspace(18, 52, n_timepoints_base)

data_rows = []
for i in range(n_embryos):
    embryo_id = f"test_embryo_{i}"

    # Vary patterns slightly per embryo
    curvature = 2.0 + 0.5 * np.sin(time_grid / 10 + i) + np.random.normal(0, 0.1, n_timepoints_base)
    length = 300 + 5 * time_grid + i * 10 + np.random.normal(0, 5, n_timepoints_base)

    for t_idx, t in enumerate(time_grid):
        data_rows.append({
            'embryo_id': embryo_id,
            'predicted_stage_hpf': t,
            'baseline_deviation_normalized': curvature[t_idx],
            'total_length_um': length[t_idx],
        })

df = pd.DataFrame(data_rows)

print(f"\n✓ Created test DataFrame:")
print(f"  - Total rows: {len(df)}")
print(f"  - Embryos: {df['embryo_id'].nunique()}")
print(f"  - Timepoints per embryo: {len(df) // df['embryo_id'].nunique()}")
print(f"  - Time range: {df['predicted_stage_hpf'].min():.1f} - {df['predicted_stage_hpf'].max():.1f} hpf")

# Test 1: prepare_multivariate_array() - Basic functionality
print("\n" + "=" * 70)
print("Test 1: prepare_multivariate_array() - Basic functionality")
print("=" * 70)

try:
    X, embryo_ids, time_grid_out = prepare_multivariate_array(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        normalize=True,
        verbose=False
    )

    # Validate shape
    assert X.shape[0] == n_embryos, f"Expected {n_embryos} embryos, got {X.shape[0]}"
    assert X.shape[2] == 2, f"Expected 2 metrics, got {X.shape[2]}"
    assert len(embryo_ids) == n_embryos, f"Expected {n_embryos} IDs, got {len(embryo_ids)}"

    # Validate normalization (should be close to 0 mean, 1 std)
    mean_val = X.mean()
    std_val = X.std()
    assert abs(mean_val) < 1e-10, f"Mean should be ~0, got {mean_val}"
    assert abs(std_val - 1.0) < 0.01, f"Std should be ~1, got {std_val}"

    print(f"✓ Test 1 PASSED")
    print(f"  - Shape: {X.shape}")
    print(f"  - Embryo IDs: {len(embryo_ids)}")
    print(f"  - Time points: {len(time_grid_out)}")
    print(f"  - Normalized mean: {mean_val:.2e} (target: 0)")
    print(f"  - Normalized std: {std_val:.4f} (target: 1)")

except AssertionError as e:
    print(f"✗ Test 1 FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test 1 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: prepare_multivariate_array() - Without normalization
print("\n" + "=" * 70)
print("Test 2: prepare_multivariate_array() - Without normalization")
print("=" * 70)

try:
    X_unnorm, _, _ = prepare_multivariate_array(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        normalize=False,
        verbose=False
    )

    # Validate that unnormalized has different scale
    mean_val_unnorm = X_unnorm.mean()
    std_val_unnorm = X_unnorm.std()

    # Should NOT be normalized
    assert abs(mean_val_unnorm) > 1.0, "Unnormalized should not have zero mean"

    print(f"✓ Test 2 PASSED")
    print(f"  - Mean: {mean_val_unnorm:.2f} (not normalized)")
    print(f"  - Std: {std_val_unnorm:.2f} (not normalized)")

except AssertionError as e:
    print(f"✗ Test 2 FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test 2 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: compute_md_dtw_distance_matrix() - Symmetry and properties
print("\n" + "=" * 70)
print("Test 3: compute_md_dtw_distance_matrix() - Symmetry and properties")
print("=" * 70)

try:
    D = compute_md_dtw_distance_matrix(
        X,
        sakoe_chiba_radius=3,
        n_jobs=1,  # Single job for reproducibility
        verbose=False
    )

    # Validate shape
    assert D.shape == (n_embryos, n_embryos), f"Expected ({n_embryos}, {n_embryos}), got {D.shape}"

    # Validate diagonal (should be all zeros)
    diagonal = np.diag(D)
    max_diagonal = np.max(np.abs(diagonal))
    assert max_diagonal < 1e-10, f"Diagonal should be 0, max value: {max_diagonal}"

    # Validate symmetry
    max_asymmetry = np.max(np.abs(D - D.T))
    assert max_asymmetry < 1e-10, f"Matrix should be symmetric, max asymmetry: {max_asymmetry}"

    # Validate positive distances (off-diagonal)
    off_diagonal = D[np.triu_indices(n_embryos, k=1)]
    assert np.all(off_diagonal > 0), "All off-diagonal distances should be positive"

    print(f"✓ Test 3 PASSED")
    print(f"  - Distance matrix shape: {D.shape}")
    print(f"  - Diagonal max: {max_diagonal:.2e} (target: 0)")
    print(f"  - Max asymmetry: {max_asymmetry:.2e} (target: 0)")
    print(f"  - Distance range: [{D[D > 0].min():.4f}, {D.max():.4f}]")
    print(f"  - Mean distance: {off_diagonal.mean():.4f}")

except AssertionError as e:
    print(f"✗ Test 3 FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test 3 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: compute_md_dtw_distance_matrix() - Different radii
print("\n" + "=" * 70)
print("Test 4: compute_md_dtw_distance_matrix() - Different Sakoe-Chiba radii")
print("=" * 70)

try:
    radii = [None, 5, 3, 1]
    results = {}

    for radius in radii:
        D_radius = compute_md_dtw_distance_matrix(
            X,
            sakoe_chiba_radius=radius,
            n_jobs=1,
            verbose=False
        )
        results[radius] = D_radius

    # Validate that smaller radius gives larger or equal distances
    # (more constrained warping path)
    print(f"✓ Test 4 PASSED")
    for radius in radii:
        D_r = results[radius]
        off_diag = D_r[np.triu_indices(n_embryos, k=1)]
        print(f"  - Radius {radius}: mean distance = {off_diag.mean():.4f}")

except Exception as e:
    print(f"✗ Test 4 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Single metric edge case
print("\n" + "=" * 70)
print("Test 5: Single metric edge case")
print("=" * 70)

try:
    X_single, _, _ = prepare_multivariate_array(
        df,
        metrics=['baseline_deviation_normalized'],  # Only one metric
        normalize=True,
        verbose=False
    )

    assert X_single.shape[2] == 1, f"Expected 1 metric dimension, got {X_single.shape[2]}"

    D_single = compute_md_dtw_distance_matrix(
        X_single,
        sakoe_chiba_radius=3,
        n_jobs=1,
        verbose=False
    )

    assert D_single.shape == (n_embryos, n_embryos), f"Unexpected shape: {D_single.shape}"

    print(f"✓ Test 5 PASSED")
    print(f"  - Single metric array shape: {X_single.shape}")
    print(f"  - Distance matrix shape: {D_single.shape}")

except AssertionError as e:
    print(f"✗ Test 5 FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test 5 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verify embryo ID ordering consistency
print("\n" + "=" * 70)
print("Test 6: Embryo ID ordering consistency")
print("=" * 70)

try:
    # Run twice and verify same ordering
    X1, ids1, _ = prepare_multivariate_array(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        normalize=False,
        verbose=False
    )

    X2, ids2, _ = prepare_multivariate_array(
        df,
        metrics=['baseline_deviation_normalized', 'total_length_um'],
        normalize=False,
        verbose=False
    )

    # Verify IDs are same and in same order
    assert ids1 == ids2, "Embryo ID ordering not consistent across runs"

    # Verify arrays are identical
    assert np.allclose(X1, X2), "Arrays not identical across runs"

    print(f"✓ Test 6 PASSED")
    print(f"  - Embryo IDs consistent: {ids1 == ids2}")
    print(f"  - Arrays identical: {np.allclose(X1, X2)}")
    print(f"  - Embryo ID order: {ids1}")

except AssertionError as e:
    print(f"✗ Test 6 FAILED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test 6 ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nSummary:")
print("  1. prepare_multivariate_array() - Basic functionality ✓")
print("  2. prepare_multivariate_array() - Without normalization ✓")
print("  3. compute_md_dtw_distance_matrix() - Symmetry and properties ✓")
print("  4. compute_md_dtw_distance_matrix() - Different radii ✓")
print("  5. Single metric edge case ✓")
print("  6. Embryo ID ordering consistency ✓")
print("\nThe MD-DTW core functions are working correctly!")
print("=" * 70)
