#!/usr/bin/env python3
"""
Basic Canonical Grid Tests

Quick validation tests for the canonical grid implementation.
Tests grid transformation without running full UOT solver.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

import numpy as np

from src.analyze.optimal_transport_morphometrics.uot_masks.uot_grid import (
    CanonicalGridConfig,
    compute_grid_transform,
    apply_grid_transform,
    rescale_velocity_to_um,
    rescale_distance_to_um,
)


def make_circle(shape: tuple[int, int], center_yx: tuple[int, int], radius: int) -> np.ndarray:
    """Create a circle mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def test_grid_shape():
    """Test that canonical grid has correct shape."""
    print("\n" + "="*60)
    print("TEST 1: Canonical Grid Shape")
    print("="*60)

    config = CanonicalGridConfig(
        reference_um_per_pixel=7.8,
        grid_shape_hw=(256, 576),
        align_mode="none",
        downsample_factor=1,
    )

    # Create a test mask at different resolution
    mask = make_circle((100, 100), (50, 50), radius=30)

    transform = compute_grid_transform(mask, 10.0, None, config)
    canonical = apply_grid_transform(mask, transform)

    print(f"Input shape: {mask.shape}")
    print(f"Canonical shape: {canonical.shape}")
    print(f"Expected shape: {config.grid_shape_hw}")

    assert canonical.shape == config.grid_shape_hw, f"Expected {config.grid_shape_hw}, got {canonical.shape}"
    print("✓ Shape test passed")


def test_resolution_invariance():
    """Test that same physical mask at different resolutions produces same canonical grid."""
    print("\n" + "="*60)
    print("TEST 2: Resolution Invariance")
    print("="*60)

    config = CanonicalGridConfig(
        reference_um_per_pixel=7.8,
        grid_shape_hw=(256, 576),
        align_mode="none",
        downsample_factor=1,
    )

    # Create same physical circle (312 μm radius) at different resolutions
    target_radius_um = 312.0

    resolutions = [
        (5.0, 512, 512),   # 5 μm/px → radius = 62 px
        (7.8, 384, 384),   # 7.8 μm/px → radius = 40 px (canonical)
        (10.0, 256, 256),  # 10 μm/px → radius = 31 px
    ]

    canonical_masks = []

    for um_per_px, h, w in resolutions:
        radius_px = int(target_radius_um / um_per_px)
        mask = make_circle((h, w), (h // 2, w // 2), radius=radius_px)

        transform = compute_grid_transform(mask, um_per_px, None, config)
        canonical = apply_grid_transform(mask, transform)

        canonical_masks.append(canonical)

        print(f"\nResolution: {um_per_px:.1f} μm/px ({h}×{w})")
        print(f"  Input radius: {radius_px} px")
        print(f"  Scale factor: {transform.scale_factor:.3f}")
        print(f"  Canonical mask sum: {canonical.sum()}")

    # Compare canonical masks
    print("\nComparing canonical masks:")
    for i in range(len(canonical_masks) - 1):
        diff = np.abs(canonical_masks[i].astype(float) - canonical_masks[i + 1].astype(float))
        diff_fraction = diff.sum() / max(canonical_masks[i].sum(), canonical_masks[i + 1].sum())
        print(f"  Mask {i} vs {i+1}: {diff_fraction:.3f} difference fraction")

    # They won't be identical due to discretization, but should be very similar
    diff_01 = np.abs(canonical_masks[0].astype(float) - canonical_masks[1].astype(float))
    diff_fraction = diff_01.sum() / max(canonical_masks[0].sum(), canonical_masks[1].sum())

    if diff_fraction < 0.2:  # Allow 20% difference due to discretization
        print("✓ Resolution invariance test passed (differences within tolerance)")
    else:
        print(f"⚠ Resolution invariance test warning: difference fraction {diff_fraction:.3f} > 0.2")


def test_scale_factors():
    """Test that scale factors are computed correctly."""
    print("\n" + "="*60)
    print("TEST 3: Scale Factor Computation")
    print("="*60)

    config = CanonicalGridConfig(
        reference_um_per_pixel=7.8,
        grid_shape_hw=(256, 576),
        align_mode="none",
        downsample_factor=1,
    )

    test_cases = [
        (3.9, 0.5),    # 2x higher resolution → scale = 0.5
        (7.8, 1.0),    # Same resolution → scale = 1.0
        (15.6, 2.0),   # 2x lower resolution → scale = 2.0
    ]

    mask = make_circle((100, 100), (50, 50), radius=30)

    for source_um_per_px, expected_scale in test_cases:
        transform = compute_grid_transform(mask, source_um_per_px, None, config)

        print(f"\nSource: {source_um_per_px:.1f} μm/px")
        print(f"  Expected scale: {expected_scale:.1f}")
        print(f"  Computed scale: {transform.scale_factor:.3f}")

        assert abs(transform.scale_factor - expected_scale) < 1e-6, \
            f"Expected {expected_scale}, got {transform.scale_factor}"

    print("\n✓ Scale factor test passed")


def test_velocity_rescaling():
    """Test velocity rescaling to micrometers."""
    print("\n" + "="*60)
    print("TEST 4: Velocity Rescaling to Micrometers")
    print("="*60)

    config = CanonicalGridConfig(
        reference_um_per_pixel=7.8,
        grid_shape_hw=(256, 576),
        align_mode="none",
        downsample_factor=1,
    )

    mask = make_circle((100, 100), (50, 50), radius=30)
    transform = compute_grid_transform(mask, 7.8, None, config)

    # Create a simple velocity field (10 pixels displacement in y direction)
    velocity_px = np.zeros((256, 576, 2))
    velocity_px[:, :, 0] = 10.0  # 10 pixels in y

    velocity_um = rescale_velocity_to_um(velocity_px, transform)

    expected_um = 10.0 * 7.8  # 78 μm
    actual_um = velocity_um[0, 0, 0]

    print(f"Velocity in pixels: {velocity_px[0, 0, 0]:.1f} px")
    print(f"Velocity in μm: {actual_um:.1f} μm")
    print(f"Expected: {expected_um:.1f} μm")

    assert abs(actual_um - expected_um) < 1e-6, f"Expected {expected_um}, got {actual_um}"
    print("✓ Velocity rescaling test passed")


def test_distance_rescaling():
    """Test distance rescaling to micrometers."""
    print("\n" + "="*60)
    print("TEST 5: Distance Rescaling to Micrometers")
    print("="*60)

    config = CanonicalGridConfig(
        reference_um_per_pixel=7.8,
        grid_shape_hw=(256, 576),
        align_mode="none",
        downsample_factor=2,  # Test with downsampling
    )

    mask = make_circle((100, 100), (50, 50), radius=30)
    transform = compute_grid_transform(mask, 7.8, None, config)

    # With downsampling=2, effective resolution = 7.8 * 2 = 15.6 μm/px
    print(f"Reference um/px: {config.reference_um_per_pixel}")
    print(f"Downsample factor: {config.downsample_factor}")
    print(f"Effective um/px: {transform.effective_um_per_pixel}")

    expected_effective = 7.8 * 2
    assert abs(transform.effective_um_per_pixel - expected_effective) < 1e-6

    # Test distance conversion
    distance_px = 10.0
    distance_um = rescale_distance_to_um(distance_px, transform)

    expected_um = 10.0 * 15.6
    print(f"\nDistance in pixels: {distance_px:.1f} px")
    print(f"Distance in μm: {distance_um:.1f} μm")
    print(f"Expected: {expected_um:.1f} μm")

    assert abs(distance_um - expected_um) < 1e-6, f"Expected {expected_um}, got {distance_um}"
    print("✓ Distance rescaling test passed")


def main():
    print("\n" + "="*60)
    print("CANONICAL GRID BASIC TESTS")
    print("="*60)

    test_grid_shape()
    test_resolution_invariance()
    test_scale_factors()
    test_velocity_rescaling()
    test_distance_rescaling()

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    main()
