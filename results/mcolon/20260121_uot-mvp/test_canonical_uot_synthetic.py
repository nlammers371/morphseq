#!/usr/bin/env python3
"""
Test canonical grid with full UOT pipeline on synthetic data.

This tests the complete integration:
1. Create synthetic masks at different resolutions
2. Transform to canonical grid
3. Run full UOT solver
4. Verify results are resolution-invariant and in physical units
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.analyze.optimal_transport_morphometrics.uot_masks import run_uot_pair
from src.analyze.utils.optimal_transport import UOTConfig, UOTFramePair, UOTFrame, MassMode


OUTPUT_DIR = Path("results/mcolon/20260121_uot-mvp/canonical_uot_synthetic")


def make_circle(shape: tuple[int, int], center_yx: tuple[int, int], radius: int) -> np.ndarray:
    """Create a circle mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def test_translation_at_resolution(um_per_px: float, grid_shape: tuple[int, int], test_name: str):
    """
    Test translation at a specific resolution.

    Create two circles with a known shift and verify the velocity is correct in μm.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test_name} @ {um_per_px:.1f} μm/px")
    print('='*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_DIR / f"{test_name}_{um_per_px:.1f}um_per_px"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Physical parameters (in micrometers)
    target_radius_um = 312.0  # 40 px at 7.8 um/px
    target_shift_um = 78.0    # 10 px at 7.8 um/px (diagonal shift)

    # Convert to pixels at current resolution
    radius_px = int(target_radius_um / um_per_px)
    shift_px = int(target_shift_um / um_per_px / np.sqrt(2))  # Each axis shift for diagonal

    # Create masks
    cy, cx = grid_shape[0] // 2, grid_shape[1] // 2
    src_mask = make_circle(grid_shape, (cy, cx), radius=radius_px)
    tgt_mask = make_circle(grid_shape, (cy + shift_px, cx + shift_px), radius=radius_px)

    print(f"Grid shape: {grid_shape}")
    print(f"Radius: {radius_px} px = {target_radius_um:.1f} μm")
    print(f"Shift per axis: {shift_px} px = {shift_px * um_per_px:.1f} μm")
    print(f"Expected diagonal shift: {target_shift_um:.1f} μm")

    # Save input masks
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(src_mask, cmap="gray")
    axes[0].set_title(f"Source\n{um_per_px:.1f} μm/px")
    axes[0].axis("off")
    axes[1].imshow(tgt_mask, cmap="gray")
    axes[1].set_title(f"Target\n{um_per_px:.1f} μm/px")
    axes[1].axis("off")
    fig.savefig(output_dir / "input_masks.png", dpi=150)
    plt.close(fig)

    # Create UOT config with canonical grid
    config = UOTConfig(
        use_canonical_grid=True,
        canonical_grid_um_per_pixel=7.8,
        canonical_grid_shape_hw=(256, 576),
        canonical_grid_align_mode="none",  # No rotation for synthetic

        epsilon=1e-2,
        marginal_relaxation=10.0,
        downsample_factor=1,
        mass_mode=MassMode.UNIFORM,
        max_support_points=5000,
        store_coupling=True,  # Required for transport maps
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=1.0,

        padding_px=0,
        downsample_divisor=1,
        align_mode="none",
    )

    # Create frames with metadata
    pair = UOTFramePair(
        src=UOTFrame(
            embryo_mask=src_mask,
            meta={"um_per_pixel": um_per_px}
        ),
        tgt=UOTFrame(
            embryo_mask=tgt_mask,
            meta={"um_per_pixel": um_per_px}
        ),
    )

    # Run UOT
    print("\nRunning UOT...")
    result = run_uot_pair(pair, config=config)

    # Check canonical grid was used
    is_canonical = result.transform_meta.get("canonical_grid", False)
    print(f"Canonical grid used: {is_canonical}")

    if not is_canonical:
        print("ERROR: Canonical grid was not used!")
        return None

    # Get transform info
    preprocess = result.transform_meta.get("preprocess", {})
    src_transform = preprocess.get("src_transform")
    print(f"Canonical shape: {src_transform.grid_shape_hw}")
    print(f"Scale factor: {src_transform.scale_factor:.3f}")
    print(f"Effective um/px: {src_transform.effective_um_per_pixel:.3f}")

    # Velocity is now in μm!
    velocity_mag = np.sqrt(
        result.velocity_field_yx_hw2[..., 0]**2 +
        result.velocity_field_yx_hw2[..., 1]**2
    )
    mean_vel_um = float(np.mean(velocity_mag[velocity_mag > 0]))
    max_vel_um = float(np.max(velocity_mag))

    print(f"\nResults:")
    print(f"  Cost: {result.cost:.6f}")
    print(f"  Mean velocity: {mean_vel_um:.2f} μm")
    print(f"  Max velocity: {max_vel_um:.2f} μm")
    print(f"  Expected: {target_shift_um:.2f} μm")
    print(f"  Error: {abs(mean_vel_um - target_shift_um):.2f} μm ({abs(mean_vel_um - target_shift_um)/target_shift_um*100:.1f}%)")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im = axes[0].imshow(velocity_mag, cmap="viridis")
    axes[0].set_title(f"Velocity magnitude\n(μm/frame)")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0])

    im = axes[1].imshow(result.mass_created_hw, cmap="Reds")
    axes[1].set_title("Mass created")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    im = axes[2].imshow(result.mass_destroyed_hw, cmap="Blues")
    axes[2].set_title("Mass destroyed")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2])

    fig.suptitle(f"{test_name} @ {um_per_px:.1f} μm/px\nMean velocity: {mean_vel_um:.2f} μm", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "results.png", dpi=150)
    plt.close(fig)

    print(f"Saved results to {output_dir}")

    return {
        "test_name": test_name,
        "um_per_pixel": um_per_px,
        "cost": result.cost,
        "mean_velocity_um": mean_vel_um,
        "max_velocity_um": max_vel_um,
        "expected_velocity_um": target_shift_um,
        "error_um": abs(mean_vel_um - target_shift_um),
        "error_percent": abs(mean_vel_um - target_shift_um) / target_shift_um * 100,
    }


def main():
    print("\n" + "="*60)
    print("CANONICAL GRID + UOT SYNTHETIC TEST")
    print("="*60)

    # Test at multiple resolutions
    test_cases = [
        (5.0, (512, 512), "high_res"),
        (7.8, (384, 384), "canonical_res"),
        (10.0, (256, 256), "low_res"),
    ]

    results = []
    for um_per_px, grid_shape, test_name in test_cases:
        result = test_translation_at_resolution(um_per_px, grid_shape, test_name)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results:
        print(f"\n{'Test':<20} {'Resolution':<12} {'Velocity (μm)':<15} {'Expected':<10} {'Error %':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['test_name']:<20} {r['um_per_pixel']:<12.1f} {r['mean_velocity_um']:<15.2f} {r['expected_velocity_um']:<10.2f} {r['error_percent']:<10.1f}")

        # Check resolution invariance
        velocities = [r['mean_velocity_um'] for r in results]
        vel_std = np.std(velocities)
        vel_mean = np.mean(velocities)
        print(f"\nVelocity statistics across resolutions:")
        print(f"  Mean: {vel_mean:.2f} μm")
        print(f"  Std: {vel_std:.2f} μm")
        print(f"  CV: {vel_std/vel_mean*100:.1f}%")

        if vel_std / vel_mean < 0.1:  # Less than 10% variation
            print("\n✓ RESOLUTION INVARIANCE: PASSED")
        else:
            print("\n⚠ RESOLUTION INVARIANCE: HIGH VARIATION")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
