#!/usr/bin/env python3
"""
Canonical Grid Validation Tests

Tests the canonical grid implementation with synthetic masks at different resolutions.
Validates that results are resolution-invariant after transformation.

Test cases:
1. Identity test: same mask → zero velocity, zero cost
2. Translation test: shifted mask → uniform velocity = shift distance (in μm!)
3. Scale test: different source resolutions → identical results after canonical transform
4. Shape test: circle→oval → outward velocity at expansion zone
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from src.analyze.optimal_transport_morphometrics.uot_masks.uot_grid import (
    CanonicalGridConfig,
    GridTransform,
    compute_grid_transform,
    apply_grid_transform,
    rescale_velocity_to_um,
    rescale_distance_to_um,
)
from src.analyze.optimal_transport_morphometrics.uot_masks import run_uot_pair
from src.analyze.utils.optimal_transport import UOTConfig, UOTFramePair, UOTFrame, MassMode
from src.analyze.optimal_transport_morphometrics.uot_masks.viz import (
    plot_creation_destruction,
    plot_velocity_overlay,
)


OUTPUT_DIR = Path("results/mcolon/20260121_uot-mvp/synthetic_canonical_grid")


def make_circle(shape: Tuple[int, int], center_yx: Tuple[int, int], radius: int) -> np.ndarray:
    """Create a circle mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def make_ellipse(
    shape: Tuple[int, int],
    center_yx: Tuple[int, int],
    radius_y: int,
    radius_x: int
) -> np.ndarray:
    """Create an ellipse mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = ((yy - cy) / float(radius_y)) ** 2 + ((xx - cx) / float(radius_x)) ** 2 <= 1.0
    return mask.astype(np.uint8)


def create_synthetic_at_resolution(
    shape: Tuple[int, int],
    um_per_pixel: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create synthetic test cases at a given resolution.

    Args:
        shape: Image shape (H, W)
        um_per_pixel: Physical resolution

    Returns:
        Dictionary of test cases with source and target masks
    """
    cy, cx = shape[0] // 2, shape[1] // 2

    # Scale radii and shifts based on resolution to maintain physical size
    # At 7.8 um/px: radius=40px = 312 um, shift=20px = 156 um
    target_radius_um = 312.0
    target_shift_um = 156.0

    radius_px = int(target_radius_um / um_per_pixel)
    shift_px = int(target_shift_um / um_per_pixel)

    circle = make_circle(shape, (cy, cx), radius=radius_px)
    circle_shifted = make_circle(shape, (cy + shift_px, cx + shift_px), radius=radius_px)
    oval = make_ellipse(
        shape,
        (cy, cx),
        radius_y=int(radius_px * 0.75),
        radius_x=int(radius_px * 1.5)
    )

    return {
        "identity": {
            "src": circle.copy(),
            "tgt": circle.copy(),
            "expected_cost": 0.0,
            "expected_velocity_um": 0.0,
        },
        "translation": {
            "src": circle,
            "tgt": circle_shifted,
            "expected_cost": None,  # Not zero due to mass redistribution
            "expected_velocity_um": np.sqrt(2) * target_shift_um,  # Diagonal shift
        },
        "shape_change": {
            "src": circle,
            "tgt": oval,
            "expected_cost": None,
            "expected_velocity_um": None,
        },
    }


def apply_canonical_grid(
    mask: np.ndarray,
    source_um_per_pixel: float,
    config: CanonicalGridConfig,
) -> Tuple[np.ndarray, GridTransform]:
    """Apply canonical grid transformation to a mask."""
    transform = compute_grid_transform(
        mask=mask,
        source_um_per_pixel=source_um_per_pixel,
        yolk_mask=None,  # No yolk for synthetic tests
        config=config,
    )

    canonical_mask = apply_grid_transform(mask, transform)

    return canonical_mask, transform


def run_canonical_grid_test(
    case_name: str,
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    source_um_per_pixel: float,
    config: CanonicalGridConfig,
    uot_config: UOTConfig,
    output_dir: Path,
) -> Dict:
    """Run UOT on masks transformed to canonical grid."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transform to canonical grid
    src_canonical, src_transform = apply_canonical_grid(src_mask, source_um_per_pixel, config)
    tgt_canonical, tgt_transform = apply_canonical_grid(tgt_mask, source_um_per_pixel, config)

    # Verify both transforms are identical (same resolution)
    assert abs(src_transform.scale_factor - tgt_transform.scale_factor) < 1e-6
    assert abs(src_transform.rotation_rad - tgt_transform.rotation_rad) < 1e-6

    # Save transformed masks for inspection
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(src_canonical, cmap="gray")
    axes[0].set_title("Source (canonical)")
    axes[0].axis("off")
    axes[1].imshow(tgt_canonical, cmap="gray")
    axes[1].set_title("Target (canonical)")
    axes[1].axis("off")
    fig.suptitle(f"{case_name} @ {source_um_per_pixel:.2f} μm/px", fontsize=12)
    fig.savefig(output_dir / "canonical_masks.png", dpi=200)
    plt.close(fig)

    # Run UOT on canonical grid
    pair = UOTFramePair(
        src=UOTFrame(
            embryo_mask=src_canonical,
            meta={"um_per_pixel": src_transform.effective_um_per_pixel}
        ),
        tgt=UOTFrame(
            embryo_mask=tgt_canonical,
            meta={"um_per_pixel": tgt_transform.effective_um_per_pixel}
        ),
    )

    result = run_uot_pair(pair, config=uot_config)

    # Rescale results to micrometers
    velocity_um = rescale_velocity_to_um(result.velocity_field_yx_hw2, src_transform)

    # Compute mean velocity magnitude
    velocity_mag = np.sqrt(velocity_um[..., 0]**2 + velocity_um[..., 1]**2)
    mean_velocity_um = float(np.mean(velocity_mag[velocity_mag > 0]))

    # Get metrics
    metrics = result.diagnostics.get("metrics", {}) if result.diagnostics else {}

    # Visualize
    fig = plot_creation_destruction(
        result.mass_created_hw,
        result.mass_destroyed_hw,
        output_path=str(output_dir / "creation_destruction.png"),
    )
    plt.close(fig)

    fig = plot_velocity_overlay(
        result.mass_created_hw,
        velocity_um,  # Plot in μm units
        stride=8,
        output_path=str(output_dir / "velocity_field_um.png"),
    )
    plt.close(fig)

    return {
        "case": case_name,
        "source_um_per_pixel": source_um_per_pixel,
        "canonical_um_per_pixel": src_transform.effective_um_per_pixel,
        "cost": result.cost,
        "mean_velocity_um": mean_velocity_um,
        **metrics,
    }


def run_resolution_independence_test(
    canonical_config: CanonicalGridConfig,
    uot_config: UOTConfig,
) -> None:
    """
    Test that different source resolutions produce identical results on canonical grid.
    """
    output_root = OUTPUT_DIR / "resolution_independence"
    output_root.mkdir(parents=True, exist_ok=True)

    # Test at multiple source resolutions
    test_resolutions = [
        (512, 512, 5.0),   # Higher resolution than canonical
        (384, 384, 7.8),   # Same as canonical
        (256, 256, 12.0),  # Lower resolution than canonical
    ]

    all_results = []

    for shape_h, shape_w, um_per_px in test_resolutions:
        print(f"\nTesting resolution: {shape_h}×{shape_w} @ {um_per_px:.1f} μm/px")

        cases = create_synthetic_at_resolution((shape_h, shape_w), um_per_px)

        for case_name, case_data in cases.items():
            print(f"  Case: {case_name}")

            output_dir = output_root / f"{um_per_px:.1f}um_per_px" / case_name

            result = run_canonical_grid_test(
                case_name=case_name,
                src_mask=case_data["src"],
                tgt_mask=case_data["tgt"],
                source_um_per_pixel=um_per_px,
                config=canonical_config,
                uot_config=uot_config,
                output_dir=output_dir,
            )

            result["expected_velocity_um"] = case_data.get("expected_velocity_um", None)
            all_results.append(result)

            print(f"    Cost: {result['cost']:.6f}")
            print(f"    Mean velocity: {result['mean_velocity_um']:.2f} μm")
            if case_data.get("expected_velocity_um") is not None:
                print(f"    Expected velocity: {case_data['expected_velocity_um']:.2f} μm")

    # Save summary
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(output_root / "resolution_independence_summary.csv", index=False)

    # Analyze resolution invariance
    print("\n" + "="*60)
    print("RESOLUTION INDEPENDENCE ANALYSIS")
    print("="*60)

    for case_name in ["identity", "translation", "shape_change"]:
        case_results = df[df["case"] == case_name]
        if len(case_results) == 0:
            continue

        print(f"\n{case_name.upper()}:")
        print(f"  Cost range: [{case_results['cost'].min():.6f}, {case_results['cost'].max():.6f}]")
        print(f"  Cost std: {case_results['cost'].std():.6f}")

        if case_name == "translation":
            print(f"  Velocity range: [{case_results['mean_velocity_um'].min():.2f}, {case_results['mean_velocity_um'].max():.2f}] μm")
            print(f"  Velocity std: {case_results['mean_velocity_um'].std():.2f} μm")
            expected = case_results['expected_velocity_um'].iloc[0]
            if expected is not None:
                print(f"  Expected velocity: {expected:.2f} μm")
                print(f"  Mean error: {abs(case_results['mean_velocity_um'].mean() - expected):.2f} μm")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test canonical grid with synthetic masks.")
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--marginal-relaxation", type=float, default=100.0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Canonical grid config (matching snip export)
    canonical_config = CanonicalGridConfig(
        reference_um_per_pixel=7.8,
        grid_shape_hw=(256, 576),
        align_mode="none",  # No yolk for synthetic tests
        downsample_factor=1,  # No additional downsampling
    )

    # UOT config
    uot_config = UOTConfig(
        epsilon=args.epsilon,
        marginal_relaxation=args.marginal_relaxation,
        downsample_factor=1,  # No downsampling (already on canonical grid)
        downsample_divisor=1,
        padding_px=0,  # No padding (already on canonical grid)
        mass_mode=MassMode.UNIFORM,
        align_mode="none",  # Already aligned
        max_support_points=10000,
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=1.0,  # Will be interpreted in canonical grid pixels
    )

    print("Running resolution independence tests...")
    run_resolution_independence_test(canonical_config, uot_config)

    print("\n" + "="*60)
    print("Tests complete. Results saved to:", OUTPUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()
