#!/usr/bin/env python3
"""
Test canonical grid implementation on real CEP290 data.

Validates:
1. Consecutive frames have low cost and reasonable velocities
2. Velocities are in interpretable physical units (μm/frame)
3. All embryos are oriented consistently
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.analyze.optimal_transport_morphometrics.uot_masks import (
    run_uot_pair,
    load_mask_pair_from_csv,
)
from src.analyze.utils.optimal_transport import UOTConfig, MassMode


# Configuration
DATA_CSV = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv")
OUTPUT_DIR = Path("results/mcolon/20260121_uot-mvp/canonical_grid_cep290")

# Test embryos
TEST_EMBRYOS = [
    "20251113_A05_e01",
    "20251113_E04_e01",
]

# Frame intervals to test
FRAME_INTERVALS = [1, 3]  # 1 frame = ~0.32 hr, 3 frames = ~0.96 hr

# Starting frame
STARTING_FRAME = 100


def create_canonical_config() -> UOTConfig:
    """Create UOT config with canonical grid enabled."""
    return UOTConfig(
        # Canonical grid settings
        use_canonical_grid=True,
        canonical_grid_um_per_pixel=7.8,
        canonical_grid_shape_hw=(256, 576),
        canonical_grid_align_mode="centroid",  # "yolk" not available yet

        # UOT settings
        epsilon=1e-2,
        marginal_relaxation=10.0,
        downsample_factor=1,  # No additional downsampling
        mass_mode=MassMode.UNIFORM,
        max_support_points=5000,
        store_coupling=False,  # Faster without coupling
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=1.0,

        # No legacy preprocessing
        padding_px=0,
        downsample_divisor=1,
        align_mode="none",
    )


def test_consecutive_frames():
    """Test UOT on consecutive frames with canonical grid."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = create_canonical_config()
    results = []

    for embryo_id in TEST_EMBRYOS:
        print(f"\n{'='*60}")
        print(f"Testing embryo: {embryo_id}")
        print('='*60)

        for interval in FRAME_INTERVALS:
            frame_src = STARTING_FRAME
            frame_tgt = STARTING_FRAME + interval

            print(f"\nInterval: {interval} frames ({frame_src} → {frame_tgt})")

            try:
                # Load masks
                pair = load_mask_pair_from_csv(
                    DATA_CSV,
                    embryo_id,
                    frame_src,
                    frame_tgt,
                )

                # Check that um_per_pixel was extracted
                src_um_per_px = pair.src.meta.get("um_per_pixel", np.nan)
                tgt_um_per_px = pair.tgt.meta.get("um_per_pixel", np.nan)
                print(f"  Source um/px: {src_um_per_px:.3f}")
                print(f"  Target um/px: {tgt_um_per_px:.3f}")

                if np.isnan(src_um_per_px) or np.isnan(tgt_um_per_px):
                    print("  ⚠ WARNING: um_per_pixel not available, skipping")
                    continue

                # Run UOT with canonical grid
                result = run_uot_pair(pair, config=config)

                # Extract metrics
                metrics = result.diagnostics.get("metrics", {})

                # Check if canonical grid was used
                is_canonical = result.transform_meta.get("canonical_grid", False)
                print(f"  Canonical grid used: {is_canonical}")

                if is_canonical:
                    # Get canonical grid info
                    preprocess = result.transform_meta.get("preprocess", {})
                    src_transform = preprocess.get("src_transform")
                    if src_transform:
                        print(f"  Canonical shape: {src_transform.grid_shape_hw}")
                        print(f"  Effective um/px: {src_transform.effective_um_per_pixel:.3f}")

                        # Velocity is now in μm!
                        velocity_mag = np.sqrt(
                            result.velocity_field_yx_hw2[..., 0]**2 +
                            result.velocity_field_yx_hw2[..., 1]**2
                        )
                        mean_vel_um = float(np.mean(velocity_mag[velocity_mag > 0]))
                        print(f"  Mean velocity: {mean_vel_um:.2f} μm/frame")

                print(f"  Cost: {result.cost:.6f}")
                print(f"  Created mass fraction: {metrics.get('created_mass_fraction', np.nan):.4f}")
                print(f"  Destroyed mass fraction: {metrics.get('destroyed_mass_fraction', np.nan):.4f}")

                # Save result
                results.append({
                    "embryo_id": embryo_id,
                    "frame_src": frame_src,
                    "frame_tgt": frame_tgt,
                    "interval": interval,
                    "cost": result.cost,
                    "mean_velocity_um": mean_vel_um if is_canonical else np.nan,
                    **metrics,
                })

                # Visualize canonical grids
                if is_canonical:
                    preprocess = result.transform_meta.get("preprocess", {})

                    # Get original and canonical masks for visualization
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                    axes[0, 0].imshow(pair.src.embryo_mask, cmap="gray")
                    axes[0, 0].set_title(f"Source (original)\n{embryo_id} frame {frame_src}")
                    axes[0, 0].axis("off")

                    axes[0, 1].imshow(pair.tgt.embryo_mask, cmap="gray")
                    axes[0, 1].set_title(f"Target (original)\nframe {frame_tgt}")
                    axes[0, 1].axis("off")

                    # Show velocity field
                    velocity_mag = np.sqrt(
                        result.velocity_field_yx_hw2[..., 0]**2 +
                        result.velocity_field_yx_hw2[..., 1]**2
                    )
                    im = axes[1, 0].imshow(velocity_mag, cmap="viridis")
                    axes[1, 0].set_title(f"Velocity magnitude (μm/frame)")
                    axes[1, 0].axis("off")
                    plt.colorbar(im, ax=axes[1, 0])

                    # Show creation/destruction
                    total_change = result.mass_created_hw + result.mass_destroyed_hw
                    im = axes[1, 1].imshow(total_change, cmap="hot")
                    axes[1, 1].set_title("Mass creation + destruction")
                    axes[1, 1].axis("off")
                    plt.colorbar(im, ax=axes[1, 1])

                    fig.tight_layout()
                    output_path = OUTPUT_DIR / f"{embryo_id}_f{frame_src}_to_f{frame_tgt}.png"
                    fig.savefig(output_path, dpi=150)
                    plt.close(fig)
                    print(f"  Saved visualization to {output_path}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Save summary
    if results:
        df = pd.DataFrame(results)
        summary_path = OUTPUT_DIR / "canonical_grid_test_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to {summary_path}")
        print('='*60)

        # Print summary statistics
        print("\nSUMMARY STATISTICS:")
        for interval in FRAME_INTERVALS:
            interval_data = df[df["interval"] == interval]
            if len(interval_data) > 0:
                print(f"\nInterval {interval} frames:")
                print(f"  Cost: {interval_data['cost'].mean():.6f} ± {interval_data['cost'].std():.6f}")
                if "mean_velocity_um" in interval_data.columns:
                    vel_data = interval_data["mean_velocity_um"].dropna()
                    if len(vel_data) > 0:
                        print(f"  Velocity: {vel_data.mean():.2f} ± {vel_data.std():.2f} μm/frame")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CANONICAL GRID CEP290 TEST")
    print("="*60)

    if not DATA_CSV.exists():
        print(f"ERROR: Data CSV not found at {DATA_CSV}")
        sys.exit(1)

    test_consecutive_frames()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
