#!/usr/bin/env python3
"""
UOT MVP: Consecutive Frames Analysis

Test UOT on consecutive frames within a single embryo to validate the pipeline.
Computes transport between frame pairs at different time intervals (consecutive, ~1hr, ~2hr).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.analyze.optimal_transport_morphometrics.uot_masks import (
    run_uot_pair,
    load_mask_pair_from_csv,
)
from src.analyze.utils.optimal_transport import UOTConfig, UOTFramePair, UOTFrame, MassMode
from src.analyze.optimal_transport_morphometrics.uot_masks.viz import (
    plot_creation_destruction,
    plot_creation_destruction_overlay,
    plot_velocity_overlay,
    plot_transport_spectrum,
)


# ==================== Configuration ====================

DATA_CSV = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv")
OUTPUT_DIR = Path("results/mcolon/20260121_uot-mvp/consecutive_frames")

# Frame intervals to test (in frame units)
# ~19.2 min between frames, so:
# - 1 frame = ~0.32 hr
# - 3 frames = ~0.96 hr
# - 6 frames = ~1.92 hr
FRAME_INTERVALS = [1, 3, 6]

# Representative embryos to test
TEST_EMBRYOS = [
    "20251113_A05_e01",  # From target list
    "20251113_E04_e01",  # From target list
]

# Starting frames to test (representative developmental stages)
# Frame indices chosen to ensure we have sufficient future frames
STARTING_FRAMES = [80, 100, 120]  # Different time points across development


# ==================== Helper Functions ====================

def create_uot_config() -> UOTConfig:
    """Create standard UOT configuration for MVP."""
    return UOTConfig(
        epsilon=1e-2,
        marginal_relaxation=10.0,
        downsample_factor=4,
        mass_mode=MassMode.UNIFORM,
        max_support_points=5000,
        store_coupling=True,
        random_seed=42,
    )


def load_embryo_data(csv_path: Path, embryo_id: str) -> pd.DataFrame:
    """Load all frames for a specific embryo from CSV."""
    df = pd.read_csv(
        csv_path,
        usecols=["embryo_id", "frame_index", "mask_rle", "mask_height_px", "mask_width_px",
                 "relative_time_s", "predicted_stage_hpf"]
    )
    subset = df[df["embryo_id"] == embryo_id].sort_values("frame_index")
    return subset


def verify_frame_exists(df: pd.DataFrame, frame_index: int) -> bool:
    """Check if a frame exists in the dataframe."""
    return frame_index in df["frame_index"].values


def run_single_pair_analysis(
    csv_path: Path,
    embryo_id: str,
    frame_src: int,
    frame_tgt: int,
    config: UOTConfig,
) -> Tuple[Dict, UOTFramePair]:
    """Run UOT on a single frame pair and extract metrics."""
    print(f"  Processing frames {frame_src} -> {frame_tgt}...")

    # Load mask pair
    pair = load_mask_pair_from_csv(csv_path, embryo_id, frame_src, frame_tgt)

    # Run UOT
    result = run_uot_pair(pair, config=config)

    # Extract metrics
    metrics = result.diagnostics.get("metrics", {})
    metrics_dict = {
        "embryo_id": embryo_id,
        "frame_src": frame_src,
        "frame_tgt": frame_tgt,
        "frame_interval": frame_tgt - frame_src,
        "cost": result.cost,
        **metrics,
    }

    # Add result to pair for visualization
    pair.result = result

    return metrics_dict, pair


def save_visualizations(
    pair: UOTFramePair,
    output_dir: Path,
) -> None:
    """Generate and save all visualization plots."""
    result = pair.result

    # 1. Creation/destruction heatmaps
    fig = plot_creation_destruction(
        result.mass_created_hw,
        result.mass_destroyed_hw,
        output_path=str(output_dir / "creation_destruction.png"),
    )
    plt.close(fig)

    # 1b. Creation/destruction overlay on masks
    fig = plot_creation_destruction_overlay(
        pair.src.embryo_mask,
        pair.tgt.embryo_mask,
        result.mass_created_hw,
        result.mass_destroyed_hw,
        transform_meta=result.transform_meta,
        output_path=str(output_dir / "creation_destruction_overlay.png"),
    )
    plt.close(fig)

    # 2. Velocity field overlay (use creation mask as background since velocity field is downsampled)
    # The velocity field is at the downsampled resolution, so use creation_hw as background
    fig = plot_velocity_overlay(
        result.mass_created_hw,  # Use downsampled resolution
        result.velocity_field_yx_hw2,
        stride=1,  # No additional stride needed since already downsampled
        output_path=str(output_dir / "velocity_field.png"),
    )
    plt.close(fig)

    # 3. Transport spectrum
    if result.coupling is not None:
        fig = plot_transport_spectrum(
            result.coupling,
            result.support_src_yx,
            result.support_tgt_yx,
            output_path=str(output_dir / "transport_spectrum.png"),
        )
        plt.close(fig)

    # 4. Side-by-side mask comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(pair.src.embryo_mask, cmap="gray")
    axes[0].set_title(f"Source (frame {pair.src.meta['frame_index']})")
    axes[0].axis("off")

    axes[1].imshow(pair.tgt.embryo_mask, cmap="gray")
    axes[1].set_title(f"Target (frame {pair.tgt.meta['frame_index']})")
    axes[1].axis("off")

    fig.savefig(output_dir / "side_by_side_masks.png", dpi=200)
    plt.close(fig)


def plot_cost_over_intervals(
    metrics_df: pd.DataFrame,
    embryo_id: str,
    output_dir: Path,
) -> None:
    """Plot cost vs frame interval."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    # Group by interval and compute mean/std
    grouped = metrics_df.groupby("frame_interval")["cost"].agg(["mean", "std", "count"])

    intervals = grouped.index.values
    means = grouped["mean"].values
    stds = grouped["std"].values

    ax.errorbar(intervals, means, yerr=stds, marker="o", capsize=5,
                linewidth=2, markersize=8, label=f"{embryo_id}")

    ax.set_xlabel("Frame interval")
    ax.set_ylabel("UOT cost")
    ax.set_title(f"Cost vs. Frame Interval: {embryo_id}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(output_dir / "cost_over_intervals.png", dpi=200)
    plt.close(fig)


def save_metrics_summary(
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save metrics to CSV and print summary statistics."""
    # Save full metrics
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    # Create summary statistics
    summary = metrics_df.groupby("frame_interval").agg({
        "cost": ["mean", "std", "min", "max"],
        "created_mass": ["mean", "std"],
        "destroyed_mass": ["mean", "std"],
        "mean_transport_distance": ["mean", "std"],
    })

    # Save summary
    summary.to_csv(output_dir / "metrics_summary_stats.csv")

    print("\n=== Metrics Summary ===")
    print(summary)


# ==================== Main Execution ====================

def main():
    """Main execution function."""
    print("=" * 60)
    print("UOT MVP: Consecutive Frames Analysis")
    print("=" * 60)

    # Create UOT configuration
    config = create_uot_config()
    print(f"\nUOT Configuration:")
    print(f"  epsilon: {config.epsilon}")
    print(f"  marginal_relaxation: {config.marginal_relaxation}")
    print(f"  downsample_factor: {config.downsample_factor}")
    print(f"  max_support_points: {config.max_support_points}")
    print(f"  random_seed: {config.random_seed}")

    # Process each embryo
    for embryo_id in TEST_EMBRYOS:
        print(f"\n{'=' * 60}")
        print(f"Processing embryo: {embryo_id}")
        print(f"{'=' * 60}")

        # Create output directory
        embryo_dir = OUTPUT_DIR / embryo_id
        embryo_dir.mkdir(parents=True, exist_ok=True)

        # Load embryo data
        embryo_df = load_embryo_data(DATA_CSV, embryo_id)
        print(f"Loaded {len(embryo_df)} frames for {embryo_id}")

        # Collect metrics across all pairs
        all_metrics = []

        # Test different starting frames
        for start_frame in STARTING_FRAMES:
            if not verify_frame_exists(embryo_df, start_frame):
                print(f"  Skipping start_frame={start_frame} (not found)")
                continue

            print(f"\nStarting from frame {start_frame}:")

            # Test different intervals
            for interval in FRAME_INTERVALS:
                target_frame = start_frame + interval

                if not verify_frame_exists(embryo_df, target_frame):
                    print(f"  Skipping interval={interval} (target frame {target_frame} not found)")
                    continue

                # Create output directory for this pair
                pair_dir = embryo_dir / f"frames_{start_frame}_to_{target_frame}"
                pair_dir.mkdir(parents=True, exist_ok=True)

                try:
                    # Run UOT analysis
                    metrics_dict, pair = run_single_pair_analysis(
                        DATA_CSV, embryo_id, start_frame, target_frame, config
                    )

                    # Save visualizations
                    save_visualizations(pair, pair_dir)

                    # Collect metrics
                    all_metrics.append(metrics_dict)

                    print(f"    Cost: {metrics_dict['cost']:.4f}")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

        # Save metrics summary for this embryo
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            save_metrics_summary(metrics_df, embryo_dir)
            plot_cost_over_intervals(metrics_df, embryo_id, embryo_dir)
            print(f"\nSaved {len(all_metrics)} pair analyses to {embryo_dir}")
        else:
            print(f"\nWARNING: No valid pairs processed for {embryo_id}")

    print("\n" + "=" * 60)
    print("Consecutive frames analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
