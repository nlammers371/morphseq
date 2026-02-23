#!/usr/bin/env python3
"""
UOT MVP: Cross-Embryo Comparison

Compare two different embryos (20251113_A05 vs 20251113_E04) at similar
developmental stages (~48hpf) to test UOT on morphological differences.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

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
    load_mask_from_csv,
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
OUTPUT_DIR = Path("results/mcolon/20260121_uot-mvp/cross_embryo")

# Target embryos for comparison
EMBRYO_A = "20251113_A05_e01"
EMBRYO_B = "20251113_E04_e01"

# Target developmental stage
TARGET_STAGE_HPF = 48.0
STAGE_TOLERANCE_HPF = 1.0  # +/- 1 hour


# ==================== Helper Functions ====================

def create_uot_config(marginal_relaxation: float = 10.0) -> UOTConfig:
    """Create UOT configuration for cross-embryo comparison."""
    return UOTConfig(
        epsilon=1e-2,
        marginal_relaxation=marginal_relaxation,
        downsample_factor=4,
        mass_mode=MassMode.UNIFORM,
        max_support_points=5000,
        store_coupling=True,
        random_seed=42,
    )


def find_frame_at_stage(
    csv_path: Path,
    embryo_id: str,
    target_hpf: float,
    tolerance_hpf: float,
) -> Tuple[Optional[int], Optional[float]]:
    """Find frame closest to target developmental stage."""
    df = pd.read_csv(
        csv_path,
        usecols=["embryo_id", "frame_index", "predicted_stage_hpf"]
    )

    # Filter for this embryo and stage range
    subset = df[
        (df["embryo_id"] == embryo_id) &
        (df["predicted_stage_hpf"] >= target_hpf - tolerance_hpf) &
        (df["predicted_stage_hpf"] <= target_hpf + tolerance_hpf)
    ]

    if subset.empty:
        return None, None

    # Find closest to target
    subset["dist"] = (subset["predicted_stage_hpf"] - target_hpf).abs()
    closest = subset.loc[subset["dist"].idxmin()]

    return int(closest["frame_index"]), float(closest["predicted_stage_hpf"])


def create_cross_embryo_pair(
    csv_path: Path,
    embryo_a_id: str,
    frame_a: int,
    embryo_b_id: str,
    frame_b: int,
) -> UOTFramePair:
    """Create a pseudo-pair from two different embryos."""
    # Load individual frames
    frame_a_obj = load_mask_from_csv(csv_path, embryo_a_id, frame_a)
    frame_b_obj = load_mask_from_csv(csv_path, embryo_b_id, frame_b)

    # Create pair
    pair = UOTFramePair(
        src=frame_a_obj,
        tgt=frame_b_obj,
        pair_meta={
            "comparison_type": "cross_embryo",
            "embryo_a": embryo_a_id,
            "embryo_b": embryo_b_id,
            "frame_a": frame_a,
            "frame_b": frame_b,
        }
    )

    return pair


def run_cross_embryo_analysis(
    pair: UOTFramePair,
    config: UOTConfig,
) -> Dict:
    """Run UOT on cross-embryo pair and extract metrics."""
    print(f"  Running UOT between {pair.pair_meta['embryo_a']} and {pair.pair_meta['embryo_b']}...")

    # Run UOT
    result = run_uot_pair(pair, config=config)

    # Extract metrics
    metrics = result.diagnostics.get("metrics", {})
    metrics_dict = {
        "embryo_a": pair.pair_meta["embryo_a"],
        "embryo_b": pair.pair_meta["embryo_b"],
        "frame_a": pair.pair_meta["frame_a"],
        "frame_b": pair.pair_meta["frame_b"],
        "stage_a_hpf": pair.src.meta.get("predicted_stage_hpf"),
        "stage_b_hpf": pair.tgt.meta.get("predicted_stage_hpf"),
        "cost": result.cost,
        **metrics,
    }

    # Attach result to pair for visualization
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

    # 4. Side-by-side comparison with labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].imshow(pair.src.embryo_mask, cmap="gray")
    stage_a = pair.src.meta.get('predicted_stage_hpf', 'N/A')
    stage_a_str = f"{stage_a:.1f}" if isinstance(stage_a, (int, float)) else str(stage_a)
    axes[0].set_title(
        f"{pair.pair_meta['embryo_a']}\n"
        f"Frame {pair.pair_meta['frame_a']} "
        f"({stage_a_str} hpf)"
    )
    axes[0].axis("off")

    axes[1].imshow(pair.tgt.embryo_mask, cmap="gray")
    stage_b = pair.tgt.meta.get('predicted_stage_hpf', 'N/A')
    stage_b_str = f"{stage_b:.1f}" if isinstance(stage_b, (int, float)) else str(stage_b)
    axes[1].set_title(
        f"{pair.pair_meta['embryo_b']}\n"
        f"Frame {pair.pair_meta['frame_b']} "
        f"({stage_b_str} hpf)"
    )
    axes[1].axis("off")

    fig.suptitle("Cross-Embryo Comparison", fontsize=14, fontweight="bold")
    fig.savefig(output_dir / "side_by_side_comparison.png", dpi=200)
    plt.close(fig)

    # 5. Overlay with transparency showing differences
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    # Create RGB overlay: red=src only, green=tgt only, yellow=overlap
    src_mask = pair.src.embryo_mask.astype(bool)
    tgt_mask = pair.tgt.embryo_mask.astype(bool)

    overlay = np.zeros((*src_mask.shape, 3))
    overlay[src_mask & ~tgt_mask] = [1, 0, 0]  # Red: only in A
    overlay[~src_mask & tgt_mask] = [0, 1, 0]  # Green: only in B
    overlay[src_mask & tgt_mask] = [1, 1, 0]   # Yellow: overlap

    ax.imshow(overlay)
    ax.set_title("Overlay: Red=A only, Green=B only, Yellow=overlap")
    ax.axis("off")

    fig.savefig(output_dir / "overlay_with_difference.png", dpi=200)
    plt.close(fig)


def save_metrics_summary(
    metrics_dict: Dict,
    output_dir: Path,
) -> None:
    """Save metrics to text file with formatted summary."""
    output_file = output_dir / "metrics_summary.txt"

    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Cross-Embryo UOT Comparison Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Embryo A: {metrics_dict['embryo_a']}\n")
        f.write(f"  Frame: {metrics_dict['frame_a']}\n")
        f.write(f"  Stage: {metrics_dict['stage_a_hpf']:.2f} hpf\n\n")

        f.write(f"Embryo B: {metrics_dict['embryo_b']}\n")
        f.write(f"  Frame: {metrics_dict['frame_b']}\n")
        f.write(f"  Stage: {metrics_dict['stage_b_hpf']:.2f} hpf\n\n")

        f.write("-" * 60 + "\n")
        f.write("Transport Metrics\n")
        f.write("-" * 60 + "\n")
        f.write(f"UOT Cost: {metrics_dict['cost']:.4f}\n")

        # Print other metrics if available
        metric_names = [
            "transported_mass", "created_mass", "destroyed_mass",
            "mean_transport_distance", "max_transport_distance",
            "transport_mass_fraction"
        ]

        for metric in metric_names:
            if metric in metrics_dict:
                f.write(f"{metric}: {metrics_dict[metric]:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"  Saved metrics summary to {output_file}")


# ==================== Main Execution ====================

def main():
    """Main execution function."""
    print("=" * 60)
    print("UOT MVP: Cross-Embryo Comparison")
    print("=" * 60)

    # Find frames at target developmental stage
    print(f"\nFinding frames at ~{TARGET_STAGE_HPF} hpf...")

    frame_a, stage_a = find_frame_at_stage(
        DATA_CSV, EMBRYO_A, TARGET_STAGE_HPF, STAGE_TOLERANCE_HPF
    )
    frame_b, stage_b = find_frame_at_stage(
        DATA_CSV, EMBRYO_B, TARGET_STAGE_HPF, STAGE_TOLERANCE_HPF
    )

    if frame_a is None:
        print(f"ERROR: Could not find {EMBRYO_A} at {TARGET_STAGE_HPF} +/- {STAGE_TOLERANCE_HPF} hpf")
        return

    if frame_b is None:
        print(f"ERROR: Could not find {EMBRYO_B} at {TARGET_STAGE_HPF} +/- {STAGE_TOLERANCE_HPF} hpf")
        return

    print(f"  {EMBRYO_A}: frame {frame_a} ({stage_a:.2f} hpf)")
    print(f"  {EMBRYO_B}: frame {frame_b} ({stage_b:.2f} hpf)")

    # Create UOT configuration
    config = create_uot_config()
    print(f"\nUOT Configuration:")
    print(f"  epsilon: {config.epsilon}")
    print(f"  marginal_relaxation: {config.marginal_relaxation}")
    print(f"  downsample_factor: {config.downsample_factor}")
    print(f"  max_support_points: {config.max_support_points}")

    # Create output directory
    comparison_name = f"{EMBRYO_A.split('_')[1]}_vs_{EMBRYO_B.split('_')[1]}_{TARGET_STAGE_HPF:.0f}hpf"
    output_dir = OUTPUT_DIR / comparison_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Create cross-embryo pair
    print(f"\nLoading masks...")
    pair = create_cross_embryo_pair(
        DATA_CSV, EMBRYO_A, frame_a, EMBRYO_B, frame_b
    )

    # Run UOT analysis
    print(f"\nRunning UOT analysis...")
    metrics_dict, pair = run_cross_embryo_analysis(pair, config)

    print(f"\n=== Results ===")
    print(f"UOT Cost: {metrics_dict['cost']:.4f}")
    if "created_mass" in metrics_dict:
        print(f"Created mass: {metrics_dict['created_mass']:.4f}")
    if "destroyed_mass" in metrics_dict:
        print(f"Destroyed mass: {metrics_dict['destroyed_mass']:.4f}")
    if "mean_transport_distance" in metrics_dict:
        print(f"Mean transport distance: {metrics_dict['mean_transport_distance']:.2f} px")

    # Save visualizations
    print(f"\nGenerating visualizations...")
    save_visualizations(pair, output_dir)

    # Save metrics
    save_metrics_summary(metrics_dict, output_dir)

    # Also save as CSV for easy loading
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    print("\n" + "=" * 60)
    print("Cross-embryo comparison complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
