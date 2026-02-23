#!/usr/bin/env python3
"""
Quick debug script for quiver plot visualization.

Loads existing UOT result and experiments with different thresholds/scales
WITHOUT re-running the solver (which takes 30-90s each time).

Usage:
    python debug_quiver_viz.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

from src.analyze.optimal_transport_morphometrics.uot_masks import run_uot_pair
from src.analyze.utils.optimal_transport import (
    UOTConfig, UOTFramePair, UOTFrame, MassMode
)

# ==== CONSTANTS ====
CANONICAL_GRID_SHAPE = (256, 576)
UM_PER_PX = 7.8
COORD_SCALE = 1.0 / max(CANONICAL_GRID_SHAPE)

# Use absolute path based on script location
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "debug_params"


def make_circle(shape, center_yx, radius):
    """Create a circle mask."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    cy, cx = center_yx
    mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2
    return mask.astype(np.uint8)


def run_once_and_cache(epsilon=0.1, marginal_relaxation=10.0):
    """Run UOT once with specified params and cache the result."""
    print(f"Running UOT with epsilon={epsilon}, reg_m={marginal_relaxation}...")
    
    # Make identity test
    cy, cx = CANONICAL_GRID_SHAPE[0] // 2, CANONICAL_GRID_SHAPE[1] // 2
    circle = make_circle(CANONICAL_GRID_SHAPE, (cy, cx), 40)
    
    config = UOTConfig(
        epsilon=epsilon,
        marginal_relaxation=marginal_relaxation,
        downsample_factor=1,
        downsample_divisor=1,
        padding_px=16,
        mass_mode=MassMode.UNIFORM,
        align_mode="none",
        max_support_points=5000,
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=COORD_SCALE,
        use_pair_frame=True,
        use_canonical_grid=True,
        canonical_grid_um_per_pixel=UM_PER_PX,
        canonical_grid_shape_hw=CANONICAL_GRID_SHAPE,
        canonical_grid_align_mode="none",
    )
    
    pair = UOTFramePair(
        src=UOTFrame(embryo_mask=circle, meta={"test": "identity", "um_per_pixel": UM_PER_PX}),
        tgt=UOTFrame(embryo_mask=circle, meta={"test": "identity", "um_per_pixel": UM_PER_PX}),
    )
    
    result = run_uot_pair(pair, config=config)
    print("Done!")
    return circle, result


def plot_quiver_experiments(src_mask, result, output_dir):
    """
    Generate multiple quiver plots with different settings.
    This is the main debugging function - tweak parameters here!
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use μm/frame if available (like main script), else fall back to px/frame
    velocity_field = (result.velocity_um_per_frame_yx
                     if result.velocity_um_per_frame_yx is not None
                     else result.velocity_px_per_frame_yx)
    unit_label = "μm/frame" if result.velocity_um_per_frame_yx is not None else "px/frame"
    
    velocity_mag = np.sqrt(velocity_field[..., 0]**2 + velocity_field[..., 1]**2)
    
    # Print velocity statistics
    print("\n=== VELOCITY FIELD STATISTICS ===")
    print(f"Using: {unit_label}")
    print(f"Shape: {velocity_field.shape}")
    print(f"Velocity magnitude range: [{velocity_mag.min():.4f}, {velocity_mag.max():.4f}] {unit_label}")
    print(f"Mean velocity: {velocity_mag.mean():.4f} {unit_label}")
    print(f"Median velocity: {np.median(velocity_mag):.4f} {unit_label}")
    print(f"Non-zero pixels: {(velocity_mag > 0).sum()} / {velocity_mag.size}")
    
    # Percentiles
    nonzero = velocity_mag[velocity_mag > 0]
    if len(nonzero) > 0:
        print(f"\nPercentiles of non-zero velocities:")
        for p in [50, 75, 90, 95, 99]:
            print(f"  {p}th: {np.percentile(nonzero, p):.4f} {unit_label}")

    # Histogram of velocities - PROPORTION-based so we can see zeros!
    hist_path = output_dir / "velocity_histogram.png"
    fig_hist, axes_hist = plt.subplots(1, 3, figsize=(15, 4))
    
    total_pixels = velocity_mag.size
    n_zero = (velocity_mag == 0).sum()
    n_nonzero = (velocity_mag > 0).sum()
    pct_zero = 100.0 * n_zero / total_pixels
    pct_nonzero = 100.0 * n_nonzero / total_pixels
    
    # Plot 1: Bar chart showing zero vs non-zero proportion
    axes_hist[0].bar(['Zero\nvelocity', 'Non-zero\nvelocity'], 
                     [pct_zero, pct_nonzero],
                     color=['forestgreen', 'crimson'], alpha=0.8)
    axes_hist[0].set_ylabel("% of all pixels")
    axes_hist[0].set_title(f"Velocity Distribution\n({n_zero:,} zero, {n_nonzero:,} non-zero)")
    axes_hist[0].set_ylim(0, 100)
    for i, (val, pct) in enumerate([(n_zero, pct_zero), (n_nonzero, pct_nonzero)]):
        axes_hist[0].text(i, pct + 2, f"{pct:.1f}%", ha='center', fontsize=10)
    
    # Plot 2: Histogram of non-zero velocities (with density=True for proportion)
    if len(nonzero) > 0:
        counts, bins, _ = axes_hist[1].hist(nonzero, bins=50, color="steelblue", alpha=0.8, 
                                            weights=np.ones(len(nonzero)) / total_pixels * 100)
        axes_hist[1].set_title(f"Non-zero Velocities\n(as % of all pixels)")
        axes_hist[1].set_xlabel(unit_label)
        axes_hist[1].set_ylabel("% of all pixels")
    else:
        axes_hist[1].text(0.5, 0.5, "No non-zero velocities", ha='center', va='center', transform=axes_hist[1].transAxes)
    
    # Plot 3: Log-scale histogram
    if len(nonzero) > 0:
        axes_hist[2].hist(np.log10(nonzero + 1e-12), bins=50, color="purple", alpha=0.8,
                         weights=np.ones(len(nonzero)) / total_pixels * 100)
        axes_hist[2].set_title(f"Non-zero Velocities (log10)\n(as % of all pixels)")
        axes_hist[2].set_xlabel(f"log10({unit_label})")
        axes_hist[2].set_ylabel("% of all pixels")
    else:
        axes_hist[2].text(0.5, 0.5, "No non-zero velocities", ha='center', va='center', transform=axes_hist[2].transAxes)

    fig_hist.tight_layout()
    fig_hist.savefig(hist_path, dpi=150)
    plt.close(fig_hist)
    print(f"\nSaved velocity histogram to: {hist_path}")
    
    # ==== EXPERIMENT WITH DIFFERENT SETTINGS ====
    # IMPORTANT: All threshold values here are in CURRENT UNITS (px/frame or μm/frame)
    # min_velocity_px_val is the PIXEL threshold (always 1.0 px noise floor)
    min_velocity_px_val = 1.0
    if unit_label == "μm/frame":
        # Convert pixel threshold to μm for these experiments
        min_velocity_um_val = min_velocity_px_val * UM_PER_PX
    else:
        min_velocity_um_val = min_velocity_px_val
    
    # IQR-based threshold (robust to outliers)
    if len(nonzero) > 0:
        q1 = np.percentile(nonzero, 25)
        q3 = np.percentile(nonzero, 75)
        iqr = q3 - q1
        iqr_threshold = q3 + 1.5 * iqr
    else:
        iqr_threshold = 0.0

    experiments = [
        # (name, min_velocity_threshold_in_current_units, min_velocity_pct, quiver_scale, stride)
        ("current_defaults (1px)", min_velocity_um_val, 0.02, 100.0, 4),
        ("high_abs (10px)", min_velocity_um_val * 10, 0.02, 100.0, 4),
        ("high_rel (10%)", min_velocity_um_val, 0.10, 100.0, 4),
        ("larger_scale", min_velocity_um_val, 0.02, 500.0, 4),
        ("huge_scale", min_velocity_um_val, 0.02, 2000.0, 4),
        ("aggressive (20px+20%)", min_velocity_um_val * 20, 0.20, 500.0, 4),
        ("IQR (Q3+1.5*IQR)", iqr_threshold, 0.0, 100.0, 4),
        ("medium_stride (stride=6)", min_velocity_um_val, 0.02, 100.0, 6),
        ("sparse_stride (stride=8)", min_velocity_um_val, 0.02, 100.0, 8),
    ]
    
    n_exp = len(experiments)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    h_vel, w_vel = velocity_field.shape[:2]
    
    for idx, (name, min_velocity_threshold, min_vel_pct, scale, stride) in enumerate(experiments):
        ax = axes[idx]
        
        # Background
        ax.imshow(src_mask, cmap="gray", alpha=0.3, aspect='equal',
                  extent=[0, src_mask.shape[1], src_mask.shape[0], 0])
        
        # Subsample
        yy_vel, xx_vel = np.meshgrid(
            np.arange(0, h_vel, stride), 
            np.arange(0, w_vel, stride), 
            indexing='ij'
        )
        u = velocity_field[::stride, ::stride, 1]
        v = velocity_field[::stride, ::stride, 0]
        mag_sub = velocity_mag[::stride, ::stride]
        
        # Option 4: Combined threshold (absolute + relative)
        max_vel = velocity_mag.max()
        threshold = max(min_velocity_threshold, max_vel * min_vel_pct)
        mask_sub = mag_sub > threshold
        
        n_arrows = mask_sub.sum()
        
        if n_arrows == 0:
            ax.text(0.5, 0.5, "No arrows\n(all filtered)", 
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='green', weight='bold')
        else:
            ax.quiver(
                xx_vel[mask_sub], yy_vel[mask_sub],
                u[mask_sub], v[mask_sub],
                mag_sub[mask_sub],
                cmap="hot",
                scale=scale,
                scale_units='xy',
                angles='xy',
            )
        
        ax.set_title(f"{name}\nthresh={threshold:.1f}, scale={scale}, arrows={n_arrows}", fontsize=9)
        ax.set_xlim(0, w_vel)
        ax.set_ylim(h_vel, 0)
        ax.set_aspect('equal')
    
    # Empty last subplot - use for legend/notes
    ax = axes[-1]
    ax.axis('off')
    
    # Calculate IQR stats for display
    if len(nonzero) > 0:
        q1 = np.percentile(nonzero, 25)
        q3 = np.percentile(nonzero, 75)
        iqr = q3 - q1
        iqr_thresh = q3 + 1.5 * iqr
    else:
        q1 = q3 = iqr = iqr_thresh = 0.0
    
    notes = f"""
    VELOCITY STATS ({unit_label}):
    max: {velocity_mag.max():.1f}
    median: {np.median(velocity_mag):.1f}
    Q1: {q1:.1f}
    Q3: {q3:.1f}
    IQR: {iqr:.1f}
    IQR threshold: {iqr_thresh:.1f}
    
    GOAL for identity test:
    Show NO arrows (all filtered)
    
    KEY INSIGHT:
    Max velocity is huge ({velocity_mag.max():.0f})
    even for identity test. This is 
    likely due to a few random pixels
    getting incorrectly mapped far away.
    
    Using IQR (robust to outliers)
    should filter these better than
    using max velocity.
    """
    ax.text(0.05, 0.95, notes, transform=ax.transAxes, va='top', 
            fontsize=9, family='monospace')
    
    fig.suptitle("Quiver Visualization Experiments - Identity Test", fontsize=14, weight='bold')
    fig.tight_layout()
    
    output_path = output_dir / "quiver_experiments.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved experiments to: {output_path}")
    
    return experiments


if __name__ == "__main__":
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with different epsilons and marginal relaxation
    # Hypothesis: epsilon is the issue, not reg_m
    test_configs = [
        (0.1, 10.0, "eps_0.1_regm_10"),
        (0.1, np.inf, "eps_0.1_regm_inf"),
        (0.01, 10.0, "eps_0.01_regm_10"),
        (0.001, 10.0, "eps_0.001_regm_10"),
    ]
    
    for epsilon, reg_m, suffix in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing with epsilon={epsilon}, marginal_relaxation={reg_m}")
        print('='*60)
        
        # Run UOT
        src_mask, result = run_once_and_cache(epsilon=epsilon, marginal_relaxation=reg_m)
        
        # Run visualization experiments
        plot_quiver_experiments(src_mask, result, output_dir / suffix)
    
    print("\n" + "="*60)
    print("Done! Check the output folders for each configuration.")
    print("="*60)
