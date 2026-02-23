#!/usr/bin/env python3
"""
UOT Parameter Debugging and Sensitization Script

    Sequential testing approach:
    1. Test 1: Identity
    2. Test 2: Non-overlapping circles
    3. Test 3: Shape change
    4. Test 4: Combined transport + shape change

Records all metrics for posterity and generates visualizations.
Real embryo testing is beyond scope - synthetic only.

USAGE:
    python results/mcolon/20260121_uot-mvp/debug_uot_params.py --test 1
    python results/mcolon/20260121_uot-mvp/debug_uot_params.py --test all

OUTPUT:
    - Results CSV per test with all parameter combinations
    - Visualizations per parameter combination
    - Parameter sensitivity plots
    - recommended_params.json with viable ranges
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import argparse
import time
import os

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
from matplotlib.patches import Patch

from src.analyze.optimal_transport_morphometrics.uot_masks import (
    run_uot_pair,
    plot_uot_quiver,
    plot_uot_cost_field,
    plot_uot_creation_destruction,
    plot_uot_overlay_with_transport,
    UOTVizConfig,
    DEFAULT_UOT_VIZ_CONFIG,
)
from src.analyze.optimal_transport_morphometrics.uot_masks.frame_mask_io import (
    load_mask_pair_from_csv,
    load_mask_from_csv,
)
# preprocess_pair_canonical: transforms real embryo masks to canonical grid for plotting
# (synthetic tests skip this since masks are already on canonical grid)
from src.analyze.optimal_transport_morphometrics.uot_masks.preprocess import preprocess_pair_canonical
from src.analyze.optimal_transport_morphometrics.uot_masks.uot_grid import CanonicalGridConfig
from src.analyze.optimal_transport_morphometrics.uot_masks.viz import (
    _overlay_masks_rgb,
    _plot_extent,
    _set_axes_limits,
    _quiver_transform,
)
from src.analyze.utils.optimal_transport import (
    UOTConfig, UOTFramePair, UOTFrame, UOTResult, MassMode, POTBackend
)
import ot

# ==== CONSTANTS ====

# CANONICAL GRID - All masks are created DIRECTLY on this grid
# NOTE: With pair_frame enabled, these are now properly tracked through the pipeline
# rather than hard-coded in every function
CANONICAL_GRID_SHAPE = (256, 576)  # Height x Width in pixels
CANONICAL_UM_PER_PX = 10.0  # Micrometers per pixel
# Physical dimensions: 1996.8 μm (2.0 mm) × 4492.8 μm (4.5 mm)

# All test masks should be created on canonical grid
IMAGE_SHAPE = CANONICAL_GRID_SHAPE
UM_PER_PX = CANONICAL_UM_PER_PX
COORD_SCALE = 1.0 / max(CANONICAL_GRID_SHAPE)  # Scale based on max dimension

# Parameter grid
EPSILON_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
REG_M_GRID = [0.1, 1.0, 10.0, 100.0,  np.inf]

# Quick mode parameter grid (reduced for faster testing)
QUICK_EPSILON_GRID = [1e-1, 1e-0]
QUICK_REG_M_GRID = [1.0, 10.0]

# Use absolute path based on script location
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "debug_params"

# Display mode for plots: "image" (y down) or "cartesian" (y up)
# Now handled by viz module, but keep for backward compatibility
DISPLAY_MODE = os.environ.get("MORPHSEQ_DISPLAY_MODE", "image").lower()

# Optional real-embryo defaults (from DEBUG_PARAMS_README / cross-embryo comparison)
DEFAULT_REAL_DATA_CSV = Path(
    "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)
DEFAULT_EMBRYO_A = "20251113_A05_e01"
DEFAULT_EMBRYO_B = "20251113_E04_e01"
DEFAULT_TARGET_STAGE_HPF = 48.0
DEFAULT_STAGE_TOLERANCE_HPF = 1.0


# Use UOTVizConfig from viz module (backward compatibility alias)
VisualizationConfig = UOTVizConfig
VIZ_CONFIG = DEFAULT_UOT_VIZ_CONFIG


# ==== TEST CASE DEFINITIONS ====

def make_circle(shape: Tuple[int, int], center_yx: Tuple[int, int], radius: int) -> np.ndarray:
    """Create a circle mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def make_ellipse(
    shape: Tuple[int, int], center_yx: Tuple[int, int], radius_y: int, radius_x: int
) -> np.ndarray:
    """Create an ellipse mask."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = ((yy - cy) / float(radius_y)) ** 2 + ((xx - cx) / float(radius_x)) ** 2 <= 1.0
    return mask.astype(np.uint8)


def make_identity_test(shape: Tuple[int, int], radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Test 1: Identity (null test) - Circle to same circle."""
    cy, cx = shape[0] // 2, shape[1] // 2
    circle = make_circle(shape, (cy, cx), radius)
    return circle, circle


def make_nonoverlap_test(
    shape: Tuple[int, int], radius: int, separation: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Test 2: Non-overlapping circles (pure transport) - No shape change."""
    cy, cx = shape[0] // 2, shape[1] // 2
    src = make_circle(shape, (cy - separation // 2, cx), radius)
    tgt = make_circle(shape, (cy + separation // 2, cx), radius)
    return src, tgt


def make_shape_change_test(shape: Tuple[int, int], radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Test 3: Circle to oval (shape change) - Same centroid."""
    cy, cx = shape[0] // 2, shape[1] // 2
    circle = make_circle(shape, (cy, cx), radius)
    # Oval with same area as circle: π*r² = π*ry*rx, so ry*rx = r²
    # Let's make it 1.5x wider and proportionally shorter
    radius_x = int(radius * 1.5)
    radius_y = int(radius * radius / radius_x)
    oval = make_ellipse(shape, (cy, cx), radius_y, radius_x)
    return circle, oval


def make_combined_test(
    shape: Tuple[int, int], radius: int, shift: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Test 4: Circle to shifted oval (combined transport + shape change)."""
    cy, cx = shape[0] // 2, shape[1] // 2
    circle = make_circle(shape, (cy, cx), radius)
    radius_x = int(radius * 1.5)
    radius_y = int(radius * radius / radius_x)
    oval = make_ellipse(shape, (cy + shift, cx + shift), radius_y, radius_x)
    return circle, oval


# ==== DIAGNOSTICS ====

def compute_surface_metrics(mask: np.ndarray, um_per_px: float) -> Dict[str, float]:
    """Track surface area in physical units."""
    area_px = float(mask.sum())
    area_um2 = area_px * (um_per_px ** 2)

    # Simple perimeter estimation via edge detection
    from scipy import ndimage
    edges = ndimage.sobel(mask.astype(float))
    perimeter_px = float((edges > 0).sum())
    perimeter_um = perimeter_px * um_per_px

    return {
        "area_px": area_px,
        "area_um2": area_um2,
        "perimeter_px": perimeter_px,
        "perimeter_um": perimeter_um,
    }


def diagnose_cost_matrix(cost_matrix: np.ndarray, epsilon: float) -> Dict[str, float]:
    """Analyze cost matrix for numerical health."""
    return {
        "cost_min": float(cost_matrix.min()),
        "cost_max": float(cost_matrix.max()),
        "cost_mean": float(cost_matrix.mean()),
        "cost_std": float(cost_matrix.std()),
        "cost_ratio_to_epsilon": float(cost_matrix.mean() / epsilon),
    }


def diagnose_gibbs_kernel(cost_matrix: np.ndarray, epsilon: float) -> Dict[str, float]:
    """
    Check K = exp(-C/epsilon) for numerical health.
    The solver operates on K, not C. If K is all zeros or ones, solver fails silently.
    """
    K = np.exp(-cost_matrix / epsilon)
    K_nonzero = K[K > 0]

    return {
        "K_min": float(K.min()),
        "K_max": float(K.max()),
        "K_mean": float(K.mean()),
        "K_zeros": int((K == 0).sum()),           # Underflow indicator
        "K_ones": int((K == 1).sum()),            # Epsilon too large
        "K_dynamic_range": float(K.max() / max(K_nonzero.min(), 1e-20)) if len(K_nonzero) > 0 else 0.0,
        "K_healthy": bool(K.min() > 1e-10 and K.max() < 1 - 1e-10),  # Has variation
    }


def diagnose_coupling_sparsity(coupling: np.ndarray, threshold: float = 1e-6) -> Dict[str, float]:
    """
    Biological motion is local - coupling should be sparse.
    Low sparsity = mass diffusion = epsilon too high.
    """
    coupling_arr = np.asarray(coupling)
    total_entries = coupling_arr.size
    nonzero_entries = int((coupling_arr > threshold).sum())
    sparsity = 1 - (nonzero_entries / total_entries)

    return {
        "sparsity": sparsity,              # Should be > 0.9 for biological transport
        "nonzero_entries": nonzero_entries,
        "is_sparse": bool(sparsity > 0.8),       # Warning threshold
    }


def compute_velocity_metrics(velocity_field_yx_hw2: np.ndarray) -> Dict[str, float]:
    """Compute velocity field statistics."""
    velocity_mag = np.sqrt(velocity_field_yx_hw2[..., 0]**2 + velocity_field_yx_hw2[..., 1]**2)
    velocity_nonzero = velocity_mag[velocity_mag > 0]

    return {
        "mean_velocity_px": float(velocity_mag.mean()),
        "max_velocity_px": float(velocity_mag.max()),
        "mean_nonzero_velocity_px": float(velocity_nonzero.mean()) if len(velocity_nonzero) > 0 else 0.0,
        "velocity_has_nan": bool(np.isnan(velocity_mag).any()),
    }


def compute_mass_metrics(
    result: 'UOTResult',  # Pass full result instead of individual params
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute mass metrics using UOTResult properties.
    
    ⚠️ CRITICAL CONTRACT VIOLATION WARNING ⚠️
    created_mass_pct and destroyed_mass_pct are calculated from src_support points ONLY,
    NOT from entire source/target masks. These ~5000 support points represent a sampling
    of the full source/target. Percentages are: (mass_at_support / total_mass) × 100.
    
    These are SAMPLING-BASED ESTIMATES. Failing to make this distinction will lead to bugs!
    """
    metrics = result.diagnostics.get("metrics", {}) if result.diagnostics else {}
    backend = result.diagnostics.get("backend", {}) if result.diagnostics else {}

    m_src = float(backend.get("m_src", np.nan))
    m_tgt = float(backend.get("m_tgt", np.nan))
    created_mass = float(metrics.get("created_mass", np.nan))
    destroyed_mass = float(metrics.get("destroyed_mass", np.nan))
    transported_mass = float(metrics.get("transported_mass", np.nan))

    # Use UOTResult properties for μm² areas (handles pair_frame internally)
    created_area_um2 = float("nan")
    destroyed_area_um2 = float("nan")
    if result.mass_created_um2 is not None:
        created_area_um2 = float(result.mass_created_um2.sum())
        destroyed_area_um2 = float(result.mass_destroyed_um2.sum())

    # Extract percentage metrics (already computed by pipeline)
    created_mass_pct = float(metrics.get("created_mass_pct", np.nan))
    destroyed_mass_pct = float(metrics.get("destroyed_mass_pct", np.nan))
    proportion_transported = float(metrics.get("proportion_transported", np.nan))

    return {
        "m_src": m_src,
        "m_tgt": m_tgt,
        "transported_mass": transported_mass,
        "created_mass": created_mass,
        "destroyed_mass": destroyed_mass,
        "created_area_um2": created_area_um2,  # From property, not manual calc
        "destroyed_area_um2": destroyed_area_um2,  # From property, not manual calc
        "created_mass_pct": created_mass_pct,
        "destroyed_mass_pct": destroyed_mass_pct,
        "proportion_transported": proportion_transported,
    }


# ==== VISUALIZATION ====

def plot_input_masks_with_metrics(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    src_metrics: Dict[str, float],
    tgt_metrics: Dict[str, float],
    output_path: Path,
    title: str,
) -> None:
    """Plot input masks with annotated metrics.
    
    Uses explicit extent to ensure coordinate consistency with OT outputs.
    Both masks should be on canonical grid (256×576).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use canonical grid dimensions for extent
    canon_h, canon_w = CANONICAL_GRID_SHAPE
    h_src, w_src = src_mask.shape
    h_tgt, w_tgt = tgt_mask.shape

    # Source mask
    extent, origin = _plot_extent((h_src, w_src))
    axes[0].imshow(src_mask, cmap="gray", aspect='equal',
                   extent=extent, origin=origin, interpolation='nearest')
    axes[0].set_title(
        f"Source (shape: {src_mask.shape})\n"
        f"Area: {src_metrics['area_um2']:.1f} μm²\n"
        f"Perimeter: {src_metrics['perimeter_um']:.1f} μm"
    )
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    _set_axes_limits(axes[0], (canon_h, canon_w))

    # Target mask  
    extent, origin = _plot_extent((h_tgt, w_tgt))
    axes[1].imshow(tgt_mask, cmap="gray", aspect='equal',
                   extent=extent, origin=origin, interpolation='nearest')
    axes[1].set_title(
        f"Target (shape: {tgt_mask.shape})\n"
        f"Area: {tgt_metrics['area_um2']:.1f} μm²\n"
        f"Perimeter: {tgt_metrics['perimeter_um']:.1f} μm"
    )
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("y (px)")
    _set_axes_limits(axes[1], (canon_h, canon_w))

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cost_and_gibbs(
    cost_matrix: np.ndarray,
    epsilon: float,
    cost_diag: Dict[str, float],
    gibbs_diag: Dict[str, float],
    output_path: Path,
) -> None:
    """Plot cost matrix and Gibbs kernel with diagnostics."""
    K = np.exp(-cost_matrix / epsilon)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Cost matrix
    im0 = axes[0].imshow(cost_matrix, cmap="viridis")
    axes[0].set_title(f"Cost Matrix\nMean: {cost_diag['cost_mean']:.2e}")
    plt.colorbar(im0, ax=axes[0])

    # Log10(cost)
    log_cost = np.log10(cost_matrix + 1e-20)
    im1 = axes[1].imshow(log_cost, cmap="viridis")
    axes[1].set_title(f"Log10(Cost)\nRange: [{log_cost.min():.1f}, {log_cost.max():.1f}]")
    plt.colorbar(im1, ax=axes[1])

    # Gibbs kernel
    im2 = axes[2].imshow(K, cmap="viridis")
    axes[2].set_title(
        f"Gibbs Kernel (K=exp(-C/ε))\nHealthy: {gibbs_diag['K_healthy']}\n"
        f"Zeros: {gibbs_diag['K_zeros']}, Range: [{gibbs_diag['K_min']:.2e}, {gibbs_diag['K_max']:.2e}]"
    )
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"Cost Matrix Analysis (ε={epsilon:.2e})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_diagnostics_sidecar(
    output_path: Path,
    support_mask: np.ndarray,
    velocity_field: np.ndarray,
    result: 'UOTResult',
) -> None:
    """Write diagnostics JSON sidecar following PLOTTING_CONTRACT.md requirement 6.
    
    Provides machine-readable diagnostics for each plot.
    """
    velocity_mag = np.sqrt(velocity_field[..., 0]**2 + velocity_field[..., 1]**2)
    
    # Support coverage
    support_pct = 100.0 * support_mask.sum() / support_mask.size
    
    # Velocity statistics on support only
    if support_mask.any():
        support_velocities = velocity_mag[support_mask]
        stats = {
            "p10": float(np.percentile(support_velocities, 10)),
            "p25": float(np.percentile(support_velocities, 25)),
            "p50": float(np.percentile(support_velocities, 50)),
            "p75": float(np.percentile(support_velocities, 75)),
            "p90": float(np.percentile(support_velocities, 90)),
            "p95": float(np.percentile(support_velocities, 95)),
            "p99": float(np.percentile(support_velocities, 99)),
            "max": float(support_velocities.max()),
            "mean": float(support_velocities.mean()),
            "std": float(support_velocities.std()),
        }
    else:
        stats = {k: 0.0 for k in ["p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean", "std"]}
    
    diagnostics = {
        "support_coverage": {
            "n_pixels_total": int(support_mask.size),
            "n_pixels_defined": int(support_mask.sum()),
            "pct_defined": float(support_pct),
        },
        "velocity_statistics": stats,
        "unit": "μm/frame" if result.velocity_um_per_frame_yx is not None else "px/frame",
        "resolution_hw": list(velocity_field.shape[:2]),
        "contract_version": "1.0",  # Track which plotting contract this follows
    }
    
    import json
    sidecar_path = output_path.parent / (output_path.stem + "_diagnostics.json")
    with open(sidecar_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)


def plot_flow_field(
    src_mask: np.ndarray,
    result: 'UOTResult',  # Pass full result to access properties
    proportion_transported: float,
    output_path: Path,
    stride: int = None,  # If None, use viz_config default
    viz_config: VisualizationConfig = None,
) -> None:
    """Plot velocity field as quiver plot overlaying source mask.
    
    PLOTTING CONTRACT ENFORCED:
    - Uses NaN for non-support regions (not zeros)
    - Shows support mask explicitly
    - Displays statistics on support points only
    - No fabrication via smoothing

    CRITICAL: Data should already be on canonical grid from preprocessing.
    We just display it as-is with proper aspect ratio and coordinate labels.
    NO stretching or remapping - just display on canonical grid coordinates.

    Uses UOTResult properties for unit conversion - no manual calculations.
    """
    if viz_config is None:
        viz_config = VIZ_CONFIG
    
    if stride is None:
        stride = viz_config.quiver_stride

    # Use property for μm/frame (fallback to pixels if pair_frame unavailable)
    velocity_field = (result.velocity_um_per_frame_yx
                     if result.velocity_um_per_frame_yx is not None
                     else result.velocity_px_per_frame_yx)

    velocity_mag = np.sqrt(velocity_field[..., 0]**2 + velocity_field[..., 1]**2)

    # Set label based on which units we're using
    unit_label = "μm/frame" if result.velocity_um_per_frame_yx is not None else "px/frame"

    # Create support mask: pixels with non-zero velocity
    support_mask = velocity_mag > 0
    support_pct = 100.0 * support_mask.sum() / support_mask.size
    
    # PLOTTING CONTRACT: Replace zeros with NaN outside support
    velocity_mag_masked = velocity_mag.copy()
    velocity_mag_masked[~support_mask] = np.nan
    
    # Statistics on support points only
    if support_mask.any():
        support_velocities = velocity_mag[support_mask]
        p50 = np.percentile(support_velocities, 50)
        p90 = np.percentile(support_velocities, 90)
        p99 = np.percentile(support_velocities, 99)
        v_max = support_velocities.max()
    else:
        p50 = p90 = p99 = v_max = 0.0

    # Use CANONICAL grid dimensions
    canon_h, canon_w = CANONICAL_GRID_SHAPE
    h_vel, w_vel = velocity_field.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Support mask (shows which pixels are defined)
    extent, origin = _plot_extent((h_vel, w_vel))
    axes[0].imshow(support_mask, cmap="gray", aspect='equal',
                   extent=extent, origin=origin, interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title(f"Support Coverage\n{support_pct:.2f}% defined ({support_mask.sum():,} pixels)")
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    _set_axes_limits(axes[0], (canon_h, canon_w))

    # Panel 2: Velocity magnitude (NaN outside support)
    im1 = axes[1].imshow(velocity_mag_masked, cmap="viridis", aspect='equal',
                         extent=extent, origin=origin, interpolation='nearest',
                         vmin=viz_config.velocity_vmin, vmax=viz_config.velocity_vmax)
    axes[1].set_title(
        f"Velocity Magnitude (support only)\n"
        f"p50/p90/p99: {p50:.1f}/{p90:.1f}/{p99:.1f} {unit_label}\n"
        f"max: {v_max:.1f} {unit_label}"
    )
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("y (px)")
    _set_axes_limits(axes[1], (canon_h, canon_w))
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=unit_label)

    # Panel 3: Velocity histogram (support only)
    if support_mask.any():
        axes[2].hist(support_velocities, bins=50, color="steelblue", alpha=0.8, edgecolor='black')
        axes[2].axvline(p50, color='orange', linestyle='--', label=f'p50: {p50:.1f}')
        axes[2].axvline(p90, color='red', linestyle='--', label=f'p90: {p90:.1f}')
        axes[2].set_title(f"Velocity Distribution\n({support_mask.sum():,} support points)")
        axes[2].set_xlabel(unit_label)
        axes[2].set_ylabel("Count")
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No support points\n(No transport)", 
                    transform=axes[2].transAxes, ha='center', va='center',
                    fontsize=12, style='italic')
        axes[2].set_xlabel(unit_label)
        axes[2].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    # PLOTTING CONTRACT: Write diagnostics sidecar
    write_diagnostics_sidecar(output_path, support_mask, velocity_field, result)


# plot_flow_field_quiver is now plot_uot_quiver (imported from viz module)


# _overlay_masks_rgb is now imported from viz module


def plot_mask_overlay_only(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Plot neutral overlay of source/target masks only."""
    h, w = src_mask.shape
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    overlay = _overlay_masks_rgb(src_mask, tgt_mask)
    extent, origin = _plot_extent((h, w))
    ax.imshow(overlay, extent=extent, origin=origin, interpolation='nearest')
    _set_axes_limits(ax, (CANONICAL_GRID_SHAPE[0], CANONICAL_GRID_SHAPE[1]))
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Source/Target Overlay (Masks Only)")
    legend_handles = [
        Patch(facecolor=(1.0, 0.80, 0.80), edgecolor="none", label="Source"),
        Patch(facecolor=(0.80, 0.85, 1.0), edgecolor="none", label="Target"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# plot_overlay_transport_field is now plot_uot_overlay_with_transport (imported from viz module)


# plot_transport_cost_field is now plot_uot_cost_field (imported from viz module)


# plot_creation_destruction_maps is now plot_uot_creation_destruction (imported from viz module)


def plot_sensitivity_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str,
    log_scale: bool = False,
) -> None:
    """Create 2D heatmap of parameter sensitivity."""
    # Pivot table with epsilon as rows, reg_m as columns
    pivot = df.pivot_table(values=metric_col, index='epsilon', columns='marginal_relaxation')

    fig, ax = plt.subplots(figsize=(10, 6))

    if log_scale:
        im = ax.imshow(np.log10(pivot.values + 1e-20), aspect='auto', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'log10({metric_col})')
    else:
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_col)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    ax.set_xlabel('marginal_relaxation')

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.0e}" for x in pivot.index])
    ax.set_ylabel('epsilon')

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_comparison_grid(
    output_dir: Path,
    df: pd.DataFrame,
    test_name: str,
    metric_to_show: str = "created_mass_pct",
    quick_mode: bool = False,
) -> None:
    """
    Create a comparison grid showing key results for all parameter combinations.

    This allows visual inspection of how different parameters perform.
    Shows normalized metrics for each combination in a grid layout.
    """
    # Sort by epsilon and marginal_relaxation for consistent ordering
    df_sorted = df.sort_values(['epsilon', 'marginal_relaxation'])

    # Determine grid dimensions based on mode
    epsilon_grid = QUICK_EPSILON_GRID if quick_mode else EPSILON_GRID
    reg_m_grid = QUICK_REG_M_GRID if quick_mode else REG_M_GRID
    n_cols = len(reg_m_grid)
    n_rows = len(epsilon_grid)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each parameter combination
    for idx, row in df_sorted.iterrows():
        eps = row['epsilon']
        regm = row['marginal_relaxation']

        # Find grid position
        eps_idx = epsilon_grid.index(eps)
        regm_idx = reg_m_grid.index(regm)

        ax = axes[eps_idx, regm_idx]

        # Show normalized metric values as text
        metric_val = row.get(metric_to_show, np.nan)
        cost_val = row.get('cost', np.nan)
        proportion_transported = row.get('proportion_transported', np.nan)
        stable = row.get('numerical_stable', False)

        status_icon = "✓" if stable else "✗"
        color = "green" if stable else "red"

        # Format metric display based on type
        if 'pct' in metric_to_show:
            metric_display = f"{metric_to_show}={metric_val:.2f}%"
        elif metric_to_show == 'proportion_transported':
            metric_display = f"proportion={proportion_transported*100:.2f}%"
        else:
            metric_display = f"{metric_to_show}={metric_val:.2e}"

        ax.text(
            0.5, 0.5,
            f"ε={eps:.0e}\nreg_m={regm:.0e}\n{status_icon}\n"
            f"cost={cost_val:.2e}\n{metric_display}",
            ha='center', va='center', fontsize=8,
            color=color, weight='bold',
            transform=ax.transAxes
        )
        # Use CANONICAL grid dimensions
        canon_h, canon_w = CANONICAL_GRID_SHAPE
        ax.set_xlim(0, canon_w)
        ax.set_ylim(canon_h, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add border color based on stability
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.suptitle(f"{test_name} - Parameter Comparison\n{metric_to_show}", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_comparison_grid.png", dpi=150)
    plt.close(fig)
    print(f"  Saved comparison grid to {output_dir / 'parameter_comparison_grid.png'}")


# ==== TEST EXECUTION ====

def run_single_param_combo(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    epsilon: float,
    marginal_relaxation: float,
    output_dir: Path,
    test_name: str,
    pair_override: Optional[UOTFramePair] = None,
) -> Dict:
    """Run UOT with a single parameter combination and collect all metrics."""

    # Create output directory
    param_dir = output_dir / f"eps_{epsilon:.0e}_regm_{marginal_relaxation:.0e}"
    param_dir.mkdir(parents=True, exist_ok=True)

    # Create UOT config
    canonical_align_mode = "none"
    if pair_override is not None:
        # Real embryo masks: enable canonical alignment (yolk-based if available)
        canonical_align_mode = "yolk"
    config = UOTConfig(
        epsilon=epsilon,
        marginal_relaxation=marginal_relaxation,
        downsample_factor=1,  # No downsampling on synthetic tests
        downsample_divisor=1,
        padding_px=16,  # Use proper padding (was 0)
        mass_mode=MassMode.UNIFORM,
        align_mode="none",
        max_support_points=5000,
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=COORD_SCALE,
        use_pair_frame=True,  # Enable pair frame
        # NEW: Tell pipeline masks are already on canonical grid
        use_canonical_grid=True,
        canonical_grid_um_per_pixel=UM_PER_PX,
        canonical_grid_shape_hw=CANONICAL_GRID_SHAPE,
        canonical_grid_align_mode=canonical_align_mode,
        canonical_grid_center_mode="joint_centering",
    )

    # Create frame pair
    if pair_override is not None:
        pair = pair_override
    else:
        pair = UOTFramePair(
            src=UOTFrame(embryo_mask=src_mask, meta={"test": test_name, "um_per_pixel": UM_PER_PX}),
            tgt=UOTFrame(embryo_mask=tgt_mask, meta={"test": test_name, "um_per_pixel": UM_PER_PX}),
        )

    # =========================================================================
    # COORDINATE ALIGNMENT FOR PLOTTING
    # =========================================================================
    # OT outputs (velocity, mass creation/destruction) are always on CANONICAL grid.
    # For plotting to be consistent, input masks must be on the same grid.
    #
    # Two cases:
    # 1. SYNTHETIC tests: masks are created directly on canonical grid (256×576 @ 7.8 μm/px)
    #    → Use directly, no transformation needed
    #
    # 2. REAL embryo data (pair_override): masks come from CSV at native resolution
    #    (e.g., 1200×2400 @ ~1.5 μm/px) → Must transform to canonical grid to match OT outputs
    #
    # preprocess_pair_canonical() applies:
    #   1. Scale from native resolution to 7.8 μm/px
    #   2. Rotate (yolk-based or PCA orientation)
    #   3. Crop/pad to 256×576 centered on joint centroid
    # =========================================================================
    
    if pair_override is not None:
        # REAL DATA: transform to canonical grid for plot consistency
        canonical_config = CanonicalGridConfig(
            reference_um_per_pixel=config.canonical_grid_um_per_pixel,
            grid_shape_hw=config.canonical_grid_shape_hw,
            align_mode=config.canonical_grid_align_mode,
            downsample_factor=1,
        )
        plot_src_mask, plot_tgt_mask, preprocess_meta = preprocess_pair_canonical(
            pair.src, pair.tgt, config, canonical_config
        )
        print(f"    Preprocessed to canonical grid: {plot_src_mask.shape}")
    else:
        # SYNTHETIC: already on canonical grid
        plot_src_mask = src_mask
        plot_tgt_mask = tgt_mask

    # Compute surface metrics for input masks (in canonical units)
    src_metrics = compute_surface_metrics(plot_src_mask, UM_PER_PX)
    tgt_metrics = compute_surface_metrics(plot_tgt_mask, UM_PER_PX)

    # Plot input masks (canonical grid if enabled)
    plot_input_masks_with_metrics(
        plot_src_mask, plot_tgt_mask, src_metrics, tgt_metrics,
        param_dir / "input_masks.png",
        f"{test_name} | ε={epsilon:.0e}, reg_m={marginal_relaxation:.0e}"
    )

    # Run UOT with timing
    try:
        start_time = time.time()
        result = run_uot_pair(pair, config=config)
        elapsed_time = time.time() - start_time
        compute_time_minutes = elapsed_time / 60.0
        print(f"  Computed in {elapsed_time:.2f}s ({compute_time_minutes:.4f} min)")
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "epsilon": epsilon,
            "marginal_relaxation": marginal_relaxation,
            "cost": np.nan,
            "cost_is_nan": True,
            "compute_time_minutes": np.nan,
            "error": str(e),
        }

    # NEW: Validate pair frame results
    if config.use_pair_frame and hasattr(result, 'transform_meta'):
        if result.transform_meta.get("preprocess", {}).get("pair_frame_used"):
            # Verify outputs are canonical-shaped
            assert result.mass_created_hw.shape == CANONICAL_GRID_SHAPE, \
                f"Mass created not canonical shaped: {result.mass_created_hw.shape}"
            assert result.velocity_field_yx_hw2.shape[:2] == CANONICAL_GRID_SHAPE, \
                f"Velocity not canonical shaped: {result.velocity_field_yx_hw2.shape}"

    if getattr(result, "cost_src_px", None) is None:
        raise RuntimeError("UOTResult missing cost_src_px; cannot plot transport cost field.")

    # Collect all diagnostics
    cost_is_nan = np.isnan(result.cost)

    # Compute cost matrix for diagnostics
    backend = POTBackend()
    coords_src = result.support_src_yx.astype(np.float64) * float(COORD_SCALE)
    coords_tgt = result.support_tgt_yx.astype(np.float64) * float(COORD_SCALE)
    cost_matrix = ot.dist(coords_src, coords_tgt, metric="sqeuclidean")

    cost_diag = diagnose_cost_matrix(cost_matrix, epsilon)
    gibbs_diag = diagnose_gibbs_kernel(cost_matrix, epsilon)
    coupling_diag = diagnose_coupling_sparsity(result.coupling) if result.coupling is not None else {}
    velocity_diag = compute_velocity_metrics(result.velocity_px_per_frame_yx)
    mass_diag = compute_mass_metrics(result, src_mask, tgt_mask)

    # Numerical stability check
    numerical_stable = (
        not cost_is_nan
        and not velocity_diag["velocity_has_nan"]
        and gibbs_diag["K_healthy"]
    )

    # Extract percentage metrics
    created_mass_pct = mass_diag.get("created_mass_pct", float("nan"))
    destroyed_mass_pct = mass_diag.get("destroyed_mass_pct", float("nan"))
    proportion_transported = mass_diag.get("proportion_transported", float("nan"))

    # Plot diagnostics
    plot_cost_and_gibbs(cost_matrix, epsilon, cost_diag, gibbs_diag, param_dir / "cost_and_gibbs.png")
    plot_uot_cost_field(result, param_dir / "transport_cost_field.png", canonical_shape=CANONICAL_GRID_SHAPE)
    plot_flow_field(plot_src_mask, result, proportion_transported,
                    param_dir / "flow_field.png")  # Uses viz_config default stride
    plot_uot_quiver(plot_src_mask, result, param_dir / "flow_field_quiver.png", stride=6, canonical_shape=CANONICAL_GRID_SHAPE)
    plot_mask_overlay_only(plot_src_mask, plot_tgt_mask, param_dir / "overlay_masks.png")
    plot_uot_overlay_with_transport(
        plot_src_mask,
        plot_tgt_mask,
        result,
        param_dir / "overlay_transport.png",
        stride=6,
        canonical_shape=CANONICAL_GRID_SHAPE,
    )
    plot_uot_creation_destruction(
        result.mass_created_px, result.mass_destroyed_px,
        created_mass_pct, destroyed_mass_pct,
        param_dir / "creation_destruction.png",
        canonical_shape=CANONICAL_GRID_SHAPE,
    )

    # Compile metrics
    metrics = {
        "epsilon": epsilon,
        "marginal_relaxation": marginal_relaxation,
        "coord_scale": COORD_SCALE,
        "compute_time_minutes": compute_time_minutes,
        "cost": result.cost,
        "cost_is_nan": cost_is_nan,
        "numerical_stable": numerical_stable,
        **cost_diag,
        **gibbs_diag,
        **coupling_diag,
        **velocity_diag,
        **mass_diag,
        "src_area_um2": src_metrics["area_um2"],
        "tgt_area_um2": tgt_metrics["area_um2"],
    }

    return metrics


def run_test_with_grid(
    test_num: int,
    test_name: str,
    test_fn,
    test_params: Dict,
    viable_params: Optional[List[Dict]] = None,
    quick_mode: bool = False,
    output_base: Path = None,
    pair_override: Optional[UOTFramePair] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run a test across parameter grid and return results + viable params."""

    if output_base is None:
        output_base = OUTPUT_DIR

    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {test_name}")
    print('='*60)

    # Add _quick_results suffix if in quick mode
    test_dir_name = f"test{test_num}_{test_name.lower().replace(' ', '_')}"
    if quick_mode:
        test_dir_name += "_quick_results"
    output_dir = output_base / test_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test masks
    src_mask, tgt_mask = test_fn(**test_params)

    # Determine parameter combinations to test
    if viable_params is not None:
        print(f"Testing {len(viable_params)} viable parameter combinations from previous test...")
        param_combos = [(p['epsilon'], p['marginal_relaxation']) for p in viable_params]
    else:
        # Select grid based on quick mode
        epsilon_grid = QUICK_EPSILON_GRID if quick_mode else EPSILON_GRID
        reg_m_grid = QUICK_REG_M_GRID if quick_mode else REG_M_GRID
        mode_label = "quick" if quick_mode else "full"
        print(f"Testing {mode_label} parameter grid ({len(epsilon_grid)} × {len(reg_m_grid)} = {len(epsilon_grid) * len(reg_m_grid)} combinations)...")
        param_combos = [(eps, regm) for eps in epsilon_grid for regm in reg_m_grid]

    # Run all parameter combinations
    all_metrics = []
    for eps, regm in param_combos:
        print(f"  Running ε={eps:.0e}, reg_m={regm:.0e}...", end=" ")
        metrics = run_single_param_combo(
            src_mask, tgt_mask, eps, regm, output_dir, test_name, pair_override=pair_override
        )
        all_metrics.append(metrics)

        # Quick feedback
        if metrics.get("numerical_stable", False):
            print(f"✓ (cost={metrics['cost']:.4f})")
        else:
            print(f"✗ (unstable)")

    # Save results
    df = pd.DataFrame(all_metrics)
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"\nResults saved to {output_dir / 'results.csv'}")

    # Apply pass criteria based on test type
    viable_next = []

    if test_num == 0:  # Real embryo data
        df_pass = df[
            (df['numerical_stable'] == True)
        ]
    elif test_num == 1:  # Identity test
        # Pass criteria: minimal creation/destruction/velocity
        df_pass = df[
            (df['cost'] < 1e-6) &
            (df['created_mass_pct'] < 0.1) &  # Less than 0.1% creation
            (df['destroyed_mass_pct'] < 0.1) &  # Less than 0.1% destruction
            (df['mean_velocity_px'] < 1e-6) &
            (df['K_healthy'] == True)
        ]
    elif test_num == 2:  # Non-overlapping circles
        # Pass criteria: transport happened, no creation/destruction
        df_pass = df[
            (df['cost'] > 0) &
            (df['created_mass_pct'] < 1.0) &  # Less than 1% creation
            (df['destroyed_mass_pct'] < 1.0) &  # Less than 1% destruction
            (df['mean_velocity_px'] > 0) &
            (df['sparsity'] > 0.8)
        ]
    elif test_num == 3:  # Shape change
        # Pass criteria: creation/destruction at expected locations
        df_pass = df[
            (df['created_mass_pct'] > 1.0) &  # At least 1% creation
            (df['destroyed_mass_pct'] > 1.0) &  # At least 1% destruction
            (df['numerical_stable'] == True)
        ]
    else:  # Test 4: Combined
        # Pass criteria: combined behavior
        df_pass = df[
            (df['cost'] > 0) &
            (df['created_mass_pct'] > 1.0) &  # At least 1% creation
            (df['destroyed_mass_pct'] > 1.0) &  # At least 1% destruction
            (df['numerical_stable'] == True)
        ]

    print(f"\n{len(df_pass)} / {len(df)} parameter combinations passed test criteria")

    if len(df_pass) > 0:
        viable_next = df_pass[['epsilon', 'marginal_relaxation']].to_dict('records')

        # Save viable params
        with open(output_dir / "viable_params.json", 'w') as f:
            json.dump(viable_next, f, indent=2)
    else:
        print("WARNING: No parameter combinations passed!")

    # Create sensitivity plots
    if len(df) > 0:
        print("\nGenerating sensitivity plots...")

        # Plot for key metrics
        if test_num == 1:
            plot_sensitivity_heatmap(
                df, 'created_mass', output_dir / "sensitivity_created_mass.png",
                f"Test {test_num}: Created Mass Sensitivity", log_scale=True
            )
        elif test_num == 2:
            if 'sparsity' in df.columns:
                plot_sensitivity_heatmap(
                    df, 'sparsity', output_dir / "sensitivity_sparsity.png",
                    f"Test {test_num}: Coupling Sparsity Sensitivity", log_scale=False
                )

        # Always plot cost
        plot_sensitivity_heatmap(
            df, 'cost', output_dir / "sensitivity_cost.png",
            f"Test {test_num}: Cost Sensitivity", log_scale=True
        )
        
        # Plot compute time heatmap
        if 'compute_time_minutes' in df.columns:
            plot_sensitivity_heatmap(
                df, 'compute_time_minutes', output_dir / "sensitivity_compute_time.png",
                f"Test {test_num}: Compute Time Sensitivity (minutes)", log_scale=False
            )
        
        # Plot average velocity heatmap
        if 'mean_velocity_px' in df.columns:
            plot_sensitivity_heatmap(
                df, 'mean_velocity_px', output_dir / "sensitivity_mean_velocity.png",
                f"Test {test_num}: Mean Velocity Sensitivity (px/frame)", log_scale=False
            )

        # Create parameter comparison grid
        print("\nGenerating parameter comparison grid...")
        # Use percentage-based metrics for display
        metric_to_show = 'created_mass_pct' if test_num in [1, 3, 4] else 'proportion_transported'
        plot_parameter_comparison_grid(output_dir, df, test_name, metric_to_show, quick_mode=quick_mode)

    return df, viable_next


def find_frame_at_stage(
    csv_path: Path,
    embryo_id: str,
    target_hpf: float,
    tolerance_hpf: float,
) -> Tuple[Optional[int], Optional[float]]:
    """Find frame closest to target developmental stage."""
    df = pd.read_csv(
        csv_path,
        usecols=["embryo_id", "frame_index", "predicted_stage_hpf"],
    )
    subset = df[
        (df["embryo_id"] == embryo_id) &
        (df["predicted_stage_hpf"] >= target_hpf - tolerance_hpf) &
        (df["predicted_stage_hpf"] <= target_hpf + tolerance_hpf)
    ]
    if subset.empty:
        return None, None
    subset = subset.copy()
    subset["dist"] = (subset["predicted_stage_hpf"] - target_hpf).abs()
    closest = subset.loc[subset["dist"].idxmin()]
    return int(closest["frame_index"]), float(closest["predicted_stage_hpf"])


# ==== MAIN WORKFLOW ====

def main():
    parser = argparse.ArgumentParser(description="Debug UOT parameters with synthetic tests")
    parser.add_argument(
        '--test',
        type=str,
        default='all',
        help='Which test to run: 1, 2, 3, 4, or "all"'
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=40,
        help='Circle radius in pixels'
    )
    parser.add_argument(
        '--separation',
        type=int,
        default=120,
        help='Separation for non-overlapping test'
    )
    parser.add_argument(
        '--shift',
        type=int,
        default=10,
        help='Shift for combined test'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use reduced parameter grid (4 combinations instead of 20) for faster testing'
    )
    parser.add_argument(
        '--cross-embryo',
        action='store_true',
        help='Use cross-embryo comparison defaults (A05 vs E04 near 48 hpf)'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=None,
        help='CSV path for real embryo masks (mask export CSV)'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='Data root containing segmentation/yolk_v1_0050_predictions (optional)'
    )
    parser.add_argument(
        '--embryo-id',
        type=str,
        default=None,
        help='Embryo ID in CSV (required if --csv is provided)'
    )
    parser.add_argument(
        '--frame-src',
        type=int,
        default=None,
        help='Source frame_index in CSV (required if --csv is provided)'
    )
    parser.add_argument(
        '--frame-tgt',
        type=int,
        default=None,
        help='Target frame_index in CSV (required if --csv is provided)'
    )
    parser.add_argument(
        '--embryo-a',
        type=str,
        default=DEFAULT_EMBRYO_A,
        help=f'Cross-embryo A (default: {DEFAULT_EMBRYO_A})'
    )
    parser.add_argument(
        '--embryo-b',
        type=str,
        default=DEFAULT_EMBRYO_B,
        help=f'Cross-embryo B (default: {DEFAULT_EMBRYO_B})'
    )
    parser.add_argument(
        '--target-hpf',
        type=float,
        default=DEFAULT_TARGET_STAGE_HPF,
        help=f'Target stage (hpf) for cross-embryo selection (default: {DEFAULT_TARGET_STAGE_HPF})'
    )
    parser.add_argument(
        '--stage-tol',
        type=float,
        default=DEFAULT_STAGE_TOLERANCE_HPF,
        help=f'Stage tolerance (hpf) for cross-embryo selection (default: {DEFAULT_STAGE_TOLERANCE_HPF})'
    )
    args = parser.parse_args()

    # Output directory (same base for both modes)
    output_base = OUTPUT_DIR
    output_base.mkdir(parents=True, exist_ok=True)

    # Track viable parameters across tests
    viable_params = None
    all_results = {}

    def _run_cross_embryo() -> None:
        nonlocal viable_params
        csv_path = args.csv if args.csv is not None else DEFAULT_REAL_DATA_CSV
        if not csv_path.exists():
            print(f"\nWARNING: Cross-embryo CSV not found: {csv_path} (skipping)")
            return
        frame_a, stage_a = find_frame_at_stage(
            csv_path, args.embryo_a, args.target_hpf, args.stage_tol
        )
        frame_b, stage_b = find_frame_at_stage(
            csv_path, args.embryo_b, args.target_hpf, args.stage_tol
        )
        if frame_a is None or frame_b is None:
            print(
                f"\nWARNING: Could not find frames near {args.target_hpf} hpf "
                f"for {args.embryo_a} or {args.embryo_b} in {csv_path} (skipping)"
            )
            return
        src_frame = load_mask_from_csv(csv_path, args.embryo_a, frame_a, data_root=args.data_root)
        tgt_frame = load_mask_from_csv(csv_path, args.embryo_b, frame_b, data_root=args.data_root)
        pair = UOTFramePair(
            src=src_frame,
            tgt=tgt_frame,
            pair_meta={
                "comparison_type": "cross_embryo",
                "embryo_a": args.embryo_a,
                "embryo_b": args.embryo_b,
                "frame_a": frame_a,
                "frame_b": frame_b,
                "stage_a_hpf": stage_a,
                "stage_b_hpf": stage_b,
            },
        )

        def _make_real_test() -> Tuple[np.ndarray, np.ndarray]:
            return pair.src.embryo_mask, pair.tgt.embryo_mask

        test_name = (
            f"Cross Embryo {args.embryo_a} f{frame_a} "
            f"vs {args.embryo_b} f{frame_b} (~{args.target_hpf:.1f} hpf)"
        )
        df, viable_params = run_test_with_grid(
            0, test_name, _make_real_test,
            {},
            viable_params=None,
            quick_mode=args.quick,
            output_base=output_base,
            pair_override=pair,
        )
        all_results['cross_embryo'] = df

    if args.cross_embryo:
        _run_cross_embryo()
        tests_to_run = []
    elif args.csv is not None:
        if args.embryo_id is None or args.frame_src is None or args.frame_tgt is None:
            raise ValueError("When using --csv, you must provide --embryo-id, --frame-src, and --frame-tgt.")
        pair = load_mask_pair_from_csv(
            args.csv,
            args.embryo_id,
            args.frame_src,
            args.frame_tgt,
            data_root=args.data_root,
        )

        def _make_real_test() -> Tuple[np.ndarray, np.ndarray]:
            return pair.src.embryo_mask, pair.tgt.embryo_mask

        test_name = f"Real Embryo {args.embryo_id} f{args.frame_src}_f{args.frame_tgt}"
        df, viable_params = run_test_with_grid(
            0, test_name, _make_real_test,
            {},
            viable_params=None,
            quick_mode=args.quick,
            output_base=output_base,
            pair_override=pair,
        )
        all_results['real'] = df
        tests_to_run = []
    else:
        tests_to_run = ['1', '2', '3', '4'] if args.test == 'all' else [args.test]

    # If running all synthetic tests, also run cross-embryo comparison by default.
    if args.test == "all" and not args.cross_embryo and args.csv is None:
        _run_cross_embryo()

    for test_id in tests_to_run:
        if test_id == '1':
            df, viable_params = run_test_with_grid(
                1, "Identity", make_identity_test,
                {"shape": IMAGE_SHAPE, "radius": args.radius},
                viable_params=None,  # Full grid for first test
                quick_mode=args.quick,
                output_base=output_base
            )
            all_results['test1'] = df

        elif test_id == '2':
            if viable_params is None or len(viable_params) == 0:
                print("\nWARNING: No viable params from Test 1; running full grid for Test 2")
                viable_params = None
            df, viable_params = run_test_with_grid(
                2, "Non-overlapping Circles", make_nonoverlap_test,
                {"shape": IMAGE_SHAPE, "radius": args.radius, "separation": args.separation},
                viable_params=viable_params,
                quick_mode=args.quick,
                output_base=output_base
            )
            all_results['test2'] = df

        elif test_id == '3':
            if viable_params is None or len(viable_params) == 0:
                print("\nWARNING: No viable params from Test 2; running full grid for Test 3")
                viable_params = None
            df, viable_params = run_test_with_grid(
                3, "Shape Change", make_shape_change_test,
                {"shape": IMAGE_SHAPE, "radius": args.radius},
                viable_params=viable_params,
                quick_mode=args.quick,
                output_base=output_base
            )
            all_results['test3'] = df

        elif test_id == '4':
            if viable_params is None or len(viable_params) == 0:
                print("\nWARNING: No viable params from Test 3; running full grid for Test 4")
                viable_params = None
            df, viable_params = run_test_with_grid(
                4, "Combined", make_combined_test,
                {"shape": IMAGE_SHAPE, "radius": args.radius, "shift": args.shift},
                viable_params=viable_params,
                quick_mode=args.quick,
                output_base=output_base
            )
            all_results['test4'] = df

    # Final recommendations
    if viable_params is not None and len(viable_params) > 0:
        print(f"\n{'='*60}")
        print("FINAL RECOMMENDATIONS")
        print('='*60)
        print(f"\n{len(viable_params)} parameter combinations passed all tests:")
        for p in viable_params[:10]:  # Show first 10
            print(f"  ε={p['epsilon']:.0e}, reg_m={p['marginal_relaxation']:.0e}")

        if len(viable_params) > 10:
            print(f"  ... and {len(viable_params) - 10} more")

        # Save recommendations
        with open(output_base / "recommended_params.json", 'w') as f:
            json.dump(viable_params, f, indent=2)

        print(f"\nRecommendations saved to {output_base / 'recommended_params.json'}")
    else:
        print("\nWARNING: No parameter combinations passed all tests!")

    print(f"\n{'='*60}")
    print("DIAGNOSTICS COMPLETE")
    print('='*60)
    print(f"Results saved to: {output_base}")


if __name__ == "__main__":
    main()
