"""
Feature Extraction from UOTResult for Phenotype Localization

Extracts canonical-grid features from UOTResult objects for downstream analysis:
- Cost density c(x) - Transport cost per pixel
- Displacement field d(x) - Vector field (u, v) in canonical grid
- Mass delta Δm(x) - Net mass change (creation - destruction)
- Displacement magnitude |d| - Scalar field
- Divergence ∇·d - Scalar field (mass flux)

All features are on the canonical grid (H, W) with real physical units (μm).

Author: Generated for subtle-phenotype localization pilot
Date: 2026-02-13
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[4]
if str(morphseq_root) not in sys.path:
    sys.path.insert(0, str(morphseq_root))

from src.analyze.utils.optimal_transport import UOTResult


@dataclass
class OTFeatures:
    """
    Extracted features from a single UOT comparison.

    All fields are (H, W) arrays on the canonical grid, with NaN outside support.
    Physical units are in micrometers (μm).

    Attributes:
        cost_density: c(x) - Transport cost per pixel (μm²)
        displacement_u: u(x) - Displacement in x direction (μm/frame)
        displacement_v: v(x) - Displacement in y direction (μm/frame)
        displacement_mag: |d(x)| - Magnitude of displacement (μm/frame)
        mass_delta: Δm(x) - Net mass change (creation - destruction) (μm²)
        mass_created: Mass created at each pixel (μm²)
        mass_destroyed: Mass destroyed at each pixel (μm²)
        divergence: ∇·d - Divergence of displacement field (1/frame)
        support_mask: Boolean mask indicating where features are defined
        shape_hw: (H, W) canonical grid shape
        um_per_pixel: Physical resolution (μm/pixel)
        total_cost: Scalar OT cost
    """
    cost_density: np.ndarray
    displacement_u: np.ndarray
    displacement_v: np.ndarray
    displacement_mag: np.ndarray
    mass_delta: np.ndarray
    mass_created: np.ndarray
    mass_destroyed: np.ndarray
    divergence: np.ndarray
    support_mask: np.ndarray
    shape_hw: Tuple[int, int]
    um_per_pixel: float
    total_cost: float


def extract_ot_features(
    result: UOTResult,
    mask_ref: Optional[np.ndarray] = None,
    compute_divergence: bool = True,
) -> OTFeatures:
    """
    Extract all features from UOTResult object.

    Args:
        result: UOTResult from OT computation
        mask_ref: Optional reference mask to restrict support (if None, use cost support)
        compute_divergence: Whether to compute divergence field (can be slow for large grids)

    Returns:
        OTFeatures object with all extracted features

    Raises:
        ValueError: If required fields are missing from UOTResult
    """
    # Extract physical resolution
    if result.pair_frame is None:
        raise ValueError("UOTResult must have pair_frame for physical unit conversion")

    um_per_pixel = result.pair_frame.px_size_um

    # 1. Cost density c(x) - Use source cost (mutant side)
    if result.cost_src_px is None:
        raise ValueError("UOTResult.cost_src_px is None; cost field required")

    cost_density = result.cost_src_px.copy()
    shape_hw = cost_density.shape

    # 2. Displacement field d(x) = (u, v)
    # Note: velocity_px_per_frame_yx is stored as (H, W, 2) with [y, x] order
    displacement_field = result.velocity_px_per_frame_yx * um_per_pixel  # Convert to μm/frame
    displacement_v = displacement_field[..., 0]  # y component
    displacement_u = displacement_field[..., 1]  # x component

    # 3. Mass delta Δm(x) - Use μm² units
    if result.mass_created_um2 is None or result.mass_destroyed_um2 is None:
        raise ValueError("UOTResult must have mass_created_um2 and mass_destroyed_um2")

    mass_created = result.mass_created_um2.copy()
    mass_destroyed = result.mass_destroyed_um2.copy()
    mass_delta = mass_created - mass_destroyed

    # 4. Displacement magnitude |d|
    displacement_mag = np.sqrt(displacement_u**2 + displacement_v**2)

    # 5. Support mask (where features are defined)
    # Use cost density as primary indicator (non-zero = support)
    if mask_ref is not None:
        support_mask = (cost_density > 0) & mask_ref
    else:
        support_mask = cost_density > 0

    # Apply NaN masking outside support (plotting contract)
    cost_density = np.where(support_mask, cost_density, np.nan)
    displacement_u = np.where(support_mask, displacement_u, np.nan)
    displacement_v = np.where(support_mask, displacement_v, np.nan)
    displacement_mag = np.where(support_mask, displacement_mag, np.nan)
    mass_delta = np.where(support_mask, mass_delta, np.nan)
    mass_created = np.where(support_mask, mass_created, np.nan)
    mass_destroyed = np.where(support_mask, mass_destroyed, np.nan)

    # 6. Divergence ∇·d (optional, more expensive)
    if compute_divergence:
        divergence = compute_divergence_field(
            displacement_u, displacement_v, um_per_pixel, support_mask
        )
    else:
        divergence = np.full(shape_hw, np.nan, dtype=np.float32)

    return OTFeatures(
        cost_density=cost_density,
        displacement_u=displacement_u,
        displacement_v=displacement_v,
        displacement_mag=displacement_mag,
        mass_delta=mass_delta,
        mass_created=mass_created,
        mass_destroyed=mass_destroyed,
        divergence=divergence,
        support_mask=support_mask,
        shape_hw=shape_hw,
        um_per_pixel=um_per_pixel,
        total_cost=result.cost,
    )


def compute_divergence_field(
    u: np.ndarray,
    v: np.ndarray,
    um_per_pixel: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute divergence ∇·d = ∂u/∂x + ∂v/∂y using finite differences.

    Divergence represents mass flux: positive = source, negative = sink.

    Args:
        u: (H, W) displacement in x direction (μm/frame)
        v: (H, W) displacement in y direction (μm/frame)
        um_per_pixel: Physical resolution for gradient scaling
        mask: Optional boolean mask (divergence = NaN outside mask)

    Returns:
        divergence: (H, W) divergence field (1/frame, dimensionless)

    Notes:
        - Uses central differences: ∂u/∂x ≈ (u[i, j+1] - u[i, j-1]) / (2 * dx)
        - Boundary pixels use forward/backward differences
        - NaN values are handled by treating as zero for gradient computation
    """
    # Handle NaN values (replace with 0 for gradient computation)
    u_safe = np.where(np.isfinite(u), u, 0.0)
    v_safe = np.where(np.isfinite(v), v, 0.0)

    # Compute gradients using numpy gradient (central differences)
    # Note: gradient returns [∂/∂y, ∂/∂x] for 2D arrays
    du_dx = np.gradient(u_safe, um_per_pixel, axis=1)  # ∂u/∂x
    dv_dy = np.gradient(v_safe, um_per_pixel, axis=0)  # ∂v/∂y

    divergence = du_dx + dv_dy

    # Apply mask
    if mask is not None:
        divergence = np.where(mask, divergence, np.nan)
    else:
        # Use original NaN mask from u or v
        nan_mask = np.isfinite(u) | np.isfinite(v)
        divergence = np.where(nan_mask, divergence, np.nan)

    return divergence.astype(np.float32)


def smooth_feature_for_viz(
    field: np.ndarray,
    mask: np.ndarray,
    sigma_um: float,
    um_per_pixel: float,
) -> np.ndarray:
    """
    Apply boundary-safe Gaussian smoothing for visualization.

    This is the same smoothing used in canonical_grid_viz.py but provided here
    for convenience when smoothing features directly.

    Args:
        field: (H, W) feature field to smooth
        mask: (H, W) boolean mask defining valid region
        sigma_um: Gaussian kernel width in micrometers (e.g., 20.0 μm)
        um_per_pixel: Physical resolution (μm/pixel)

    Returns:
        smoothed_field: (H, W) with NaN outside mask
    """
    sigma_px = sigma_um / um_per_pixel

    # Mask field and convert to 0 outside
    f = np.where(mask, field, 0.0)
    m = mask.astype(float)

    # Gaussian smooth both field and mask
    f_s = gaussian_filter(f, sigma=sigma_px)
    m_s = gaussian_filter(m, sigma=sigma_px)

    # Normalize by smoothed mask to prevent boundary bleeding
    return np.where(m_s > 1e-6, f_s / m_s, np.nan)


def summarize_features(features: OTFeatures) -> dict:
    """
    Compute summary statistics for all features.

    Args:
        features: OTFeatures object

    Returns:
        Dictionary of summary statistics
    """
    support = features.support_mask

    def safe_stats(field: np.ndarray) -> dict:
        """Compute stats on finite values only."""
        vals = field[support & np.isfinite(field)]
        if len(vals) == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
        }

    return {
        "total_cost": features.total_cost,
        "support_pct": 100.0 * support.sum() / support.size,
        "cost_density": safe_stats(features.cost_density),
        "displacement_mag": safe_stats(features.displacement_mag),
        "mass_delta": safe_stats(features.mass_delta),
        "mass_created": safe_stats(features.mass_created),
        "mass_destroyed": safe_stats(features.mass_destroyed),
        "divergence": safe_stats(features.divergence),
    }


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # This is just for testing/documentation
    print("This module extracts features from UOTResult objects.")
    print("See extract_ot_features() function for usage.")
    print("\nExample:")
    print("  from utils.ot_features import extract_ot_features")
    print("  features = extract_ot_features(result)")
    print("  print(summarize_features(features))")
