"""
Canonical Grid Visualization Utilities for Phenotype Localization

BUILDS ON TOP of proven viz.py functions from uot_masks module.
Adds phenotype-localization-specific features:
- S-bin isolines (rostral→caudal "latitude lines")
- Contour-based overlays with real μm² units
- Individual embryo visualization (not WT vs mutant comparisons)

Leverages existing:
- plot_uot_quiver() - Vector fields
- plot_uot_cost_field() - Cost heatmaps
- plot_uot_creation_destruction() - Mass delta maps
- UOTVizConfig - Consistent visualization parameters

Author: Generated for subtle-phenotype localization pilot
Date: 2026-02-13
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[4]
if str(morphseq_root) not in sys.path:
    sys.path.insert(0, str(morphseq_root))

# Import proven OT visualization functions
from src.analyze.optimal_transport_morphometrics.uot_masks import (
    plot_uot_quiver,
    plot_uot_cost_field,
    plot_uot_creation_destruction,
    UOTVizConfig,
    DEFAULT_UOT_VIZ_CONFIG,
)
from src.analyze.utils.optimal_transport import UOTResult


def smooth_inside_mask(field: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing inside mask with boundary safety.

    Prevents bleeding across embryo boundary by normalizing by smoothed mask.

    Args:
        field: (H, W) scalar field to smooth
        mask: (H, W) boolean mask defining embryo region
        sigma: Gaussian kernel width (in pixels or microns depending on grid)

    Returns:
        smoothed_field: (H, W) with NaN outside mask
    """
    f = np.where(mask, field, 0.0)
    m = mask.astype(float)
    f_s = gaussian_filter(f, sigma=sigma)
    m_s = gaussian_filter(m, sigma=sigma)
    return np.where(m_s > 1e-6, f_s / m_s, np.nan)


def quantile_levels(field: np.ndarray, mask: np.ndarray,
                   qs: Tuple[float, ...] = (0.5, 0.7, 0.85, 0.92, 0.97)) -> np.ndarray:
    """
    Compute contour levels from quantiles of field inside mask.

    This yields consistent "top X%" bands across embryos/conditions.

    Args:
        field: (H, W) scalar field
        mask: (H, W) boolean mask
        qs: Tuple of quantiles (0 to 1)

    Returns:
        levels: Array of contour level values
    """
    vals = field[mask & np.isfinite(field)]
    if len(vals) == 0:
        raise ValueError("No valid values inside mask")
    return np.quantile(vals, qs)


def symmetric_levels(field: np.ndarray, mask: np.ndarray,
                    n_levels: int = 7) -> np.ndarray:
    """
    Compute symmetric contour levels around 0 for difference maps.

    Use for mutant-minus-WT difference maps to ensure visual fairness.

    Args:
        field: (H, W) difference field (can be positive or negative)
        mask: (H, W) boolean mask
        n_levels: Number of levels (must be odd for symmetry around 0)

    Returns:
        levels: Array symmetric around 0
    """
    vals = field[mask & np.isfinite(field)]
    if len(vals) == 0:
        raise ValueError("No valid values inside mask")
    max_abs = np.max(np.abs(vals))
    return np.linspace(-max_abs, max_abs, n_levels)


def plot_canonical_overlay(
    c: np.ndarray,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    mask_ref: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    sigma: float = 2.0,
    levels: Optional[np.ndarray] = None,
    use_quantiles: bool = True,
    quantiles: Tuple[float, ...] = (0.5, 0.7, 0.85, 0.92, 0.97),
    vector_stride: int = 12,
    vector_min_mag: float = 0.0,
    s_bin_edges: Optional[np.ndarray] = None,
    title: str = "",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    um_per_pixel: float = 10.0,
) -> plt.Axes:
    """
    Plot canonical grid overlay with filled contours + vectors + S-bins.

    The "clean" visualization approach:
    1. Mask to embryo region (NaN outside)
    2. Boundary-safe Gaussian smoothing
    3. Filled contours (quantile or fixed levels)
    4. Thin contour lines for crisp boundaries
    5. Embryo outline
    6. Subsampled vector field
    7. S-bin isolines (if provided)

    Args:
        c: (H, W) cost density or scalar field to visualize
        u, v: (H, W) displacement vector field components (optional)
        mask_ref: (H, W) boolean mask for embryo region (if None, use all non-NaN in c)
        S: (H, W) spline coordinate map in [0, 1] (optional)
        sigma: Gaussian smoothing width (pixels or microns)
        levels: Custom contour levels (if None, computed from quantiles or field range)
        use_quantiles: If True, use quantile-based levels
        quantiles: Quantiles for level computation if use_quantiles=True
        vector_stride: Subsample vectors every N pixels
        vector_min_mag: Only show vectors with magnitude > this threshold
        s_bin_edges: S values for bin boundaries (e.g., [0.1, 0.2, ..., 0.9])
        title: Plot title
        cmap: Colormap for filled contours
        ax: Matplotlib axes (if None, create new figure)
        show_colorbar: Whether to add colorbar
        um_per_pixel: Microns per pixel (for scale bar, axis labels)

    Returns:
        ax: Matplotlib axes object
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Default mask: all non-NaN pixels
    if mask_ref is None:
        mask_ref = np.isfinite(c)

    # Step 1: Mask to embryo region
    c_masked = np.where(mask_ref, c, np.nan)

    # Step 2: Smooth for clean regions
    c_smooth = smooth_inside_mask(c, mask_ref, sigma=sigma)

    # Step 3: Choose contour levels
    if levels is None:
        if use_quantiles:
            levels = quantile_levels(c_smooth, mask_ref, qs=quantiles)
        else:
            # Fixed levels spanning data range
            vmin, vmax = np.nanpercentile(c_smooth[mask_ref], [1, 99])
            levels = np.linspace(vmin, vmax, 7)

    # Step 4: Filled contours (clean zones)
    cf = ax.contourf(c_smooth, levels=levels, cmap=cmap, alpha=1.0, extend='both')

    # Step 5: Thin contour lines for crisp boundaries
    ax.contour(c_smooth, levels=levels, colors='black', linewidths=0.5, alpha=0.3)

    # Step 6: Embryo outline (reference mask boundary)
    ax.contour(mask_ref.astype(float), levels=[0.5], colors='white',
               linewidths=1.5, linestyles='solid')

    # Step 7: Vectors (subsample aggressively)
    if u is not None and v is not None:
        H, W = c.shape
        Y = np.arange(H)[::vector_stride]
        X = np.arange(W)[::vector_stride]
        XX, YY = np.meshgrid(X, Y)

        # Only plot inside mask
        inside = mask_ref[YY, XX]

        # Filter by magnitude
        u_sub = u[YY, XX]
        v_sub = v[YY, XX]
        mag = np.sqrt(u_sub**2 + v_sub**2)
        valid = inside & (mag > vector_min_mag)

        if valid.any():
            ax.quiver(
                XX[valid], YY[valid],
                u_sub[valid], v_sub[valid],
                angles='xy', scale_units='xy', scale=1.0,
                width=0.003, headwidth=3, headlength=4,
                color='white', alpha=0.7, edgecolors='black', linewidths=0.5
            )

    # Step 8: S-bin boundaries (rostral→caudal section lines)
    if S is not None and s_bin_edges is not None:
        ax.contour(S, levels=s_bin_edges, colors='cyan',
                   linewidths=0.8, linestyles='dashed', alpha=0.6)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel(f'X (pixels, {um_per_pixel} μm/px)')
    ax.set_ylabel(f'Y (pixels, {um_per_pixel} μm/px)')

    if title:
        full_title = f"{title}\n(Gaussian σ={sigma} px; filled contours)"
    else:
        full_title = f"Gaussian-smoothed field (σ={sigma} px)"
    ax.set_title(full_title, fontsize=10)

    # Colorbar
    if show_colorbar:
        plt.colorbar(cf, ax=ax, label='Cost density')

    return ax


def plot_difference_map(
    c_mutant: np.ndarray,
    c_wt: np.ndarray,
    mask_ref: np.ndarray,
    sigma: float = 2.0,
    n_levels: int = 7,
    s_bin_edges: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    title: str = "Mutant - WT difference",
    ax: Optional[plt.Axes] = None,
    um_per_pixel: float = 10.0,
) -> plt.Axes:
    """
    Plot difference map (mutant - WT) with symmetric levels around 0.

    Args:
        c_mutant: (H, W) mutant cost density
        c_wt: (H, W) WT cost density
        mask_ref: (H, W) boolean mask
        sigma: Gaussian smoothing width
        n_levels: Number of symmetric levels (must be odd)
        s_bin_edges: S bin boundaries (optional)
        S: (H, W) spline coordinate map (optional)
        title: Plot title
        ax: Matplotlib axes (if None, create new)
        um_per_pixel: Microns per pixel

    Returns:
        ax: Matplotlib axes object
    """
    # Compute difference
    c_diff = c_mutant - c_wt

    # Smooth
    c_diff_smooth = smooth_inside_mask(c_diff, mask_ref, sigma=sigma)

    # Symmetric levels
    levels = symmetric_levels(c_diff_smooth, mask_ref, n_levels=n_levels)

    # Plot with diverging colormap
    return plot_canonical_overlay(
        c_diff_smooth,
        mask_ref=mask_ref,
        S=S,
        sigma=0.0,  # Already smoothed
        levels=levels,
        use_quantiles=False,
        s_bin_edges=s_bin_edges,
        title=title,
        cmap='RdBu_r',  # Diverging: red = higher in mutant, blue = higher in WT
        ax=ax,
        um_per_pixel=um_per_pixel,
    )


def plot_mean_field_comparison(
    fields_wt: List[np.ndarray],
    fields_mutant: List[np.ndarray],
    mask_ref: np.ndarray,
    sigma: float = 2.0,
    field_name: str = "cost density",
    s_bin_edges: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    um_per_pixel: float = 10.0,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """
    3-panel comparison: WT mean, mutant mean, difference.

    IMPORTANT: When aggregating across embryos, compute mean THEN smooth:
    - mean_field = np.mean(fields, axis=0)
    - smooth(mean_field)

    This is different from smooth-then-mean, which would over-blur.

    Args:
        fields_wt: List of (H, W) WT fields
        fields_mutant: List of (H, W) mutant fields
        mask_ref: (H, W) boolean mask
        sigma: Gaussian smoothing width
        field_name: Name of field for titles
        s_bin_edges: S bin boundaries (optional)
        S: (H, W) spline coordinate map (optional)
        um_per_pixel: Microns per pixel
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    # Compute means (aggregate BEFORE smoothing)
    mean_wt = np.mean(fields_wt, axis=0)
    mean_mutant = np.mean(fields_mutant, axis=0)

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: WT mean
    plot_canonical_overlay(
        mean_wt,
        mask_ref=mask_ref,
        S=S,
        sigma=sigma,
        s_bin_edges=s_bin_edges,
        title=f"WT mean {field_name}",
        ax=axes[0],
        um_per_pixel=um_per_pixel,
    )

    # Panel B: Mutant mean
    plot_canonical_overlay(
        mean_mutant,
        mask_ref=mask_ref,
        S=S,
        sigma=sigma,
        s_bin_edges=s_bin_edges,
        title=f"Mutant mean {field_name}",
        ax=axes[1],
        um_per_pixel=um_per_pixel,
    )

    # Panel C: Difference (mutant - WT)
    plot_difference_map(
        mean_mutant,
        mean_wt,
        mask_ref=mask_ref,
        sigma=sigma,
        s_bin_edges=s_bin_edges,
        S=S,
        title=f"Difference (mutant - WT)",
        ax=axes[2],
        um_per_pixel=um_per_pixel,
    )

    fig.tight_layout()
    return fig


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Synthetic example
    H, W = 256, 576

    # Create synthetic mask (ellipse)
    yy, xx = np.ogrid[:H, :W]
    mask = ((yy - H//2)**2 / (H//4)**2 + (xx - W//2)**2 / (W//3)**2) < 1

    # Create synthetic cost field
    c = np.random.randn(H, W) * 0.5 + 2.0
    c[~mask] = np.nan

    # Create synthetic vector field
    u = np.sin(yy / 20) * 5
    v = np.cos(xx / 30) * 5

    # Create synthetic S coordinate (head=0, tail=1)
    S = (xx - xx.min()) / (xx.max() - xx.min())
    S[~mask] = np.nan

    # Plot single embryo
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_canonical_overlay(
        c, u, v,
        mask_ref=mask,
        S=S,
        sigma=2.0,
        s_bin_edges=np.linspace(0, 1, 11)[1:-1],  # 10 bins
        title="Example: Cost density with vectors and S-bins",
    )
    plt.show()

    # Plot mean comparison
    fields_wt = [c + np.random.randn(H, W) * 0.2 for _ in range(5)]
    fields_mutant = [c + 1.0 + np.random.randn(H, W) * 0.2 for _ in range(5)]

    fig = plot_mean_field_comparison(
        fields_wt,
        fields_mutant,
        mask_ref=mask,
        sigma=2.0,
        field_name="cost density",
        s_bin_edges=np.linspace(0, 1, 11)[1:-1],
        S=S,
    )
    plt.show()
