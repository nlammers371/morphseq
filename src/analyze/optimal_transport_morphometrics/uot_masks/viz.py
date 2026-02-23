"""Visualization helpers for UOT mask transport."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from analyze.utils.optimal_transport import Coupling, UOTResultCanonical


# ==============================================================================
# Phase 2: Proven Plotting Functions (Migrated from debug_uot_params.py)
# ==============================================================================
# These functions enforce the UOT Plotting Contract:
# - NaN masking for non-support regions
# - Explicit support masks
# - Statistics on support points only
# - No fabrication via smoothing


@dataclass
class UOTVizConfig:
    """Visualization configuration for cross-run comparison.

    Fixed scales ensure consistent interpretation across parameter sweeps.
    """
    mass_pct_vmin: float = 0.0
    mass_pct_vmax: float = 10.0
    velocity_vmin: float = 0.0
    velocity_vmax: float = 50.0  # μm/frame
    min_velocity_px: float = 1.0
    min_velocity_pct: float = 0.02
    quiver_base_scale: float = 150.0
    quiver_stride: int = 4


DEFAULT_UOT_VIZ_CONFIG = UOTVizConfig()


# Display mode helpers (for image vs cartesian coordinate systems)
def _get_display_mode() -> str:
    """Get display mode from MORPHSEQ_DISPLAY_MODE env var (default: image)."""
    return os.environ.get("MORPHSEQ_DISPLAY_MODE", "image").lower()


def _plot_extent(hw: Tuple[int, int]) -> Tuple[list, str]:
    """Return extent and origin for current display mode."""
    h, w = hw
    display_mode = _get_display_mode()
    if display_mode == "cartesian":
        return [0, w, 0, h], "lower"
    return [0, w, h, 0], "upper"


def _set_axes_limits(ax, hw: Tuple[int, int]) -> None:
    """Set axis limits respecting display mode."""
    h, w = hw
    display_mode = _get_display_mode()
    ax.set_xlim(0, w)
    if display_mode == "cartesian":
        ax.set_ylim(0, h)
    else:
        ax.set_ylim(h, 0)


def _quiver_transform(
    xx: np.ndarray, yy: np.ndarray, u: np.ndarray, v: np.ndarray, h: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform quiver coordinates for display mode."""
    display_mode = _get_display_mode()
    if display_mode == "cartesian":
        return xx, (h - yy), u, -v
    return xx, yy, u, v


def _overlay_masks_rgb(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    src_color: Tuple[float, float, float] = (1.0, 0.80, 0.80),  # light red
    tgt_color: Tuple[float, float, float] = (0.80, 0.85, 1.0),  # light blue
    alpha: float = 0.8,
) -> np.ndarray:
    """Create RGB overlay of source/target masks."""
    h, w = src_mask.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    src = src_mask.astype(bool)
    tgt = tgt_mask.astype(bool)
    for channel in range(3):
        rgb[..., channel] = np.where(
            src,
            rgb[..., channel] * (1 - alpha) + src_color[channel] * alpha,
            rgb[..., channel],
        )
        rgb[..., channel] = np.where(
            tgt,
            rgb[..., channel] * (1 - alpha) + tgt_color[channel] * alpha,
            rgb[..., channel],
        )
    return rgb


def plot_uot_quiver(
    src_mask: np.ndarray,
    result: UOTResultCanonical,
    output_path: Path,
    stride: int = 6,
    canonical_shape: Optional[Tuple[int, int]] = None,
    viz_config: Optional[UOTVizConfig] = None,
) -> None:
    """Plot velocity field as quiver (arrows) on support points only.

    Migrated from debug_uot_params.py::plot_flow_field_quiver()

    PLOTTING CONTRACT ENFORCED:
    - Shows only support points (no fabrication)
    - Overlays on source mask for context
    - Uses fixed stride for arrow density

    Args:
        src_mask: Source mask for background context
        result: UOTResult containing velocity field
        output_path: Where to save the plot
        stride: Subsample stride for arrow density (higher = fewer arrows)
        canonical_shape: Optional (H, W) for axis limits (defaults to src_mask.shape)
        viz_config: Visualization configuration (defaults to DEFAULT_UOT_VIZ_CONFIG)
    """
    if viz_config is None:
        viz_config = DEFAULT_UOT_VIZ_CONFIG

    if canonical_shape is None:
        canonical_shape = src_mask.shape

    velocity_px = np.asarray(result.velocity_canon_px_per_step_yx, dtype=np.float32)
    scale_um = float(getattr(result, "canonical_um_per_px", float("nan")))
    if np.isfinite(scale_um):
        velocity_field = velocity_px * scale_um
        unit_label = "μm/step"
    else:
        velocity_field = velocity_px
        unit_label = "px/step"

    velocity_mag = np.sqrt(velocity_field[..., 0] ** 2 + velocity_field[..., 1] ** 2)

    # Support mask
    support_mask = velocity_mag > 0
    support_pct = 100.0 * support_mask.sum() / support_mask.size

    canon_h, canon_w = canonical_shape
    h_vel, w_vel = velocity_field.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Background: source mask
    extent, origin = _plot_extent(src_mask.shape)
    ax.imshow(
        src_mask,
        cmap="gray",
        alpha=0.3,
        aspect="equal",
        extent=extent,
        origin=origin,
        interpolation="nearest",
    )

    # Subsample for quiver (on support points only)
    yy_vel, xx_vel = np.meshgrid(
        np.arange(0, h_vel, stride), np.arange(0, w_vel, stride), indexing="ij"
    )
    u = velocity_field[::stride, ::stride, 1]  # x component
    v = velocity_field[::stride, ::stride, 0]  # y component
    mag_sub = velocity_mag[::stride, ::stride]
    support_sub = support_mask[::stride, ::stride]

    # Only show arrows on support points
    mask_sub = support_sub & (mag_sub > 0)
    n_arrows = mask_sub.sum()

    if n_arrows > 0:
        xx_plot, yy_plot, u_plot, v_plot = _quiver_transform(
            xx_vel, yy_vel, u, v, h_vel
        )
        ax.quiver(
            xx_plot[mask_sub],
            yy_plot[mask_sub],
            u_plot[mask_sub],
            v_plot[mask_sub],
            mag_sub[mask_sub],
            cmap="hot",
            scale=viz_config.quiver_base_scale,
            scale_units="xy",
            angles="xy",
        )
        title_str = f"Velocity Field (Quiver, stride={stride})\nSupport: {support_pct:.2f}%, Arrows: {n_arrows}"
    else:
        ax.text(
            0.5,
            0.5,
            "No significant flow\n(Identity or near-identity case)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            style="italic",
        )
        title_str = (
            f"Velocity Field (Quiver, stride={stride})\nNo arrows (identity case)"
        )

    ax.set_title(title_str)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    _set_axes_limits(ax, (canon_h, canon_w))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_uot_cost_field(
    result: UOTResultCanonical,
    output_path: Path,
    canonical_shape: Optional[Tuple[int, int]] = None,
    viz_config: Optional[UOTVizConfig] = None,
) -> None:
    """Plot transport cost field analysis (3-panel).

    Migrated from debug_uot_params.py::plot_transport_cost_field()

    Shows how expensive it is to transport mass from each src_support location.
    Cost per source point = sum over targets of (coupling[i,j] * cost_matrix[i,j])

    PLOTTING CONTRACT ENFORCED:
    - Only shows src_support points (NaN elsewhere)
    - Statistics on support only

    Args:
        result: UOTResultCanonical containing cost field
        output_path: Where to save the plot
        canonical_shape: Optional (H, W) for axis limits (inferred from cost field if None)
        viz_config: Visualization configuration (currently unused, for future consistency)
    """
    cost_field_canonical = getattr(result, "cost_src_canon", None)
    if cost_field_canonical is None:
        print("Warning: No cost_src_canon on result; cost field plot skipped")
        return

    if canonical_shape is None:
        canonical_shape = cost_field_canonical.shape

    canon_h, canon_w = canonical_shape
    if cost_field_canonical.shape != (canon_h, canon_w):
        print(
            f"Warning: Cost field has shape {cost_field_canonical.shape}, expected {canonical_shape}"
        )
        return

    # Apply NaN masking (plotting contract)
    support_mask = cost_field_canonical > 0
    cost_field_masked = cost_field_canonical.copy()
    cost_field_masked[~support_mask] = np.nan

    support_pct = 100.0 * support_mask.sum() / support_mask.size

    # Statistics on support only
    if support_mask.any():
        support_costs = cost_field_canonical[support_mask]
        p50 = np.percentile(support_costs, 50)
        p90 = np.percentile(support_costs, 90)
        p99 = np.percentile(support_costs, 99)
        cost_max = support_costs.max()
    else:
        p50 = p90 = p99 = cost_max = 0.0

    # Create 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Support mask
    extent, origin = _plot_extent((canon_h, canon_w))
    axes[0].imshow(
        support_mask,
        cmap="gray",
        aspect="equal",
        extent=extent,
        origin=origin,
        interpolation="nearest",
    )
    axes[0].set_title(
        f"Transport Cost Support (src_support)\n{support_pct:.2f}% defined"
    )
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")

    # Panel 2: Cost field (NaN-masked)
    im = axes[1].imshow(
        cost_field_masked,
        cmap="hot",
        aspect="equal",
        extent=extent,
        origin=origin,
        interpolation="nearest",
    )
    axes[1].set_title(
        f"Transport Cost per Source Point (src_support only)\n"
        f"p50/p90/p99: {p50:.1e}/{p90:.1e}/{p99:.1e}"
    )
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("y (px)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Cost (squared μm)")

    # Panel 3: Cost histogram
    if support_mask.any():
        axes[2].hist(
            support_costs, bins=50, color="red", alpha=0.7, edgecolor="black"
        )
        axes[2].set_xlabel("Transport Cost (squared μm)")
        axes[2].set_ylabel("Count")
        axes[2].set_title(
            f"Cost Distribution (src_support only)\nmax: {cost_max:.1e}"
        )
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No support points",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
        )
        axes[2].set_xlabel("Transport Cost (squared μm)")
        axes[2].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_uot_creation_destruction(
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    created_mass_pct: float,
    destroyed_mass_pct: float,
    output_path: Path,
    canonical_shape: Optional[Tuple[int, int]] = None,
    viz_config: Optional[UOTVizConfig] = None,
) -> None:
    """Plot creation/destruction analysis (4-panel with contract enforcement).

    Migrated from debug_uot_params.py::plot_creation_destruction_maps()
    Replaces older plot_creation_destruction() with full contract compliance.

    PLOTTING CONTRACT ENFORCED:
    - Uses NaN for non-support regions
    - Shows support mask explicitly
    - Displays statistics on support only

    CRITICAL: Data should already be on canonical grid.
    Display as-is with canonical grid axis limits (no stretching).

    Args:
        mass_created_hw: (H, W) mass creation map (percentage)
        mass_destroyed_hw: (H, W) mass destruction map (percentage)
        created_mass_pct: Total created mass percentage
        destroyed_mass_pct: Total destroyed mass percentage
        output_path: Where to save the plot
        canonical_shape: Optional (H, W) for axis limits (defaults to mass array shape)
        viz_config: Visualization configuration (for fixed vmin/vmax scales)
    """
    if viz_config is None:
        viz_config = DEFAULT_UOT_VIZ_CONFIG

    if canonical_shape is None:
        canonical_shape = mass_created_hw.shape

    # Create support masks (non-zero mass)
    created_mask = mass_created_hw > 0
    destroyed_mask = mass_destroyed_hw > 0

    # PLOTTING CONTRACT: Replace zeros with NaN outside support
    created_masked = mass_created_hw.copy()
    destroyed_masked = mass_destroyed_hw.copy()
    created_masked[~created_mask] = np.nan
    destroyed_masked[~destroyed_mask] = np.nan

    # Statistics on support only
    if created_mask.any():
        created_vals = mass_created_hw[created_mask]
        created_p50 = np.percentile(created_vals, 50)
        created_p90 = np.percentile(created_vals, 90)
    else:
        created_p50 = created_p90 = 0.0

    if destroyed_mask.any():
        destroyed_vals = mass_destroyed_hw[destroyed_mask]
        destroyed_p50 = np.percentile(destroyed_vals, 50)
        destroyed_p90 = np.percentile(destroyed_vals, 90)
    else:
        destroyed_p50 = destroyed_p90 = 0.0

    # Use canonical grid dimensions for axis limits
    canon_h, canon_w = canonical_shape
    h, w = mass_created_hw.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Row 1: Support masks
    extent, origin = _plot_extent((h, w))
    axes[0, 0].imshow(
        created_mask,
        cmap="gray",
        aspect="equal",
        extent=extent,
        origin=origin,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axes[0, 0].set_title(
        f"Creation Support (tgt_support)\n{100*created_mask.sum()/created_mask.size:.2f}% defined"
    )
    axes[0, 0].set_xlabel("x (px)")
    axes[0, 0].set_ylabel("y (px)")
    _set_axes_limits(axes[0, 0], (canon_h, canon_w))

    axes[0, 1].imshow(
        destroyed_mask,
        cmap="gray",
        aspect="equal",
        extent=extent,
        origin=origin,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title(
        f"Destruction Support (src_support)\n{100*destroyed_mask.sum()/destroyed_mask.size:.2f}% defined"
    )
    axes[0, 1].set_xlabel("x (px)")
    axes[0, 1].set_ylabel("y (px)")
    _set_axes_limits(axes[0, 1], (canon_h, canon_w))

    # Row 2: Mass heatmaps (NaN outside support)
    # Use fixed vmin/vmax for consistent cross-run comparison
    im0 = axes[1, 0].imshow(
        created_masked,
        cmap="Reds",
        aspect="equal",
        extent=extent,
        origin=origin,
        interpolation="nearest",
        vmin=viz_config.mass_pct_vmin,
        vmax=viz_config.mass_pct_vmax,
    )
    axes[1, 0].set_title(
        f"Mass Created (tgt_support only)\n"
        f"{created_mass_pct:.2f}% of total target (from tgt_support sampling ONLY!) | p50/p90: {created_p50:.2f}/{created_p90:.2f}%"
    )
    axes[1, 0].set_xlabel("x (px)")
    axes[1, 0].set_ylabel("y (px)")
    _set_axes_limits(axes[1, 0], (canon_h, canon_w))
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04, label="%")

    im1 = axes[1, 1].imshow(
        destroyed_masked,
        cmap="Blues",
        aspect="equal",
        extent=extent,
        origin=origin,
        interpolation="nearest",
        vmin=viz_config.mass_pct_vmin,
        vmax=viz_config.mass_pct_vmax,
    )
    axes[1, 1].set_title(
        f"Mass Destroyed (src_support only)\n"
        f"{destroyed_mass_pct:.2f}% of total source (from src_support sampling ONLY!) | p50/p90: {destroyed_p50:.2f}/{destroyed_p90:.2f}%"
    )
    axes[1, 1].set_xlabel("x (px)")
    axes[1, 1].set_ylabel("y (px)")
    _set_axes_limits(axes[1, 1], (canon_h, canon_w))
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04, label="%")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_uot_overlay_with_transport(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    result: UOTResultCanonical,
    output_path: Path,
    stride: int = 6,
    canonical_shape: Optional[Tuple[int, int]] = None,
    viz_config: Optional[UOTVizConfig] = None,
) -> None:
    """Plot multi-layer visualization: mask overlay + cost field + quiver.

    Migrated from debug_uot_params.py::plot_overlay_transport_field()

    Overlays src/tgt masks with cost-colored support and transport arrows.

    Args:
        src_mask: Source mask
        tgt_mask: Target mask
        result: UOTResultCanonical containing transport data
        output_path: Where to save the plot
        stride: Subsample stride for arrow density
        canonical_shape: Optional (H, W) for axis limits (defaults to src_mask.shape)
        viz_config: Visualization configuration
    """
    if viz_config is None:
        viz_config = DEFAULT_UOT_VIZ_CONFIG

    if canonical_shape is None:
        canonical_shape = src_mask.shape

    velocity_field = np.asarray(result.velocity_canon_px_per_step_yx, dtype=np.float32)
    velocity_mag = np.sqrt(velocity_field[..., 0] ** 2 + velocity_field[..., 1] ** 2)
    support_mask = velocity_mag > 0

    cost_field = getattr(result, "cost_src_canon", None)
    if cost_field is None:
        cost_field = np.zeros_like(src_mask, dtype=np.float32)

    # Mask cost field to support
    cost_masked = cost_field.copy()
    cost_masked[~support_mask] = np.nan

    h, w = src_mask.shape
    canon_h, canon_w = canonical_shape

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Base overlay of src/tgt
    overlay = _overlay_masks_rgb(src_mask, tgt_mask)
    extent, origin = _plot_extent((h, w))
    ax.imshow(overlay, extent=extent, origin=origin, interpolation="nearest")

    # Cost field overlay
    im = ax.imshow(
        cost_masked,
        cmap="Reds",
        alpha=0.6,
        extent=extent,
        origin=origin,
        interpolation="nearest",
    )
    plt.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label="Transport cost (src_support)"
    )

    # Quiver overlay
    yy, xx = np.meshgrid(
        np.arange(0, h, stride), np.arange(0, w, stride), indexing="ij"
    )
    u = velocity_field[::stride, ::stride, 1]
    v = velocity_field[::stride, ::stride, 0]
    mag_sub = velocity_mag[::stride, ::stride]
    support_sub = support_mask[::stride, ::stride]
    mask_sub = support_sub & (mag_sub > 0)

    if mask_sub.any():
        xx_plot, yy_plot, u_plot, v_plot = _quiver_transform(xx, yy, u, v, h)
        ax.quiver(
            xx_plot[mask_sub],
            yy_plot[mask_sub],
            u_plot[mask_sub],
            v_plot[mask_sub],
            mag_sub[mask_sub],
            cmap="viridis",
            scale=viz_config.quiver_base_scale,
            scale_units="xy",
            angles="xy",
            width=0.002,
        )

    _set_axes_limits(ax, (canon_h, canon_w))
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Source/Target Overlay + Transport Field (Cost Colored)")

    legend_handles = [
        Patch(facecolor=(1.0, 0.80, 0.80), edgecolor="none", label="Source"),
        Patch(facecolor=(0.80, 0.85, 1.0), edgecolor="none", label="Target"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_uot_diagnostic_suite(
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    result: UOTResultCanonical,
    output_dir: Path,
    prefix: str = "",
    canonical_shape: Optional[Tuple[int, int]] = None,
    viz_config: Optional[UOTVizConfig] = None,
) -> Dict[str, Path]:
    """Generate full diagnostic suite for UOT result.

    Creates 4 plots:
    - {prefix}quiver.png
    - {prefix}cost_field.png
    - {prefix}creation_destruction.png
    - {prefix}overlay_transport.png

    Args:
        src_mask: Source mask
        tgt_mask: Target mask
        result: UOTResultCanonical containing transport data
        output_dir: Directory to save plots
        prefix: Optional prefix for output filenames
        canonical_shape: Optional (H, W) for axis limits
        viz_config: Visualization configuration

    Returns:
        Dict mapping plot type to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    outputs["quiver"] = output_dir / f"{prefix}quiver.png"
    plot_uot_quiver(
        src_mask,
        result,
        outputs["quiver"],
        canonical_shape=canonical_shape,
        viz_config=viz_config,
    )

    outputs["cost_field"] = output_dir / f"{prefix}cost_field.png"
    plot_uot_cost_field(
        result, outputs["cost_field"], canonical_shape=canonical_shape, viz_config=viz_config
    )

    # Extract canonical mass fields from result
    mass_created = result.mass_created_canon
    mass_destroyed = result.mass_destroyed_canon
    created_pct = result.diagnostics.get("metrics", {}).get("created_mass_pct", 0.0)
    destroyed_pct = result.diagnostics.get("metrics", {}).get("destroyed_mass_pct", 0.0)

    outputs["creation_destruction"] = output_dir / f"{prefix}creation_destruction.png"
    plot_uot_creation_destruction(
        mass_created,
        mass_destroyed,
        created_pct,
        destroyed_pct,
        outputs["creation_destruction"],
        canonical_shape=canonical_shape,
        viz_config=viz_config,
    )

    outputs["overlay_transport"] = output_dir / f"{prefix}overlay_transport.png"
    plot_uot_overlay_with_transport(
        src_mask,
        tgt_mask,
        result,
        outputs["overlay_transport"],
        canonical_shape=canonical_shape,
        viz_config=viz_config,
    )

    return outputs


# ==============================================================================
# Legacy Functions (Phase 1)
# ==============================================================================


def plot_creation_destruction(
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """DEPRECATED: Use plot_uot_creation_destruction() instead.

    This function does not enforce the UOT plotting contract (NaN masking,
    support-only statistics). The new function provides full contract compliance.
    """
    mass_created_hw = np.maximum(mass_created_hw, 0.0)
    mass_destroyed_hw = np.maximum(mass_destroyed_hw, 0.0)
    vmax_created = float(mass_created_hw.max()) if mass_created_hw.size else 0.0
    vmax_destroyed = float(mass_destroyed_hw.max()) if mass_destroyed_hw.size else 0.0
    vmax_created = max(vmax_created, 1e-12)
    vmax_destroyed = max(vmax_destroyed, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(mass_created_hw, cmap="magma", vmin=0.0, vmax=vmax_created)
    axes[0].set_title("Mass created")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mass_destroyed_hw, cmap="magma", vmin=0.0, vmax=vmax_destroyed)
    axes[1].set_title("Mass destroyed")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def _downsample_mask_to_shape(mask_hw: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask_hw.shape == target_shape:
        return mask_hw
    h, w = mask_hw.shape
    th, tw = target_shape
    if th == 0 or tw == 0:
        raise ValueError("Target shape must be non-zero.")
    if h % th == 0 and w % tw == 0:
        fh = h // th
        fw = w // tw
        trimmed = mask_hw[: th * fh, : tw * fw]
        reshaped = trimmed.reshape(th, fh, tw, fw)
        return (reshaped.max(axis=(1, 3)) > 0).astype(mask_hw.dtype)

    zoom_factors = (th / float(h), tw / float(w))
    resized = ndimage.zoom(mask_hw.astype(np.float32), zoom=zoom_factors, order=0)
    if resized.shape != target_shape:
        resized = resized[:th, :tw]
        if resized.shape != target_shape:
            pad_h = th - resized.shape[0]
            pad_w = tw - resized.shape[1]
            resized = np.pad(resized, ((0, pad_h), (0, pad_w)), mode="constant")
    return (resized > 0.5).astype(mask_hw.dtype)


def _expand_mass_map_to_full(
    mass_hw: np.ndarray,
    transform_meta: Optional[dict],
) -> Optional[np.ndarray]:
    if not transform_meta:
        return None
    preprocess = transform_meta.get("preprocess", {})
    orig_shape = preprocess.get("orig_shape")
    bbox = preprocess.get("bbox_y0y1x0x1")
    pad_hw = preprocess.get("pad_hw", (0, 0))
    downsample_factor = int(transform_meta.get("downsample_factor", 1))

    if orig_shape is None or bbox is None:
        return None

    mass_up = mass_hw
    if downsample_factor > 1:
        mass_up = np.repeat(mass_up, downsample_factor, axis=0)
        mass_up = np.repeat(mass_up, downsample_factor, axis=1)

    pad_h, pad_w = pad_hw
    if pad_h or pad_w:
        mass_up = mass_up[: mass_up.shape[0] - pad_h, : mass_up.shape[1] - pad_w]

    y0, y1, x0, x1 = bbox
    target_h = y1 - y0
    target_w = x1 - x0
    h = min(target_h, mass_up.shape[0])
    w = min(target_w, mass_up.shape[1])

    full = np.zeros(orig_shape, dtype=mass_up.dtype)
    full[y0 : y0 + h, x0 : x0 + w] = mass_up[:h, :w]
    return full


def plot_creation_destruction_overlay(
    src_mask_hw: np.ndarray,
    tgt_mask_hw: np.ndarray,
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    transform_meta: Optional[dict] = None,
    alpha: float = 0.6,
    output_path: Optional[str] = None,
) -> plt.Figure:
    created_plot = _expand_mass_map_to_full(mass_created_hw, transform_meta)
    if created_plot is None:
        created_plot = mass_created_hw

    destroyed_plot = _expand_mass_map_to_full(mass_destroyed_hw, transform_meta)
    if destroyed_plot is None:
        destroyed_plot = mass_destroyed_hw

    created_plot = np.maximum(created_plot, 0.0)
    destroyed_plot = np.maximum(destroyed_plot, 0.0)
    vmax_created = float(created_plot.max()) if created_plot.size else 0.0
    vmax_destroyed = float(destroyed_plot.max()) if destroyed_plot.size else 0.0
    vmax_created = max(vmax_created, 1e-12)
    vmax_destroyed = max(vmax_destroyed, 1e-12)

    if created_plot.shape != tgt_mask_hw.shape:
        tgt_plot = _downsample_mask_to_shape(tgt_mask_hw, created_plot.shape)
    else:
        tgt_plot = tgt_mask_hw

    if destroyed_plot.shape != src_mask_hw.shape:
        src_plot = _downsample_mask_to_shape(src_mask_hw, destroyed_plot.shape)
    else:
        src_plot = src_mask_hw

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].imshow(tgt_plot, cmap="gray")
    im0 = axes[0].imshow(created_plot, cmap="magma", alpha=alpha, vmin=0.0, vmax=vmax_created)
    axes[0].set_title("Mass created (overlay on target)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(src_plot, cmap="gray")
    im1 = axes[1].imshow(destroyed_plot, cmap="magma", alpha=alpha, vmin=0.0, vmax=vmax_destroyed)
    axes[1].set_title("Mass destroyed (overlay on source)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def plot_velocity_overlay(
    mask_hw: np.ndarray,
    velocity_field_yx_hw2: np.ndarray,
    stride: int = 12,
    output_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    ax.imshow(mask_hw, cmap="gray")

    h, w = mask_hw.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    v = velocity_field_yx_hw2[0:h:stride, 0:w:stride]
    vy = v[..., 0]
    vx = v[..., 1]

    ax.quiver(xx, yy, vx, vy, color="cyan", angles="xy", scale_units="xy", scale=1.0)
    ax.set_title("Velocity field (quiver)")
    ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def _sample_distances(
    coupling: Coupling,
    support_src_yx: np.ndarray,
    support_tgt_yx: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if sp is not None and sp.issparse(coupling):
        coo = coupling.tocoo()
        if coo.nnz == 0:
            return np.array([]), None
        if coo.nnz <= max_samples:
            src = support_src_yx[coo.row]
            tgt = support_tgt_yx[coo.col]
            dists = np.linalg.norm(src - tgt, axis=1)
            return dists, coo.data
        idx = rng.choice(coo.nnz, size=max_samples, replace=False, p=coo.data / coo.data.sum())
        src = support_src_yx[coo.row[idx]]
        tgt = support_tgt_yx[coo.col[idx]]
        dists = np.linalg.norm(src - tgt, axis=1)
        return dists, None

    coupling = np.asarray(coupling)
    row_sums = coupling.sum(axis=1)
    total = float(row_sums.sum())
    if total <= 0:
        return np.array([]), None
    p_rows = row_sums / total
    n_samples = min(max_samples, coupling.shape[0] * coupling.shape[1])
    dists = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        r = rng.choice(len(p_rows), p=p_rows)
        row = coupling[r]
        row_sum = row_sums[r]
        if row_sum <= 0:
            continue
        p_cols = row / row_sum
        c = rng.choice(coupling.shape[1], p=p_cols)
        dists[i] = np.linalg.norm(support_src_yx[r] - support_tgt_yx[c])
    return dists, None


def plot_transport_spectrum(
    coupling: Coupling,
    support_src_yx: np.ndarray,
    support_tgt_yx: np.ndarray,
    bins: int = 30,
    max_samples: int = 50000,
    output_path: Optional[str] = None,
) -> plt.Figure:
    rng = np.random.default_rng(0)
    dists, weights = _sample_distances(coupling, support_src_yx, support_tgt_yx, max_samples, rng)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    if dists.size == 0:
        ax.text(0.5, 0.5, "No transport mass", ha="center", va="center")
    else:
        ax.hist(dists, bins=bins, weights=weights, density=True, color="steelblue", alpha=0.8)
    ax.set_title("Transport spectrum")
    ax.set_xlabel("Transport distance (px)")
    ax.set_ylabel("Density")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


# ---------------------------------------------------------------------------
# Phase 2: NaN contract enforcement + summary plots
# ---------------------------------------------------------------------------

def apply_nan_mask(field: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    """Apply NaN masking: non-support pixels become NaN.

    Prevents confusion between "no data" and "zero motion/mass".

    Args:
        field: (H, W) or (H, W, C) array
        support_mask: (H, W) boolean-like array, True = valid

    Returns:
        Copy of field with NaN outside support.
    """
    mask_bool = np.asarray(support_mask).astype(bool)
    out = np.array(field, dtype=np.float64)
    if out.ndim == 2:
        out[~mask_bool] = np.nan
    elif out.ndim == 3:
        out[~mask_bool, :] = np.nan
    else:
        raise ValueError(f"field must be 2D or 3D, got {out.ndim}D")
    return out


def _build_support_mask_from_result(result) -> np.ndarray:
    """Build a combined support mask from a canonical-grid result."""
    shape = result.mass_created_canon.shape[:2]
    mask = np.zeros(shape, dtype=bool)
    mask |= (result.mass_created_canon > 0)
    mask |= (result.mass_destroyed_canon > 0)
    vel_mag = np.linalg.norm(result.velocity_canon_px_per_step_yx, axis=-1)
    mask |= (vel_mag > 0)
    return mask


def plot_uot_summary(
    result,
    output_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """4-panel UOT summary with NaN masking and numeric annotations.

    Panel 1: Support mask
    Panel 2: Velocity quiver on support
    Panel 3: Mass creation heatmap (NaN masked)
    Panel 4: Mass destruction heatmap (NaN masked)
    """
    support_mask = _build_support_mask_from_result(result)

    created = apply_nan_mask(result.mass_created_canon, support_mask)
    destroyed = apply_nan_mask(result.mass_destroyed_canon, support_mask)
    velocity = result.velocity_canon_px_per_step_yx
    vel_mag = np.linalg.norm(velocity, axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=14)

    # Panel 1: Support mask
    ax = axes[0, 0]
    ax.imshow(support_mask.astype(float), cmap="gray", vmin=0, vmax=1)
    n_support = int(support_mask.sum())
    ax.set_title(f"Support mask ({n_support} px)")
    ax.axis("off")

    # Panel 2: Velocity quiver
    ax = axes[0, 1]
    ax.imshow(support_mask.astype(float), cmap="gray", vmin=0, vmax=1, alpha=0.3)
    h, w = support_mask.shape
    stride = max(1, min(h, w) // 20)
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    vy = velocity[0:h:stride, 0:w:stride, 0]
    vx = velocity[0:h:stride, 0:w:stride, 1]
    ax.quiver(xx, yy, vx, vy, color="cyan", angles="xy", scale_units="xy", scale=1.0)
    max_vel = float(np.nanmax(vel_mag)) if vel_mag.size else 0.0
    mean_vel = float(np.nanmean(vel_mag[support_mask])) if support_mask.any() else 0.0
    ax.set_title(f"Velocity (max={max_vel:.2f}, mean={mean_vel:.2f} px/step)")
    ax.axis("off")

    # Panel 3: Mass created
    ax = axes[1, 0]
    vmax_c = float(np.nanmax(created)) if np.any(np.isfinite(created)) else 1e-12
    vmax_c = max(vmax_c, 1e-12)
    im = ax.imshow(created, cmap="magma", vmin=0, vmax=vmax_c)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    total_created = float(np.nansum(created))
    ax.set_title(f"Mass created (total={total_created:.4f})")
    ax.axis("off")

    # Panel 4: Mass destroyed
    ax = axes[1, 1]
    vmax_d = float(np.nanmax(destroyed)) if np.any(np.isfinite(destroyed)) else 1e-12
    vmax_d = max(vmax_d, 1e-12)
    im = ax.imshow(destroyed, cmap="magma", vmin=0, vmax=vmax_d)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    total_destroyed = float(np.nansum(destroyed))
    ax.set_title(f"Mass destroyed (total={total_destroyed:.4f})")
    ax.axis("off")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def plot_velocity_histogram(
    result,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of velocity magnitudes on support pixels."""
    support_mask = _build_support_mask_from_result(result)
    vel_mag = np.linalg.norm(result.velocity_canon_px_per_step_yx, axis=-1)
    valid_mags = vel_mag[support_mask]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    if valid_mags.size == 0:
        ax.text(0.5, 0.5, "No velocity data", ha="center", va="center")
    else:
        ax.hist(valid_mags, bins=30, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axvline(float(np.mean(valid_mags)), color="red", linestyle="--", label=f"mean={np.mean(valid_mags):.2f}")
        ax.legend()
    ax.set_title("Velocity magnitude distribution")
    ax.set_xlabel("Velocity (px/step)")
    ax.set_ylabel("Count")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def write_diagnostics_json(result, output_path: str):
    """Write UOT result diagnostics to JSON."""
    import json

    diagnostics = result.diagnostics or {}
    metrics = diagnostics.get("metrics", {})

    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)

    out["cost"] = float(result.cost)

    support_mask = _build_support_mask_from_result(result)
    out["n_support_pixels"] = int(support_mask.sum())
    out["total_mass_created"] = float(result.mass_created_canon.sum())
    out["total_mass_destroyed"] = float(result.mass_destroyed_canon.sum())

    vel_mag = np.linalg.norm(result.velocity_canon_px_per_step_yx, axis=-1)
    valid_mags = vel_mag[support_mask]
    if valid_mags.size > 0:
        out["mean_velocity_px_per_step"] = float(np.mean(valid_mags))
        out["max_velocity_px_per_step"] = float(np.max(valid_mags))
    else:
        out["mean_velocity_px_per_step"] = 0.0
        out["max_velocity_px_per_step"] = 0.0

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
