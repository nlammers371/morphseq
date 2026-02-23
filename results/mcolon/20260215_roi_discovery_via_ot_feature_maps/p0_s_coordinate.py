"""
Phase 0 Step 4: Compute S coordinate map on the WT reference template.

Produces S_map_ref(x,y) in [0,1] for each pixel in mask_ref, where
S=0 is head (rostral) and S=1 is tail (caudal).

Also computes optional local basis vectors (tangent_ref, normal_ref)
for projecting displacement vectors into parallel/perpendicular components.

Uses the existing centerline extraction API from:
    segmentation_sandbox/scripts/body_axis_analysis/centerline_extraction.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

from roi_config import Phase0SCoordinateConfig

logger = logging.getLogger(__name__)


def extract_centerline_from_mask(
    mask_ref: np.ndarray,
    config: Phase0SCoordinateConfig = Phase0SCoordinateConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a smoothed centerline from the reference mask.

    Returns (spline_x, spline_y, curvature, arc_length).
    Orientation: head-to-tail (S=0 head, S=1 tail).
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "segmentation_sandbox" / "scripts"))
        from body_axis_analysis.centerline_extraction import extract_centerline

        spline_x, spline_y, curvature, arc_length = extract_centerline(
            mask_ref,
            method=config.centerline_method,
            orient_head_to_tail=config.orient_head_to_tail,
            bspline_smoothing=config.bspline_smoothing,
            random_seed=config.random_seed,
        )
        logger.info(
            f"Centerline extracted: {len(spline_x)} points, "
            f"length={arc_length[-1]:.1f} px"
        )
        return spline_x, spline_y, curvature, arc_length

    except ImportError:
        logger.warning("body_axis_analysis not available, using PCA fallback")
        return _pca_centerline_fallback(mask_ref)


def _pca_centerline_fallback(mask_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple PCA-based centerline when the full analyzer isn't available."""
    ys, xs = np.where(mask_ref > 0)
    if len(ys) == 0:
        raise ValueError("Empty mask")

    coords = np.column_stack([xs, ys]).astype(float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # PCA
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]  # principal component

    # Project onto PC1
    projections = centered @ pc1
    order = np.argsort(projections)
    sorted_coords = coords[order]

    # Subsample for smoothing
    n_pts = min(200, len(sorted_coords))
    indices = np.linspace(0, len(sorted_coords) - 1, n_pts, dtype=int)
    pts = sorted_coords[indices]

    # B-spline fit
    tck, u = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 5, k=3)
    u_fine = np.linspace(0, 1, 500)
    spline_x, spline_y = splev(u_fine, tck)

    # Arc length
    dx = np.diff(spline_x)
    dy = np.diff(spline_y)
    ds = np.sqrt(dx**2 + dy**2)
    arc_length = np.concatenate([[0], np.cumsum(ds)])

    # Curvature (finite differences)
    ddx = np.gradient(np.gradient(spline_x))
    ddy = np.gradient(np.gradient(spline_y))
    dx1 = np.gradient(spline_x)
    dy1 = np.gradient(spline_y)
    curvature = np.abs(dx1 * ddy - dy1 * ddx) / (dx1**2 + dy1**2 + 1e-12)**1.5

    return np.array(spline_x), np.array(spline_y), curvature, arc_length


def compute_s_map(
    mask_ref: np.ndarray,
    spline_x: np.ndarray,
    spline_y: np.ndarray,
    arc_length: np.ndarray,
) -> np.ndarray:
    """
    Compute S_map_ref(x,y) in [0,1] for each pixel in mask_ref.

    For each mask pixel, find its nearest point on the spline and assign
    the normalized arc-length parameter.

    Returns S_map of shape (H, W) with S in [0,1] inside mask, NaN outside.
    """
    H, W = mask_ref.shape
    S_map = np.full((H, W), np.nan, dtype=np.float32)

    ys, xs = np.where(mask_ref > 0)
    if len(ys) == 0:
        return S_map

    # Build KD-tree on spline points
    spline_pts = np.column_stack([spline_x, spline_y])
    tree = cKDTree(spline_pts)

    # Query nearest spline point for each mask pixel
    pixel_pts = np.column_stack([xs, ys])
    _, nearest_idx = tree.query(pixel_pts)

    # Normalize arc length to [0,1]
    total_length = arc_length[-1]
    if total_length > 0:
        s_values = arc_length[nearest_idx] / total_length
    else:
        s_values = np.zeros(len(nearest_idx))

    S_map[ys, xs] = s_values.astype(np.float32)

    logger.info(
        f"S_map computed: {len(ys)} pixels, S range=[{np.nanmin(S_map):.3f}, {np.nanmax(S_map):.3f}]"
    )
    return S_map


def compute_local_basis(
    mask_ref: np.ndarray,
    spline_x: np.ndarray,
    spline_y: np.ndarray,
    S_map: np.ndarray,
    arc_length: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local tangent and normal basis vectors at each mask pixel.

    tangent_ref[y,x] = unit tangent e_parallel at S(x,y) along the spline
    normal_ref[y,x] = unit normal e_perp (90° CCW from tangent)

    Returns (tangent_ref, normal_ref), each shape (H, W, 2).
    """
    H, W = mask_ref.shape
    tangent_ref = np.zeros((H, W, 2), dtype=np.float32)
    normal_ref = np.zeros((H, W, 2), dtype=np.float32)

    # Compute tangent along spline via finite differences
    dx = np.gradient(spline_x)
    dy = np.gradient(spline_y)
    mag = np.sqrt(dx**2 + dy**2) + 1e-12
    tangent_x = dx / mag
    tangent_y = dy / mag

    ys, xs = np.where(mask_ref > 0)
    if len(ys) == 0:
        return tangent_ref, normal_ref

    # For each pixel, find nearest spline point
    spline_pts = np.column_stack([spline_x, spline_y])
    tree = cKDTree(spline_pts)
    pixel_pts = np.column_stack([xs, ys])
    _, nearest_idx = tree.query(pixel_pts)

    # Assign tangent/normal from nearest spline point
    tangent_ref[ys, xs, 0] = tangent_x[nearest_idx]
    tangent_ref[ys, xs, 1] = tangent_y[nearest_idx]

    # Normal = 90° CCW rotation of tangent: (-ty, tx)
    normal_ref[ys, xs, 0] = -tangent_y[nearest_idx]
    normal_ref[ys, xs, 1] = tangent_x[nearest_idx]

    logger.info(f"Local basis computed for {len(ys)} pixels")
    return tangent_ref, normal_ref


def build_s_coordinate(
    mask_ref: np.ndarray,
    config: Phase0SCoordinateConfig = Phase0SCoordinateConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Full S coordinate computation: centerline → S_map → basis.

    Returns (S_map_ref, tangent_ref, normal_ref, info_dict).
    """
    spline_x, spline_y, curvature, arc_length = extract_centerline_from_mask(mask_ref, config)
    S_map = compute_s_map(mask_ref, spline_x, spline_y, arc_length)
    tangent_ref, normal_ref = compute_local_basis(mask_ref, spline_x, spline_y, S_map, arc_length)

    info = {
        "centerline_method": config.centerline_method,
        "n_spline_points": len(spline_x),
        "total_length_px": float(arc_length[-1]),
        "mean_curvature": float(np.mean(curvature)),
        "max_curvature": float(np.max(curvature)),
    }

    return S_map, tangent_ref, normal_ref, info


__all__ = [
    "extract_centerline_from_mask",
    "compute_s_map",
    "compute_local_basis",
    "build_s_coordinate",
]
