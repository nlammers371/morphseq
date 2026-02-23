"""
Total Variation with mask-aware boundary behavior.

TV is defined over edges in a 4-neighborhood graph. An edge (p, q) is
included ONLY if both p and q are inside mask_ref. This means:
- No zero-padding outside the embryo mask
- Boundary pixels have fewer valid neighbors ("reduced-degree boundary")

This module provides both a NumPy reference implementation and a
JAX-compatible implementation for use in the differentiable trainer.

See PLAN.md Section C for the full specification.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Edge list construction (NumPy, run once at setup)
# ---------------------------------------------------------------------------

def build_tv_edges(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the list of valid TV edges from a binary mask.

    An edge connects pixel p to pixel q in the 4-neighborhood
    if and only if both p and q are inside the mask.

    Parameters
    ----------
    mask : ndarray, shape (H, W), bool or uint8
        Reference embryo mask. Nonzero = inside.

    Returns
    -------
    edges_src : ndarray, shape (E,), int
        Flat indices of edge source pixels.
    edges_tgt : ndarray, shape (E,), int
        Flat indices of edge target pixels.
    """
    mask_bool = mask.astype(bool)
    H, W = mask_bool.shape

    src_list = []
    tgt_list = []

    # Horizontal edges: (i, j) <-> (i, j+1)
    valid_h = mask_bool[:, :-1] & mask_bool[:, 1:]
    rows, cols = np.where(valid_h)
    src_flat = rows * W + cols
    tgt_flat = rows * W + (cols + 1)
    src_list.append(src_flat)
    tgt_list.append(tgt_flat)

    # Vertical edges: (i, j) <-> (i+1, j)
    valid_v = mask_bool[:-1, :] & mask_bool[1:, :]
    rows, cols = np.where(valid_v)
    src_flat = rows * W + cols
    tgt_flat = (rows + 1) * W + cols
    src_list.append(src_flat)
    tgt_list.append(tgt_flat)

    edges_src = np.concatenate(src_list)
    edges_tgt = np.concatenate(tgt_list)

    return edges_src, edges_tgt


def compute_tv_numpy(
    w: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute anisotropic TV of w restricted to mask edges (NumPy reference).

    TV(w) = sum_{(p,q) in edges} |w_p - w_q|

    For multi-channel w, sums across channels.

    Parameters
    ----------
    w : ndarray, shape (H, W) or (H, W, C)
        Weight map.
    mask : ndarray, shape (H, W)
        Reference mask.

    Returns
    -------
    float
        Total variation value.
    """
    edges_src, edges_tgt = build_tv_edges(mask)

    if w.ndim == 2:
        w_flat = w.ravel()
        return float(np.sum(np.abs(w_flat[edges_src] - w_flat[edges_tgt])))
    elif w.ndim == 3:
        H, W, C = w.shape
        w_flat = w.reshape(H * W, C)
        diffs = w_flat[edges_src] - w_flat[edges_tgt]  # (E, C)
        return float(np.sum(np.abs(diffs)))
    else:
        raise ValueError(f"w must be 2D or 3D, got {w.ndim}D")


def compute_boundary_fraction(
    roi_mask: np.ndarray,
    ref_mask: np.ndarray,
    band_width: int = 3,
) -> float:
    """
    Compute the fraction of ROI pixels that lie in a thin boundary band.

    Used as a diagnostic to detect registration-driven edge artifacts.

    Parameters
    ----------
    roi_mask : ndarray, shape (H, W), bool
        Thresholded ROI region.
    ref_mask : ndarray, shape (H, W), bool
        Reference embryo mask.
    band_width : int
        Width of the boundary band in pixels.

    Returns
    -------
    float
        Fraction of ROI pixels in the boundary band (0.0 to 1.0).
    """
    from scipy.ndimage import binary_erosion

    ref_bool = ref_mask.astype(bool)
    roi_bool = roi_mask.astype(bool)

    interior = binary_erosion(ref_bool, iterations=band_width)
    boundary_band = ref_bool & ~interior

    roi_area = roi_bool.sum()
    if roi_area == 0:
        return 0.0

    roi_in_boundary = (roi_bool & boundary_band).sum()
    return float(roi_in_boundary) / float(roi_area)


# ---------------------------------------------------------------------------
# JAX-compatible TV (for differentiable training)
# ---------------------------------------------------------------------------

if _JAX_AVAILABLE:
    def compute_tv_jax(
        w_flat: jnp.ndarray,
        edges_src: jnp.ndarray,
        edges_tgt: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JAX-compatible anisotropic TV using precomputed edge indices.

        Parameters
        ----------
        w_flat : jnp array, shape (H*W,) or (H*W, C)
            Flattened weight map.
        edges_src : jnp array, shape (E,), int
            Edge source indices.
        edges_tgt : jnp array, shape (E,), int
            Edge target indices.

        Returns
        -------
        Scalar TV value (differentiable).
        """
        if w_flat.ndim == 1:
            diffs = w_flat[edges_src] - w_flat[edges_tgt]
            return jnp.sum(jnp.abs(diffs))
        else:
            diffs = w_flat[edges_src] - w_flat[edges_tgt]  # (E, C)
            return jnp.sum(jnp.abs(diffs))


__all__ = [
    "build_tv_edges",
    "compute_tv_numpy",
    "compute_boundary_fraction",
]

if _JAX_AVAILABLE:
    __all__.append("compute_tv_jax")
