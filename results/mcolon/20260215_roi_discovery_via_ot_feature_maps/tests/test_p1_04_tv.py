"""
Phase 1 / Task 1.4 — Total Variation edge construction and computation.

Checks:
- Edge list only contains edges where BOTH endpoints are inside mask
- No edges cross the mask boundary
- A constant weight map has TV = 0
- A weight map with a single sharp step has predictable TV
- Boundary fraction detects ROI at the mask edge vs interior
"""

import numpy as np
import pytest

from roi_tv import build_tv_edges, compute_boundary_fraction, compute_tv_numpy


# ---- Edge construction ----

def test_edges_only_inside_mask(mask_ref):
    """Every edge (src, tgt) must have both endpoints inside the mask."""
    edges_src, edges_tgt = build_tv_edges(mask_ref)
    H, W = mask_ref.shape
    flat_mask = mask_ref.ravel()

    assert np.all(flat_mask[edges_src]), "Some edge sources are outside mask"
    assert np.all(flat_mask[edges_tgt]), "Some edge targets are outside mask"


def test_no_edges_on_empty_mask():
    """A fully-False mask should produce zero edges."""
    mask = np.zeros((16, 16), dtype=bool)
    src, tgt = build_tv_edges(mask)
    assert len(src) == 0
    assert len(tgt) == 0


def test_full_mask_edge_count():
    """A fully-True HxW mask should have (H-1)*W + H*(W-1) horizontal+vertical edges."""
    H, W = 8, 8
    mask = np.ones((H, W), dtype=bool)
    src, tgt = build_tv_edges(mask)
    expected = (H - 1) * W + H * (W - 1)
    assert len(src) == expected, f"Expected {expected} edges, got {len(src)}"


def test_edges_are_4_neighborhood(mask_ref):
    """Each edge should connect adjacent pixels (Manhattan distance = 1)."""
    edges_src, edges_tgt = build_tv_edges(mask_ref)
    H, W = mask_ref.shape

    src_r, src_c = np.divmod(edges_src, W)
    tgt_r, tgt_c = np.divmod(edges_tgt, W)

    manhattan = np.abs(src_r - tgt_r) + np.abs(src_c - tgt_c)
    assert np.all(manhattan == 1), "Found edges with Manhattan distance != 1"


# ---- TV computation ----

def test_constant_map_has_zero_tv(mask_ref):
    """A spatially constant weight map should have TV = 0."""
    w = np.ones((mask_ref.shape[0], mask_ref.shape[1]), dtype=np.float32) * 3.14
    tv = compute_tv_numpy(w, mask_ref)
    assert tv == pytest.approx(0.0, abs=1e-6)


def test_step_function_tv():
    """
    A step function on a full 4x4 mask with a horizontal edge between
    rows 1 and 2 should have TV = 4 * |step_height| (one edge per column).
    """
    mask = np.ones((4, 4), dtype=bool)
    w = np.zeros((4, 4), dtype=np.float32)
    w[2:, :] = 5.0  # step of height 5 between row 1 and 2

    tv = compute_tv_numpy(w, mask)
    # Horizontal edges crossing the step: 4 columns * |5| = 20
    # Plus no vertical steps within top or bottom blocks
    assert tv == pytest.approx(20.0, abs=1e-6)


def test_multichannel_tv(mask_ref):
    """TV on multi-channel map sums across channels."""
    H, W = mask_ref.shape
    w = np.zeros((H, W, 2), dtype=np.float32)
    # Channel 0: constant -> TV = 0
    w[:, :, 0] = 1.0
    # Channel 1: random -> TV > 0
    rng = np.random.default_rng(42)
    w[:, :, 1] = rng.normal(0, 1, (H, W))

    tv = compute_tv_numpy(w, mask_ref)
    assert tv > 0


# ---- Boundary fraction ----

def test_boundary_fraction_interior_roi(mask_ref):
    """An ROI entirely in the interior should have low boundary fraction."""
    H, W = mask_ref.shape
    roi = np.zeros((H, W), dtype=bool)
    # Small square in the center
    roi[H // 2 - 2:H // 2 + 2, W // 2 - 2:W // 2 + 2] = True
    roi = roi & mask_ref

    bf = compute_boundary_fraction(roi, mask_ref, band_width=3)
    assert bf < 0.3, f"Interior ROI has high boundary_fraction={bf}"


def test_boundary_fraction_edge_roi(mask_ref):
    """An ROI along the mask edge should have high boundary fraction."""
    from scipy.ndimage import binary_erosion

    # ROI = mask minus eroded interior -> only the boundary band
    interior = binary_erosion(mask_ref, iterations=3)
    roi = mask_ref & ~interior

    bf = compute_boundary_fraction(roi, mask_ref, band_width=3)
    assert bf > 0.8, f"Edge ROI has low boundary_fraction={bf}"


def test_boundary_fraction_empty_roi(mask_ref):
    """Empty ROI should return 0.0."""
    roi = np.zeros_like(mask_ref)
    bf = compute_boundary_fraction(roi, mask_ref)
    assert bf == 0.0
