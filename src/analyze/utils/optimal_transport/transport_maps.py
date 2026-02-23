"""Create mass maps and velocity fields from UOT coupling."""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

from analyze.utils.optimal_transport.config import Coupling

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from .config import PairFrameGeometry

from .multiscale_sampling import downsample_density


def _compute_marginals(coupling: Coupling) -> Tuple[np.ndarray, np.ndarray]:
    if sp is not None and sp.issparse(coupling):
        mu_hat = np.asarray(coupling.sum(axis=1)).ravel()
        nu_hat = np.asarray(coupling.sum(axis=0)).ravel()
        return mu_hat, nu_hat
    coupling = np.asarray(coupling)
    return coupling.sum(axis=1), coupling.sum(axis=0)


def rasterize_mass_to_canonical(
    mass_work: np.ndarray,
    pair_frame: "PairFrameGeometry",
) -> np.ndarray:
    """
    Rasterize work-space mass map back to canonical grid using coverage-aware splatting.

    Implements spec Section 4.C.1 with correct handling of boundary pixels.
    Each work pixel's mass is distributed ONLY across the real canonical pixels
    it represents, not uniformly across s×s (which would leak into padding).

    Mass conservation: sum(output) == sum(mass_work) within float tolerance.

    Args:
        mass_work: (Hw, Ww) mass map in work space
        pair_frame: Pair frame geometry

    Returns:
        (Hc, Wc) mass map in canonical space (zeros outside pair crop)
    """
    canon_h, canon_w = pair_frame.canon_shape_hw
    canonical = np.zeros((canon_h, canon_w), dtype=np.float32)

    s = pair_frame.downsample_factor
    bbox = pair_frame.pair_crop_box_yx

    crop_h, crop_w = bbox.h, bbox.w
    pad_h, pad_w = pair_frame.crop_pad_hw
    padded_h, padded_w = crop_h + pad_h, crop_w + pad_w

    # 1) Build padded-crop valid mask: 1 for real crop pixels, 0 for padding pixels
    valid = np.ones((padded_h, padded_w), dtype=np.float32)
    if pad_h > 0:
        valid[crop_h:, :] = 0.0
    if pad_w > 0:
        valid[:, crop_w:] = 0.0

    # 2) Downsample valid mask with the SAME sum-pooling used elsewhere
    #    coverage_work[i,j] = number of real canonical pixels in that work pixel's block
    coverage_work = downsample_density(valid, s)  # shape == mass_work.shape

    # 3) Avoid divide-by-zero. Fully padded blocks should have zero mass anyway.
    #    (If they don't, that's a separate bug worth asserting.)
    assert mass_work.shape == coverage_work.shape
    if np.any((coverage_work == 0) & (mass_work != 0)):
        bad = float(np.max(np.abs(mass_work[coverage_work == 0])))
        raise AssertionError(f"Nonzero mass in fully padded work pixels; max={bad}")

    # 4) Distribute each work pixel mass across ONLY its real pixels
    mass_per_real_px = np.zeros_like(mass_work, dtype=np.float32)
    nz = coverage_work > 0
    mass_per_real_px[nz] = mass_work[nz] / coverage_work[nz]

    expanded = np.kron(mass_per_real_px, np.ones((s, s), dtype=np.float32))

    # 5) Zero out padding pixels explicitly (important for boundary blocks)
    expanded *= valid

    # 6) Paste ONLY the real crop region into canonical (padding does not exist in canonical)
    canonical[bbox.y0:bbox.y0+crop_h, bbox.x0:bbox.x0+crop_w] += expanded[:crop_h, :crop_w]

    # 7) Golden test 6.5: Mass conservation over the REAL region
    total_work = float(mass_work.sum())
    total_canon = float(canonical.sum())
    assert np.isclose(total_work, total_canon, rtol=1e-6, atol=1e-6), \
        f"Mass not conserved: work={total_work:.6f}, canon={total_canon:.6f}"

    return canonical


def rasterize_velocity_to_canonical(
    velocity_work_px: np.ndarray,
    pair_frame: "PairFrameGeometry",
    convert_to_um: bool = False,
) -> np.ndarray:
    """
    Rasterize work-space velocity field to canonical grid.

    Velocity is NOT a conserved quantity, so we don't use coverage normalization.
    We simply expand via kron, mask out padding, and paste the real crop.

    Args:
        velocity_work_px: (Hw, Ww, 2) velocity in work pixels per step/frame
        pair_frame: Pair frame geometry
        convert_to_um: If True, scale to μm/step. If False, scale to canonical px/step.

    Returns:
        (Hc, Wc, 2) velocity field in canonical space
    """
    canon_h, canon_w = pair_frame.canon_shape_hw
    canonical = np.zeros((canon_h, canon_w, 2), dtype=np.float32)

    s = pair_frame.downsample_factor
    bbox = pair_frame.pair_crop_box_yx

    crop_h, crop_w = bbox.h, bbox.w
    pad_h, pad_w = pair_frame.crop_pad_hw
    padded_h, padded_w = crop_h + pad_h, crop_w + pad_w

    # Scale factor for velocity magnitude (dy, dx ordering is preserved).
    if convert_to_um:
        scale = pair_frame.work_px_size_um
    else:
        scale = float(s)  # Convert work pixels to canonical pixels

    # Build valid mask for padding
    valid = np.ones((padded_h, padded_w), dtype=np.float32)
    if pad_h > 0:
        valid[crop_h:, :] = 0.0
    if pad_w > 0:
        valid[:, crop_w:] = 0.0

    # Expand each component via kron (no coverage division for velocity!)
    v_y = velocity_work_px[..., 0] * scale
    v_x = velocity_work_px[..., 1] * scale

    expanded_y = np.kron(v_y, np.ones((s, s), dtype=np.float32))
    expanded_x = np.kron(v_x, np.ones((s, s), dtype=np.float32))

    # Zero out padding pixels
    expanded_y *= valid
    expanded_x *= valid

    # Paste ONLY the real crop region into canonical
    canonical[bbox.y0:bbox.y0+crop_h, bbox.x0:bbox.x0+crop_w, 0] += expanded_y[:crop_h, :crop_w]
    canonical[bbox.y0:bbox.y0+crop_h, bbox.x0:bbox.x0+crop_w, 1] += expanded_x[:crop_h, :crop_w]

    return canonical


def rasterize_scalar_to_canonical(
    scalar_work: np.ndarray,
    pair_frame: "PairFrameGeometry",
) -> np.ndarray:
    """
    Rasterize a scalar work-space map to canonical grid without conservation.

    This mirrors velocity rasterization: values are expanded via kron,
    padding is masked, and the real crop is pasted into canonical.
    """
    canon_h, canon_w = pair_frame.canon_shape_hw
    canonical = np.zeros((canon_h, canon_w), dtype=np.float32)

    s = pair_frame.downsample_factor
    bbox = pair_frame.pair_crop_box_yx

    crop_h, crop_w = bbox.h, bbox.w
    pad_h, pad_w = pair_frame.crop_pad_hw
    padded_h, padded_w = crop_h + pad_h, crop_w + pad_w

    valid = np.ones((padded_h, padded_w), dtype=np.float32)
    if pad_h > 0:
        valid[crop_h:, :] = 0.0
    if pad_w > 0:
        valid[:, crop_w:] = 0.0

    expanded = np.kron(scalar_work, np.ones((s, s), dtype=np.float32))
    expanded *= valid

    canonical[bbox.y0:bbox.y0+crop_h, bbox.x0:bbox.x0+crop_w] += expanded[:crop_h, :crop_w]

    return canonical


def compute_transport_maps(
    coupling: Coupling,
    support_src_yx: np.ndarray,
    support_tgt_yx: np.ndarray,
    weights_src: np.ndarray,
    weights_tgt: np.ndarray,
    work_shape_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute transport maps on the work grid.

    Lifting to canonical is owned by `working_grid.py` (via rasterize_* helpers).
    """
    if coupling is None:
        raise ValueError("Coupling is required to compute transport maps.")

    mu_hat, nu_hat = _compute_marginals(coupling)
    mass_destroyed = np.maximum(0.0, weights_src - mu_hat)
    mass_created = np.maximum(0.0, weights_tgt - nu_hat)

    mass_destroyed_hw = np.zeros(work_shape_hw, dtype=np.float32)
    mass_created_hw = np.zeros(work_shape_hw, dtype=np.float32)

    src_y = support_src_yx[:, 0].astype(int)
    src_x = support_src_yx[:, 1].astype(int)
    tgt_y = support_tgt_yx[:, 0].astype(int)
    tgt_x = support_tgt_yx[:, 1].astype(int)

    mass_destroyed_hw[src_y, src_x] = mass_destroyed.astype(np.float32)
    mass_created_hw[tgt_y, tgt_x] = mass_created.astype(np.float32)

    velocity_field = np.zeros((*work_shape_hw, 2), dtype=np.float32)

    if sp is not None and sp.issparse(coupling):
        coupling = coupling.tocoo()
        n_src = len(weights_src)
        sum_y = np.zeros((n_src, 2), dtype=np.float64)
        np.add.at(sum_y, coupling.row, coupling.data[:, None] * support_tgt_yx[coupling.col])
        mu_hat_safe = np.maximum(mu_hat, 1e-12)
        T = sum_y / mu_hat_safe[:, None]
    else:
        coupling_dense = np.asarray(coupling, dtype=np.float64)
        mu_hat_safe = np.maximum(mu_hat, 1e-12)
        T = (coupling_dense @ support_tgt_yx) / mu_hat_safe[:, None]

    v = T - support_src_yx
    velocity_field[src_y, src_x, :] = v.astype(np.float32)

    return mass_created_hw, mass_destroyed_hw, velocity_field


def compute_cost_maps(
    cost_per_src: np.ndarray,
    cost_per_tgt: np.ndarray,
    support_src_yx: np.ndarray,
    support_tgt_yx: np.ndarray,
    work_shape_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize per-support transport cost to work or canonical grids.

    Cost is attributed per support point:
      - src: sum_j (P_ij * C_ij)
      - tgt: sum_i (P_ij * C_ij)
    """
    cost_src_hw = np.zeros(work_shape_hw, dtype=np.float32)
    cost_tgt_hw = np.zeros(work_shape_hw, dtype=np.float32)

    src_y = support_src_yx[:, 0].astype(int)
    src_x = support_src_yx[:, 1].astype(int)
    tgt_y = support_tgt_yx[:, 0].astype(int)
    tgt_x = support_tgt_yx[:, 1].astype(int)

    cost_src_hw[src_y, src_x] = cost_per_src.astype(np.float32)
    cost_tgt_hw[tgt_y, tgt_x] = cost_per_tgt.astype(np.float32)

    return cost_src_hw, cost_tgt_hw
