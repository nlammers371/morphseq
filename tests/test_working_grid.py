from pathlib import Path
import sys

import numpy as np
import pytest

# Most library code is imported as `analyze.*` with PYTHONPATH=src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze.utils.coord.transforms import TransformChain
from analyze.utils.coord.types import CanonicalGrid, CanonicalMaskResult
from analyze.utils.optimal_transport.working_grid import (
    WorkingGridConfig,
    prepare_working_grid_pair,
)


def _canon_mask(mask: np.ndarray, *, um_per_px: float = 10.0, meta: dict | None = None) -> CanonicalMaskResult:
    h, w = mask.shape
    grid = CanonicalGrid(
        um_per_px=float(um_per_px),
        shape_yx=(h, w),
        anchor_mode="yolk_anchor",
        anchor_yx=(0.5, 0.5),
    )
    if meta is None:
        meta = {"coord_frame_id": "canonical_grid", "coord_frame_version": 1, "coord_convention": "yx"}
    return CanonicalMaskResult(
        mask=mask,
        grid=grid,
        transform_chain=TransformChain.identity(shape_yx=(h, w), interp="nearest"),
        meta=meta,
    )


def test_prepare_working_grid_pair_requires_canonical_label():
    src = _canon_mask(np.ones((8, 8), dtype=np.uint8), meta={"coord_frame_id": "work_grid", "coord_frame_version": 1})
    tgt = _canon_mask(np.ones((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="canonical-grid mask"):
        prepare_working_grid_pair(src, tgt, WorkingGridConfig(downsample_factor=2, padding_px=0))


def test_prepare_working_grid_pair_requires_same_shape():
    src = _canon_mask(np.ones((8, 8), dtype=np.uint8))
    tgt = _canon_mask(np.ones((8, 9), dtype=np.uint8))
    with pytest.raises(ValueError, match="same shape"):
        prepare_working_grid_pair(src, tgt, WorkingGridConfig(downsample_factor=2, padding_px=0))


def test_lift_work_mass_conserves_sum():
    src_mask = np.zeros((10, 12), dtype=np.uint8)
    tgt_mask = np.zeros_like(src_mask)
    src_mask[2:6, 3:7] = 1  # 4x4 block (divisible by 2)
    tgt_mask[2:6, 3:7] = 1

    pair = prepare_working_grid_pair(
        _canon_mask(src_mask),
        _canon_mask(tgt_mask),
        WorkingGridConfig(downsample_factor=2, padding_px=0),
    )

    work_mass = pair.src_work_density
    canon_mass = pair.lift_work_mass_to_canonical(work_mass)

    assert np.isclose(float(work_mass.sum()), float(canon_mass.sum()), rtol=1e-6, atol=1e-6)


def test_lift_work_velocity_scales_by_downsample_factor():
    src_mask = np.zeros((10, 12), dtype=np.uint8)
    tgt_mask = np.zeros_like(src_mask)
    src_mask[2:6, 3:7] = 1
    tgt_mask[2:6, 3:7] = 1

    s = 2
    pair = prepare_working_grid_pair(
        _canon_mask(src_mask),
        _canon_mask(tgt_mask),
        WorkingGridConfig(downsample_factor=s, padding_px=0),
    )

    v_work = np.zeros((*pair.work_shape_hw, 2), dtype=np.float32)
    v_work[..., 0] = 1.0  # dy
    v_work[..., 1] = 2.0  # dx

    v_canon = pair.lift_work_velocity_to_canonical_px_per_step_yx(v_work)

    # Pick a pixel inside the real crop region.
    y, x = 2, 3
    assert np.isclose(float(v_canon[y, x, 0]), float(s * 1.0), rtol=1e-6, atol=1e-6)
    assert np.isclose(float(v_canon[y, x, 1]), float(s * 2.0), rtol=1e-6, atol=1e-6)

    # Outside crop should remain zero.
    assert np.allclose(v_canon[0, 0], 0.0)
