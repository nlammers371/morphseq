from __future__ import annotations

import numpy as np

from analyze.utils.coord.transforms import TransformChain
from analyze.utils.coord.types import CanonicalGrid, CanonicalMaskResult
from analyze.utils.optimal_transport.working_grid import WorkingGridConfig, prepare_working_grid_pair


def test_working_grid_requires_canonical_label():
    grid = CanonicalGrid(
        um_per_px=10.0,
        shape_yx=(32, 32),
        anchor_mode="yolk_anchor",
        anchor_yx=(0.5, 0.5),
    )
    src = CanonicalMaskResult(
        mask=np.ones((32, 32), dtype=np.uint8),
        grid=grid,
        transform_chain=TransformChain.identity(shape_yx=(32, 32), interp="nearest"),
        meta={"coord_frame_id": "unknown", "coord_frame_version": 1, "coord_convention": "yx"},
    )
    tgt = CanonicalMaskResult(
        mask=np.ones((32, 32), dtype=np.uint8),
        grid=grid,
        transform_chain=TransformChain.identity(shape_yx=(32, 32), interp="nearest"),
        meta={"coord_frame_id": "canonical_grid", "coord_frame_version": 1, "coord_convention": "yx"},
    )
    try:
        prepare_working_grid_pair(src, tgt, WorkingGridConfig(downsample_factor=2, padding_px=0))
    except ValueError as e:
        assert "canonical-grid mask" in str(e)
    else:
        raise AssertionError("Expected ValueError for non-canonical input")


def test_register_to_fixed_identity_on_empty_masks():
    from analyze.utils.coord.register import register_to_fixed

    moving = np.zeros((32, 32), dtype=np.uint8)
    fixed = np.zeros((32, 32), dtype=np.uint8)
    reg = register_to_fixed(moving=moving, fixed=fixed, apply=False)
    assert reg.applied is False
    assert len(reg.transform.transforms) >= 1
    assert reg.moving_in_fixed is None
