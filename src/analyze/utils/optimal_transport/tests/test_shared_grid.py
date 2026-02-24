"""Tests for SharedWorkingGrid and related functions."""

import numpy as np
import pytest

from analyze.utils.coord.types import BoxYX, CanonicalGrid, CanonicalMaskResult
from analyze.utils.coord.transforms import TransformChain
from analyze.utils.optimal_transport.shared_grid import (
    SharedWorkingGrid,
    prepare_shared_working_grid,
    make_working_grid_pair_from_shared,
)
from analyze.utils.optimal_transport.working_grid import (
    WorkingGridConfig,
    prepare_working_grid_pair,
)


def _make_canonical_mask(
    mask: np.ndarray,
    um_per_px: float = 10.0,
    shape_yx: tuple[int, int] | None = None,
    content_bbox_yx: BoxYX | None = "auto",
) -> CanonicalMaskResult:
    """Helper to make a CanonicalMaskResult for testing."""
    if shape_yx is None:
        shape_yx = mask.shape
    grid = CanonicalGrid(
        um_per_px=um_per_px,
        shape_yx=shape_yx,
        anchor_mode="yolk_anchor",
        anchor_yx=(shape_yx[0] * 0.5, shape_yx[1] * 0.5),
    )
    chain = TransformChain.identity(shape_yx=shape_yx, interp="nearest")
    meta = {
        "coord_frame_id": "canonical_grid",
        "coord_frame_version": 1,
        "coord_convention": "yx",
    }
    if content_bbox_yx == "auto":
        content_bbox_yx = BoxYX.from_mask(mask)
    return CanonicalMaskResult(
        mask=mask.astype(np.uint8),
        grid=grid,
        transform_chain=chain,
        meta=meta,
        content_bbox_yx=content_bbox_yx,
    )


def _make_test_mask(
    canvas_h: int = 64,
    canvas_w: int = 128,
    y0: int = 10,
    y1: int = 50,
    x0: int = 20,
    x1: int = 100,
) -> np.ndarray:
    """Create a simple rectangular binary mask on a canvas."""
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class TestPrepareSharedWorkingGrid:
    def test_basic_two_masks(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        masks = [_make_canonical_mask(m1), _make_canonical_mask(m2)]
        cfg = WorkingGridConfig(downsample_factor=2, padding_px=4)

        shared = prepare_shared_working_grid(masks, cfg, mask_ids=["a", "b"])

        assert shared.n_masks == 2
        assert len(shared.work_densities) == 2
        assert shared.mask_ids == ["a", "b"]
        # All densities same shape
        assert shared.work_densities[0].shape == shared.work_shape_hw
        assert shared.work_densities[1].shape == shared.work_shape_hw

    def test_single_mask(self):
        m = _make_test_mask()
        masks = [_make_canonical_mask(m)]
        cfg = WorkingGridConfig(downsample_factor=1, padding_px=2)

        shared = prepare_shared_working_grid(masks, cfg)
        assert shared.n_masks == 1
        assert shared.work_densities[0].shape == shared.work_shape_hw

    def test_all_densities_same_shape(self):
        masks = []
        for i in range(5):
            m = _make_test_mask(y0=10 + i * 2, y1=40 + i, x0=20 + i * 3, x1=90 + i)
            masks.append(_make_canonical_mask(m))
        cfg = WorkingGridConfig(downsample_factor=4, padding_px=8)

        shared = prepare_shared_working_grid(masks, cfg)
        shapes = {d.shape for d in shared.work_densities}
        assert len(shapes) == 1
        assert shapes.pop() == shared.work_shape_hw

    def test_get_pair_returns_valid_pair(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        masks = [_make_canonical_mask(m1), _make_canonical_mask(m2)]
        cfg = WorkingGridConfig(downsample_factor=2, padding_px=4)

        shared = prepare_shared_working_grid(
            masks, cfg, mask_ids=["a", "b"], keep_canon_masks=True
        )
        pair = shared.get_pair(0, 1)

        assert pair.work_shape_hw == shared.work_shape_hw
        assert pair.src_work_density.shape == shared.work_shape_hw
        assert pair.tgt_work_density.shape == shared.work_shape_hw
        assert pair.coord_frame_id == "canonical_grid"

    def test_non_canonical_raises(self):
        m = _make_test_mask()
        bad = _make_canonical_mask(m)
        bad.meta["coord_frame_id"] = "raw"
        with pytest.raises(ValueError, match="canonical-grid"):
            prepare_shared_working_grid([bad], WorkingGridConfig())

    def test_mismatched_um_per_px_raises(self):
        m1 = _make_test_mask()
        m2 = _make_test_mask()
        masks = [
            _make_canonical_mask(m1, um_per_px=10.0),
            _make_canonical_mask(m2, um_per_px=5.0),
        ]
        with pytest.raises(ValueError, match="um_per_px"):
            prepare_shared_working_grid(masks, WorkingGridConfig())

    def test_mismatched_canvas_shape_raises(self):
        m1 = _make_test_mask(canvas_h=64, canvas_w=128)
        m2 = np.zeros((32, 64), dtype=np.uint8)
        m2[5:20, 10:40] = 1
        masks = [
            _make_canonical_mask(m1, shape_yx=(64, 128)),
            _make_canonical_mask(m2, shape_yx=(32, 64)),
        ]
        with pytest.raises(ValueError, match="shape"):
            prepare_shared_working_grid(masks, WorkingGridConfig())

    def test_empty_mask_raises(self):
        m = np.zeros((64, 128), dtype=np.uint8)
        masks = [_make_canonical_mask(m, content_bbox_yx=None)]
        with pytest.raises(ValueError, match="empty"):
            prepare_shared_working_grid(
                masks, WorkingGridConfig(), mask_ids=["empty_one"]
            )

    def test_fallback_when_content_bbox_none(self):
        """Legacy results without content_bbox_yx should still work."""
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        masks = [
            _make_canonical_mask(m1, content_bbox_yx=None),
            _make_canonical_mask(m2, content_bbox_yx=None),
        ]
        cfg = WorkingGridConfig(downsample_factor=2, padding_px=4)
        shared = prepare_shared_working_grid(masks, cfg)
        assert shared.n_masks == 2
        assert shared.work_densities[0].shape == shared.work_shape_hw

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            prepare_shared_working_grid([], WorkingGridConfig())


class TestGeometryEquivalence:
    """High-value test: shared grid path produces same geometry as pairwise path."""

    def test_two_mask_geometry_matches_pairwise(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        src = _make_canonical_mask(m1)
        tgt = _make_canonical_mask(m2)

        cfg = WorkingGridConfig(downsample_factor=4, padding_px=8)

        # Old path
        pair_old = prepare_working_grid_pair(src, tgt, cfg)

        # New path
        shared = prepare_shared_working_grid(
            [src, tgt], cfg, keep_canon_masks=True
        )
        pair_new = shared.get_pair(0, 1)

        # Geometry must match exactly
        assert pair_old.pair_frame.work_shape_hw == pair_new.pair_frame.work_shape_hw
        assert pair_old.pair_frame.pair_crop_box_yx == pair_new.pair_frame.pair_crop_box_yx
        assert pair_old.pair_frame.downsample_factor == pair_new.pair_frame.downsample_factor
        assert pair_old.pair_frame.crop_pad_hw == pair_new.pair_frame.crop_pad_hw
        assert pair_old.pair_frame.canon_shape_hw == pair_new.pair_frame.canon_shape_hw

        # Densities must match
        np.testing.assert_array_equal(
            pair_old.src_work_density, pair_new.src_work_density
        )
        np.testing.assert_array_equal(
            pair_old.tgt_work_density, pair_new.tgt_work_density
        )

    def test_single_downsample_factor(self):
        """No downsampling case also matches."""
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        src = _make_canonical_mask(m1)
        tgt = _make_canonical_mask(m2)

        cfg = WorkingGridConfig(downsample_factor=1, padding_px=4)

        pair_old = prepare_working_grid_pair(src, tgt, cfg)
        shared = prepare_shared_working_grid(
            [src, tgt], cfg, keep_canon_masks=True
        )
        pair_new = shared.get_pair(0, 1)

        assert pair_old.pair_frame.work_shape_hw == pair_new.pair_frame.work_shape_hw
        np.testing.assert_array_equal(
            pair_old.src_work_density, pair_new.src_work_density
        )
        np.testing.assert_array_equal(
            pair_old.tgt_work_density, pair_new.tgt_work_density
        )


class TestMakeWorkingGridPairFromShared:
    def test_out_of_range_raises(self):
        m = _make_test_mask()
        masks = [_make_canonical_mask(m)]
        shared = prepare_shared_working_grid(masks, WorkingGridConfig())

        with pytest.raises(IndexError):
            make_working_grid_pair_from_shared(shared, 0, 1)
        with pytest.raises(IndexError):
            make_working_grid_pair_from_shared(shared, -1, 0)

    def test_canon_masks_none_gives_zero_placeholders(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        masks = [_make_canonical_mask(m1), _make_canonical_mask(m2)]
        shared = prepare_shared_working_grid(
            masks, WorkingGridConfig(), keep_canon_masks=False
        )
        pair = shared.get_pair(0, 1)
        # canon masks are zero placeholders
        assert pair.src_canon_mask.sum() == 0
        assert pair.tgt_canon_mask.sum() == 0
