"""Tests for BoxYX geometry type."""

import numpy as np
import pytest

from analyze.utils.coord.types import BoxYX


class TestBoxYXBasics:
    def test_h_w_area(self):
        box = BoxYX(10, 30, 5, 25)
        assert box.h == 20
        assert box.w == 20
        assert box.area == 400

    def test_to_slices(self):
        box = BoxYX(2, 5, 3, 8)
        arr = np.arange(100).reshape(10, 10)
        sy, sx = box.to_slices()
        region = arr[sy, sx]
        assert region.shape == (3, 5)
        assert region[0, 0] == arr[2, 3]
        assert region[-1, -1] == arr[4, 7]


class TestContains:
    def test_self_contains_self(self):
        box = BoxYX(5, 15, 10, 20)
        assert box.contains(box)

    def test_larger_contains_smaller(self):
        big = BoxYX(0, 100, 0, 100)
        small = BoxYX(10, 20, 10, 20)
        assert big.contains(small)
        assert not small.contains(big)

    def test_partial_overlap_not_contains(self):
        a = BoxYX(0, 10, 0, 10)
        b = BoxYX(5, 15, 5, 15)
        assert not a.contains(b)
        assert not b.contains(a)


class TestUnion:
    def test_union_is_commutative(self):
        a = BoxYX(5, 15, 10, 20)
        b = BoxYX(0, 10, 15, 30)
        assert a.union(b) == b.union(a)

    def test_union_contains_both(self):
        a = BoxYX(5, 15, 10, 20)
        b = BoxYX(0, 10, 15, 30)
        u = a.union(b)
        assert u.contains(a)
        assert u.contains(b)

    def test_union_with_self(self):
        a = BoxYX(5, 15, 10, 20)
        assert a.union(a) == a


class TestPadClamp:
    def test_pad_expands(self):
        box = BoxYX(10, 20, 10, 20)
        padded = box.pad(5, 3)
        assert padded == BoxYX(5, 25, 7, 23)

    def test_clamp_stays_in_canvas(self):
        box = BoxYX(-5, 30, -3, 50)
        clamped = box.clamp(25, 40)
        assert clamped == BoxYX(0, 25, 0, 40)

    def test_pad_then_clamp(self):
        box = BoxYX(2, 10, 3, 15)
        result = box.pad(5, 5).clamp(20, 20)
        assert result.y0 >= 0
        assert result.x0 >= 0
        assert result.y1 <= 20
        assert result.x1 <= 20
        # Original box should be contained
        assert result.contains(box)


class TestValidate:
    def test_valid_box(self):
        BoxYX(0, 10, 0, 10).validate(20, 20)  # should not raise

    def test_invalid_y(self):
        with pytest.raises(ValueError, match="y-range"):
            BoxYX(0, 25, 0, 10).validate(20, 20)

    def test_invalid_x(self):
        with pytest.raises(ValueError, match="x-range"):
            BoxYX(0, 10, -1, 10).validate(20, 20)

    def test_empty_box_valid(self):
        BoxYX(5, 5, 5, 5).validate(10, 10)  # zero-area is valid


class TestFromMask:
    def test_empty_mask_returns_none(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert BoxYX.from_mask(mask) is None

    def test_nonempty_mask_tight_bbox(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:10, 8:15] = 1
        bbox = BoxYX.from_mask(mask)
        assert bbox is not None
        assert bbox == BoxYX(5, 10, 8, 15)

    def test_single_pixel(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3, 7] = 1
        bbox = BoxYX.from_mask(mask)
        assert bbox == BoxYX(3, 4, 7, 8)
        assert bbox.h == 1
        assert bbox.w == 1
        assert bbox.area == 1

    def test_from_mask_selects_all_nonzero(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:10, 8:15] = 1
        mask[12, 20] = 1
        bbox = BoxYX.from_mask(mask)
        # All nonzero pixels must be inside the bbox
        ys, xs = np.where(mask > 0)
        assert np.all(ys >= bbox.y0)
        assert np.all(ys < bbox.y1)
        assert np.all(xs >= bbox.x0)
        assert np.all(xs < bbox.x1)
