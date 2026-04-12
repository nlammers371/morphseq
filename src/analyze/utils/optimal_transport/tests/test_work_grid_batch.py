"""Tests for WorkGridBatch, pack_pairs, pack_star, and batch solvers."""

import numpy as np
import pytest

from analyze.utils.coord.types import BoxYX, CanonicalGrid, CanonicalMaskResult
from analyze.utils.coord.transforms import TransformChain
from analyze.utils.optimal_transport.backends.base import BackendResult, UOTBackend
from analyze.utils.optimal_transport.work_grid_batch import (
    WorkGridBatch,
    prepare_work_grid_batch,
    pack_pairs,
    pack_star,
    PairPack,
    StarPack,
    CropPolicy,
    PerPairUnionCrop,
    GlobalUnionCrop,
    PerRefUnionCrop,
)


class _DeterministicTestBackend(UOTBackend):
    """Small, dependency-free backend for unit tests.

    Avoids importing POT/torch and is deterministic across runs.
    """

    def solve(self, src, tgt, config):  # type: ignore[override]
        coords_src = src.coords_yx.astype(np.float64) * float(config.coord_scale)
        coords_tgt = tgt.coords_yx.astype(np.float64) * float(config.coord_scale)
        weights_src = src.weights.astype(np.float64)
        weights_tgt = tgt.weights.astype(np.float64)

        m_src = float(weights_src.sum())
        m_tgt = float(weights_tgt.sum())
        if m_src <= 0 or m_tgt <= 0:
            raise ValueError("Source/target mass must be positive for UOT solve.")

        if config.metric == "sqeuclidean":
            diff = coords_src[:, None, :] - coords_tgt[None, :, :]
            cost = (diff ** 2).sum(axis=2)
        elif config.metric == "euclidean":
            diff = coords_src[:, None, :] - coords_tgt[None, :, :]
            cost = np.sqrt((diff ** 2).sum(axis=2))
        else:
            raise ValueError(f"Unsupported metric: {config.metric}")

        # Simple coupling with exact src marginals (mu_hat == weights_src).
        coupling = np.outer(weights_src, weights_tgt) / m_tgt
        weighted_cost = coupling * cost
        cost_value = float(weighted_cost.sum())

        diagnostics = {
            "m_src": m_src,
            "m_tgt": m_tgt,
            "reg": float(config.epsilon),
            "reg_m": float(config.marginal_relaxation),
            "coord_scale": float(config.coord_scale),
            "backend": "deterministic_test",
        }

        return BackendResult(
            coupling=coupling.astype(np.float64) if bool(config.store_coupling) else None,
            cost=cost_value,
            diagnostics=diagnostics,
            cost_per_src=weighted_cost.sum(axis=1).astype(np.float64),
            cost_per_tgt=weighted_cost.sum(axis=0).astype(np.float64),
        )


def _make_canonical_mask(
    mask: np.ndarray,
    um_per_px: float = 10.0,
    shape_yx: tuple[int, int] | None = None,
) -> CanonicalMaskResult:
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
    return CanonicalMaskResult(
        mask=mask.astype(np.uint8),
        grid=grid,
        transform_chain=chain,
        meta=meta,
        content_bbox_yx=BoxYX.from_mask(mask),
    )


def _make_test_mask(
    canvas_h: int = 64,
    canvas_w: int = 128,
    y0: int = 10,
    y1: int = 50,
    x0: int = 20,
    x1: int = 100,
) -> np.ndarray:
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class TestPrepareWorkGridBatch:
    def test_basic_from_canonical_masks(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        masks = [_make_canonical_mask(m1), _make_canonical_mask(m2)]
        batch = prepare_work_grid_batch(masks, downsample_factor=4, mask_ids=["a", "b"])

        assert batch.n_masks == 2
        assert batch.densities_full.shape == (2, 16, 32)
        assert batch.work_shape_hw_full == (16, 32)
        assert batch.mask_ids == ["a", "b"]

    def test_from_raw_ndarray(self):
        raw = np.zeros((3, 64, 128), dtype=np.uint8)
        raw[0, 10:40, 20:80] = 1
        raw[1, 15:45, 25:85] = 1
        raw[2, 20:50, 30:90] = 1

        batch = prepare_work_grid_batch(
            raw,
            downsample_factor=4,
            canonical_um_per_px=10.0,
            mask_ids=["a", "b", "c"],
        )
        assert batch.n_masks == 3
        assert batch.densities_full.shape == (3, 16, 32)
        assert batch.canonical_um_per_px == 10.0

    def test_same_mask_same_density_regardless_of_batch(self):
        """THE test: same mask → identical density regardless of batch composition."""
        m_shared = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m_other1 = _make_test_mask(y0=5, y1=30, x0=10, x1=60)
        m_other2 = _make_test_mask(y0=20, y1=50, x0=40, x1=100)

        batch1 = prepare_work_grid_batch(
            [_make_canonical_mask(m_shared), _make_canonical_mask(m_other1)],
            downsample_factor=4,
            mask_ids=["shared", "other1"],
        )
        batch2 = prepare_work_grid_batch(
            [_make_canonical_mask(m_shared), _make_canonical_mask(m_other2)],
            downsample_factor=4,
            mask_ids=["shared", "other2"],
        )

        np.testing.assert_array_equal(
            batch1.densities_full[0], batch2.densities_full[0]
        )

    def test_work_grid_id_deterministic_and_batch_independent(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)

        b1 = prepare_work_grid_batch(
            [_make_canonical_mask(m1)], downsample_factor=4, mask_ids=["a"]
        )
        b2 = prepare_work_grid_batch(
            [_make_canonical_mask(m2)], downsample_factor=4, mask_ids=["b"]
        )

        # Same canon_shape, ds, um_per_px → same work_grid_id
        assert b1.meta["work_grid_id"] == b2.meta["work_grid_id"]

    def test_support_bboxes(self):
        m = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        batch = prepare_work_grid_batch(
            [_make_canonical_mask(m)], downsample_factor=4, mask_ids=["a"]
        )
        bbox = batch.support_bboxes_work[0]
        assert isinstance(bbox, BoxYX)
        assert bbox.area > 0

    def test_canvas_not_divisible_raises(self):
        raw = np.zeros((1, 65, 129), dtype=np.uint8)
        raw[0, 10:40, 20:80] = 1
        with pytest.raises(ValueError, match="divisible"):
            prepare_work_grid_batch(
                raw, downsample_factor=4, canonical_um_per_px=10.0
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            prepare_work_grid_batch([], downsample_factor=4)

    def test_index_of(self):
        m = _make_test_mask()
        batch = prepare_work_grid_batch(
            [_make_canonical_mask(m)], downsample_factor=4, mask_ids=["alpha"]
        )
        assert batch.index_of("alpha") == 0
        with pytest.raises(KeyError):
            batch.index_of("nonexistent")

    def test_contains(self):
        m = _make_test_mask()
        batch = prepare_work_grid_batch(
            [_make_canonical_mask(m)], downsample_factor=4, mask_ids=["alpha"]
        )
        assert "alpha" in batch
        assert "beta" not in batch


class TestPackPairs:
    def _build_batch(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        m3 = _make_test_mask(y0=12, y1=45, x0=25, x1=90)
        masks = [_make_canonical_mask(m) for m in [m1, m2, m3]]
        return prepare_work_grid_batch(
            masks, downsample_factor=4, mask_ids=["ref", "tgt_0", "tgt_1"]
        )

    def test_indices_correct(self):
        batch = self._build_batch()
        pairs = [("ref", "tgt_0"), ("ref", "tgt_1")]
        pp = pack_pairs(batch, pairs)

        assert len(pp.src_indices) == 2
        assert len(pp.tgt_indices) == 2
        np.testing.assert_array_equal(pp.src_indices, [0, 0])
        np.testing.assert_array_equal(pp.tgt_indices, [1, 2])
        assert pp.crop_boxes_work is None

    def test_per_pair_crop(self):
        batch = self._build_batch()
        pairs = [("ref", "tgt_0")]
        pp = pack_pairs(batch, pairs, crop_policy=PerPairUnionCrop(margin_cells=1))
        assert pp.crop_boxes_work is not None
        assert len(pp.crop_boxes_work) == 1
        assert isinstance(pp.crop_boxes_work[0], BoxYX)

    def test_batch_reference_not_copy(self):
        batch = self._build_batch()
        pp = pack_pairs(batch, [("ref", "tgt_0")])
        assert pp.batch is batch

    def test_missing_id_raises(self):
        batch = self._build_batch()
        with pytest.raises(KeyError):
            pack_pairs(batch, [("ref", "nonexistent")])


class TestPackStar:
    def _build_batch(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        m3 = _make_test_mask(y0=12, y1=45, x0=25, x1=90)
        masks = [_make_canonical_mask(m) for m in [m1, m2, m3]]
        return prepare_work_grid_batch(
            masks, downsample_factor=4, mask_ids=["ref", "src_0", "src_1"]
        )

    def test_basic_star(self):
        batch = self._build_batch()
        sp = pack_star(batch, ref_ids=["ref"], src_ids=["src_0", "src_1"])

        np.testing.assert_array_equal(sp.ref_indices, [0])
        np.testing.assert_array_equal(sp.src_indices, [1, 2])
        assert sp.ref_densities.shape == (1, 16, 32)
        assert sp.crop_box_work is None

    def test_ref_densities_is_copy(self):
        batch = self._build_batch()
        sp = pack_star(batch, ref_ids=["ref"], src_ids=["src_0"])
        np.testing.assert_array_equal(sp.ref_densities[0], batch.densities_full[0])
        # It's a copy, not a view
        assert not np.shares_memory(sp.ref_densities, batch.densities_full)

    def test_global_crop(self):
        batch = self._build_batch()
        sp = pack_star(
            batch,
            ref_ids=["ref"],
            src_ids=["src_0", "src_1"],
            crop_policy=GlobalUnionCrop(margin_cells=1),
        )
        assert sp.crop_box_work is not None
        assert isinstance(sp.crop_box_work, BoxYX)

    def test_chunk_size_stored(self):
        batch = self._build_batch()
        sp = pack_star(
            batch, ref_ids=["ref"], src_ids=["src_0"], chunk_size=16
        )
        assert sp.chunk_size == 16


class TestWorkSpaceCrop:
    """Test that work-space crop + uncrop is lossless."""

    def test_crop_uncrop_lossless(self):
        m = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        batch = prepare_work_grid_batch(
            [_make_canonical_mask(m)], downsample_factor=4, mask_ids=["a"]
        )
        full_density = batch.densities_full[0]
        bbox = batch.support_bboxes_work[0]

        # Crop
        cropped = full_density[bbox.to_slices()]

        # Uncrop
        uncropped = np.zeros_like(full_density)
        uncropped[bbox.to_slices()] = cropped

        # Where the support is, values must match
        np.testing.assert_array_equal(
            uncropped[bbox.to_slices()], full_density[bbox.to_slices()]
        )

    def test_global_crop_covers_all_support(self):
        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=50, x0=30, x1=100)
        batch = prepare_work_grid_batch(
            [_make_canonical_mask(m1), _make_canonical_mask(m2)],
            downsample_factor=4,
            mask_ids=["a", "b"],
        )
        sp = pack_star(
            batch,
            ref_ids=["a"],
            src_ids=["b"],
            crop_policy=GlobalUnionCrop(margin_cells=1),
        )
        crop = sp.crop_box_work
        # Crop should contain all individual support bboxes
        for bbox in batch.support_bboxes_work:
            if bbox.area > 0:
                assert crop.contains(bbox), (
                    f"Global crop {crop} does not contain support bbox {bbox}"
                )


class TestSolvePairs:
    """Basic solve_pairs test — compare to itself (not to SharedWorkingGrid)."""

    def test_solve_pairs_runs(self):
        """Smoke test: solve_pairs returns results with correct length."""
        from analyze.utils.optimal_transport.solve_batch import solve_pairs
        from analyze.utils.optimal_transport.config import UOTConfig

        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=45, x0=25, x1=85)
        batch = prepare_work_grid_batch(
            [_make_canonical_mask(m1), _make_canonical_mask(m2)],
            downsample_factor=4,
            mask_ids=["ref", "tgt"],
        )
        pp = pack_pairs(batch, [("ref", "tgt")])

        uot_cfg = UOTConfig(epsilon=1e-1)
        backend = _DeterministicTestBackend()
        results = solve_pairs(pp, uot_cfg, backend)

        assert len(results) == 1
        assert results[0].cost > 0
        assert results[0].mass_created_work.shape[0] > 0

    def test_solve_star_chunked_matches_non_chunked(self):
        """Chunked and non-chunked star solve produce identical results."""
        from analyze.utils.optimal_transport.solve_batch import solve_star
        from analyze.utils.optimal_transport.config import UOTConfig

        m1 = _make_test_mask(y0=10, y1=40, x0=20, x1=80)
        m2 = _make_test_mask(y0=15, y1=45, x0=25, x1=85)
        m3 = _make_test_mask(y0=12, y1=42, x0=22, x1=82)
        batch = prepare_work_grid_batch(
            [_make_canonical_mask(m) for m in [m1, m2, m3]],
            downsample_factor=4,
            mask_ids=["ref", "src_0", "src_1"],
        )

        uot_cfg = UOTConfig(epsilon=1e-1)
        backend = _DeterministicTestBackend()

        sp1 = pack_star(batch, ref_ids=["ref"], src_ids=["src_0", "src_1"], chunk_size=1)
        sp2 = pack_star(batch, ref_ids=["ref"], src_ids=["src_0", "src_1"], chunk_size=100)

        r1 = solve_star(sp1, uot_cfg, backend)
        r2 = solve_star(sp2, uot_cfg, backend)

        assert set(r1.keys()) == set(r2.keys())
        for key in r1:
            np.testing.assert_allclose(r1[key].cost, r2[key].cost, rtol=1e-6)
