from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze.utils.coord.transforms import TransformChain
from analyze.utils.coord.types import CanonicalGrid, CanonicalMaskResult
from analyze.utils.optimal_transport import (
    BatchItem,
    UOTConfig,
    WorkingGridConfig,
    prepare_working_grid_pair,
    run_uot_on_working_grid,
    solve_working_grid_batch,
)
from analyze.utils.optimal_transport.backends.base import BackendResult, UOTBackend


class _FakeBackend(UOTBackend):
    def __init__(self):
        self.solve_calls = 0
        self.solve_batch_calls = 0
        self.solve_batch_sizes = []

    def solve(self, src, tgt, config):
        self.solve_calls += 1
        # Simple deterministic coupling: distribute each src weight proportionally to tgt weights.
        b = tgt.weights.astype(np.float64)
        b_sum = float(b.sum()) if float(b.sum()) > 0 else 1.0
        P = (src.weights.astype(np.float64)[:, None] * b[None, :] / b_sum).astype(np.float64)
        return BackendResult(coupling=P, cost=float(P.sum()), diagnostics={"m_src": float(src.weights.sum()), "m_tgt": float(tgt.weights.sum())})

    def solve_batch(self, problems, config):
        self.solve_batch_calls += 1
        self.solve_batch_sizes.append(len(problems))
        return [self.solve(src, tgt, config) for (src, tgt) in problems]


def _canon_mask(mask: np.ndarray, *, um_per_px: float = 10.0) -> CanonicalMaskResult:
    h, w = mask.shape
    grid = CanonicalGrid(
        um_per_px=float(um_per_px),
        shape_yx=(h, w),
        anchor_mode="yolk_anchor",
        anchor_yx=(0.5, 0.5),
    )
    meta = {"coord_frame_id": "canonical_grid", "coord_frame_version": 1, "coord_convention": "yx"}
    return CanonicalMaskResult(
        mask=mask,
        grid=grid,
        transform_chain=TransformChain.identity(shape_yx=(h, w), interp="nearest"),
        meta=meta,
    )


def test_batch_runner_buckets_by_work_shape_and_matches_single_solve():
    src1 = np.zeros((12, 12), dtype=np.uint8)
    tgt1 = np.zeros_like(src1)
    src1[2:6, 2:6] = 1
    tgt1[3:7, 3:7] = 1

    src2 = np.zeros((12, 12), dtype=np.uint8)
    tgt2 = np.zeros_like(src2)
    src2[1:5, 1:5] = 1
    tgt2[2:6, 2:6] = 1

    # Different canonical shape -> different work shape bucket
    src3 = np.zeros((10, 14), dtype=np.uint8)
    tgt3 = np.zeros_like(src3)
    src3[2:6, 4:8] = 1
    tgt3[2:6, 5:9] = 1

    wg_cfg = WorkingGridConfig(downsample_factor=2, padding_px=0)

    p1 = prepare_working_grid_pair(_canon_mask(src1), _canon_mask(tgt1), wg_cfg)
    p2 = prepare_working_grid_pair(_canon_mask(src2), _canon_mask(tgt2), wg_cfg)
    p3 = prepare_working_grid_pair(_canon_mask(src3), _canon_mask(tgt3), wg_cfg)

    items = [BatchItem(pair=p1, item_id="a"), BatchItem(pair=p2, item_id="b"), BatchItem(pair=p3, item_id="c")]

    cfg = UOTConfig(max_support_points=1000, store_coupling=True)

    backend_batch = _FakeBackend()
    batch_results = solve_working_grid_batch(items, config=cfg, backend=backend_batch)

    # Two buckets: (p1,p2) share shape, p3 differs
    assert backend_batch.solve_batch_calls == 2
    assert sorted(backend_batch.solve_batch_sizes) == [1, 2]

    backend_single = _FakeBackend()
    single_results = [
        run_uot_on_working_grid(p1, config=cfg, backend=backend_single),
        run_uot_on_working_grid(p2, config=cfg, backend=backend_single),
        run_uot_on_working_grid(p3, config=cfg, backend=backend_single),
    ]

    assert len(batch_results) == len(single_results) == 3
    for br, sr in zip(batch_results, single_results):
        assert np.isclose(float(br.cost), float(sr.cost))
        np.testing.assert_allclose(br.mass_created_work.sum(), sr.mass_created_work.sum(), atol=1e-6)
        np.testing.assert_allclose(br.mass_destroyed_work.sum(), sr.mass_destroyed_work.sum(), atol=1e-6)

