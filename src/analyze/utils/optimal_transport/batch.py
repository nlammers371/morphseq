"""Batch runner for work-grid OT solves.

Ownership:
- batching strategy (bucketing by work shape) lives here, above backends.
- backends may optionally provide `solve_batch`; if not, we loop `solve`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from .backends.base import UOTBackend
from .config import UOTConfig
from .metrics import summarize_metrics
from .multiscale_sampling import build_support
from .results import UOTResultWork
from .transport_maps import compute_cost_maps, compute_transport_maps
from .working_grid import WorkingGridPair


@dataclass(frozen=True)
class BatchItem:
    pair: WorkingGridPair
    item_id: str = ""


def solve_working_grid_batch(
    items: Iterable[BatchItem],
    *,
    config: UOTConfig,
    backend: UOTBackend,
) -> List[UOTResultWork]:
    """Solve a batch of prepared work-grid pairs, bucketing by work shape."""
    buckets: dict[tuple[int, int], list[BatchItem]] = defaultdict(list)
    for it in items:
        buckets[it.pair.work_shape_hw].append(it)

    out: list[UOTResultWork] = []

    for work_shape_hw, bucket in buckets.items():
        # Pre-build supports so backends remain math-only.
        problems = []
        supports = []
        for it in bucket:
            src_support, src_meta = build_support(
                it.pair.src_work_density,
                max_points=config.max_support_points,
                sampling_mode=config.sampling_mode,
                sampling_strategy=config.sampling_strategy,
                random_seed=config.random_seed,
            )
            tgt_support, tgt_meta = build_support(
                it.pair.tgt_work_density,
                max_points=config.max_support_points,
                sampling_mode=config.sampling_mode,
                sampling_strategy=config.sampling_strategy,
                random_seed=config.random_seed,
            )
            problems.append((src_support, tgt_support))
            supports.append((src_support, tgt_support, src_meta, tgt_meta))

        # Backend batch capability is optional.
        if hasattr(backend, "solve_batch"):
            backend_results = backend.solve_batch(problems, config)  # type: ignore[attr-defined]
        else:
            backend_results = [backend.solve(src, tgt, config) for (src, tgt) in problems]

        for it, (src_support, tgt_support, src_meta, tgt_meta), backend_result in zip(bucket, supports, backend_results):
            pair = it.pair
            mass_created_work, mass_destroyed_work, velocity_work = compute_transport_maps(
                backend_result.coupling,
                src_support.coords_yx,
                tgt_support.coords_yx,
                src_support.weights,
                tgt_support.weights,
                pair.work_shape_hw,
            )

            cost_src_work = None
            cost_tgt_work = None
            if backend_result.cost_per_src is not None and backend_result.cost_per_tgt is not None:
                cost_src_work, cost_tgt_work = compute_cost_maps(
                    backend_result.cost_per_src,
                    backend_result.cost_per_tgt,
                    src_support.coords_yx,
                    tgt_support.coords_yx,
                    pair.work_shape_hw,
                )

            m_src = backend_result.diagnostics.get("m_src") if backend_result.diagnostics else None
            m_tgt = backend_result.diagnostics.get("m_tgt") if backend_result.diagnostics else None
            metrics = summarize_metrics(
                backend_result.cost,
                backend_result.coupling,
                mass_created_work,
                mass_destroyed_work,
                config.metric,
                m_src=m_src,
                m_tgt=m_tgt,
                pair_frame=pair.pair_frame,
                coord_scale=float(config.coord_scale),
            )

            diagnostics = {
                "metrics": metrics,
                "backend": backend_result.diagnostics,
                "support_src": src_meta,
                "support_tgt": tgt_meta,
                "item_id": it.item_id,
            }

            meta = {
                "coord_convention": "yx",
                "output_frame": "work",
                "velocity_units": "work_px_per_step",
                "work_um_per_px": pair.work_um_per_px,
                "canonical_um_per_px": pair.canonical_um_per_px,
                "item_id": it.item_id,
            }

            out.append(
                UOTResultWork(
                    cost=float(backend_result.cost),
                    coupling=backend_result.coupling if bool(config.store_coupling) else None,
                    mass_created_work=mass_created_work,
                    mass_destroyed_work=mass_destroyed_work,
                    velocity_work_px_per_step_yx=velocity_work,
                    support_src_yx=src_support.coords_yx,
                    support_tgt_yx=tgt_support.coords_yx,
                    weights_src=src_support.weights,
                    weights_tgt=tgt_support.weights,
                    cost_src_support=backend_result.cost_per_src,
                    cost_tgt_support=backend_result.cost_per_tgt,
                    cost_src_work=cost_src_work,
                    cost_tgt_work=cost_tgt_work,
                    diagnostics=diagnostics,
                    work_shape_hw=pair.work_shape_hw,
                    work_um_per_px=pair.work_um_per_px,
                    pair_frame=pair.pair_frame,
                    meta=meta,
                )
            )

    return out
