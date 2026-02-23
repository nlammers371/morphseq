"""High-level UOT solve entrypoints (work-grid in/out) + lifting helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .backends.base import UOTBackend
from .config import UOTConfig
from .metrics import summarize_metrics
from .multiscale_sampling import build_support
from .results import UOTResultCanonical, UOTResultWork
from .transport_maps import compute_cost_maps, compute_transport_maps
from .working_grid import WorkingGridPair


def run_uot_on_working_grid(
    pair: WorkingGridPair,
    *,
    config: Optional[UOTConfig] = None,
    backend: Optional[UOTBackend] = None,
) -> UOTResultWork:
    """Solve UOT for a prepared working-grid pair.

    Contract:
    - `pair.src_work_density` and `pair.tgt_work_density` are nonnegative float arrays on the work grid.
    - The solver/backend sees only supports/weights; it is blind to canonical coordinates.
    """
    if config is None:
        config = UOTConfig()
    if backend is None:
        # Avoid importing POT (which may pull in torch) unless needed.
        try:
            from .backends.ott_backend import OTTBackend  # type: ignore

            backend = OTTBackend()
        except Exception:
            from .backends.pot_backend import POTBackend

            backend = POTBackend()

    src_support, src_meta = build_support(
        pair.src_work_density,
        max_points=config.max_support_points,
        sampling_mode=config.sampling_mode,
        sampling_strategy=config.sampling_strategy,
        random_seed=config.random_seed,
    )
    tgt_support, tgt_meta = build_support(
        pair.tgt_work_density,
        max_points=config.max_support_points,
        sampling_mode=config.sampling_mode,
        sampling_strategy=config.sampling_strategy,
        random_seed=config.random_seed,
    )

    backend_result = backend.solve(src_support, tgt_support, config)

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

    # Backend diagnostics may include m_src/m_tgt, but that is backend-dependent.
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
    }

    meta = {
        "coord_convention": "yx",
        "output_frame": "work",
        "velocity_units": "work_px_per_step",
        "work_um_per_px": pair.work_um_per_px,
        "canonical_um_per_px": pair.canonical_um_per_px,
        "pair_frame": {
            "downsample_factor": pair.pair_frame.downsample_factor,
            "bbox_y0y1x0x1": (
                pair.pair_frame.pair_crop_box_yx.y0,
                pair.pair_frame.pair_crop_box_yx.y1,
                pair.pair_frame.pair_crop_box_yx.x0,
                pair.pair_frame.pair_crop_box_yx.x1,
            ),
            "crop_pad_hw": pair.pair_frame.crop_pad_hw,
        },
    }

    return UOTResultWork(
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


def lift_work_result_to_canonical(
    result_work: UOTResultWork,
    pair: WorkingGridPair,
) -> UOTResultCanonical:
    """Lift a work-grid result into canonical-grid outputs (no mixed-frame fields)."""
    mass_created_c = pair.lift_work_mass_to_canonical(result_work.mass_created_work)
    mass_destroyed_c = pair.lift_work_mass_to_canonical(result_work.mass_destroyed_work)
    velocity_c = pair.lift_work_velocity_to_canonical_px_per_step_yx(result_work.velocity_work_px_per_step_yx)

    cost_src_c = None
    cost_tgt_c = None
    if result_work.cost_src_work is not None and result_work.cost_tgt_work is not None:
        cost_src_c = pair.lift_work_scalar_to_canonical(result_work.cost_src_work)
        cost_tgt_c = pair.lift_work_scalar_to_canonical(result_work.cost_tgt_work)

    meta = dict(result_work.meta or {})
    meta.update(
        {
            "output_frame": "canonical",
            "velocity_units": "canonical_px_per_step",
        }
    )

    return UOTResultCanonical(
        cost=float(result_work.cost),
        mass_created_canon=mass_created_c,
        mass_destroyed_canon=mass_destroyed_c,
        velocity_canon_px_per_step_yx=velocity_c,
        cost_src_canon=cost_src_c,
        cost_tgt_canon=cost_tgt_c,
        diagnostics=result_work.diagnostics,
        canonical_shape_hw=pair.canonical_shape_hw,
        canonical_um_per_px=pair.canonical_um_per_px,
        pair_frame=pair.pair_frame,
        meta=meta,
    )
