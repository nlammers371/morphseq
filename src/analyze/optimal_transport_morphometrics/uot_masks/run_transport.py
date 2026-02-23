"""Run UOT on embryo mask pairs (coord -> working_grid -> solver -> lift).

Design:
- Geometry/canonicalization lives in `analyze.utils.coord`.
- Working-grid crop/pad/downsample + lifting lives in `analyze.utils.optimal_transport.working_grid`.
- Solver/backends are math-only (supports/weights in, coupling/cost out).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from analyze.utils.coord.grids.canonical import CanonicalGridConfig, to_canonical_grid_mask
from analyze.utils.optimal_transport import (
    MassMode,
    UOTConfig,
    UOTFramePair,
    UOTResultCanonical,
    UOTResultWork,
    WorkingGridConfig,
    lift_work_result_to_canonical,
    prepare_working_grid_pair,
    run_uot_on_working_grid,
)
from analyze.utils.optimal_transport.backends.base import UOTBackend


def _um_per_px_from_meta(meta: Optional[dict]) -> float:
    if not meta:
        return float("nan")
    # frame_mask_io stores um_per_pixel; some callers may use um_per_px.
    for k in ("um_per_px", "um_per_pixel"):
        if k in meta and meta[k] is not None:
            return float(meta[k])
    return float("nan")


def _yolk_from_meta(meta: Optional[dict]) -> Optional[np.ndarray]:
    if not meta:
        return None
    yolk = meta.get("yolk_mask", None)
    if yolk is None:
        return None
    return np.asarray(yolk)


def run_uot_pair(
    pair: UOTFramePair,
    *,
    canonical_cfg: Optional[CanonicalGridConfig] = None,
    working_cfg: Optional[WorkingGridConfig] = None,
    solver_cfg: Optional[UOTConfig] = None,
    backend: Optional[UOTBackend] = None,
    output_frame: Literal["work", "canonical"] = "canonical",
) -> UOTResultWork | UOTResultCanonical:
    """Run OT on a single pair.

    Inputs arrive as raw embryo masks; this orchestrator canonicalizes both masks,
    prepares a working grid pair, solves on the work grid, and optionally lifts
    outputs to canonical grid.
    """
    if solver_cfg is None:
        solver_cfg = UOTConfig()
    if canonical_cfg is None:
        canonical_cfg = CanonicalGridConfig()
    if working_cfg is None:
        working_cfg = WorkingGridConfig()

    if pair.src.embryo_mask is None or pair.tgt.embryo_mask is None:
        raise ValueError("UOTFramePair must contain embryo masks.")

    um_src = _um_per_px_from_meta(pair.src.meta)
    um_tgt = _um_per_px_from_meta(pair.tgt.meta)
    if not np.isfinite(um_src) or not np.isfinite(um_tgt):
        raise ValueError("Both frames must provide um_per_pixel in meta to canonicalize.")
    if not np.isclose(um_src, um_tgt, rtol=1e-6, atol=1e-6):
        raise ValueError(f"um_per_pixel mismatch: src={um_src} tgt={um_tgt}")

    src_can = to_canonical_grid_mask(
        np.asarray(pair.src.embryo_mask),
        um_per_px=float(um_src),
        yolk_mask=_yolk_from_meta(pair.src.meta),
        cfg=canonical_cfg,
    )
    tgt_can = to_canonical_grid_mask(
        np.asarray(pair.tgt.embryo_mask),
        um_per_px=float(um_tgt),
        yolk_mask=_yolk_from_meta(pair.tgt.meta),
        cfg=canonical_cfg,
    )

    # Bind density semantics (mass_mode) to the working-grid preparation.
    working_cfg = WorkingGridConfig(
        downsample_factor=working_cfg.downsample_factor,
        padding_px=working_cfg.padding_px,
        crop_policy=working_cfg.crop_policy,
        mass_mode=working_cfg.mass_mode if working_cfg.mass_mode is not None else MassMode.UNIFORM,
    )
    pair_work = prepare_working_grid_pair(src_can, tgt_can, working_cfg)
    result_work = run_uot_on_working_grid(pair_work, config=solver_cfg, backend=backend)

    if output_frame == "work":
        return result_work
    return lift_work_result_to_canonical(result_work, pair_work)


def run_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_index_src: int,
    frame_index_tgt: int,
    *,
    canonical_cfg: Optional[CanonicalGridConfig] = None,
    working_cfg: Optional[WorkingGridConfig] = None,
    solver_cfg: Optional[UOTConfig] = None,
    backend: Optional[UOTBackend] = None,
    data_root: Optional[Path] = None,
    output_frame: Literal["work", "canonical"] = "canonical",
) -> UOTResultWork | UOTResultCanonical:
    from .frame_mask_io import load_mask_pair_from_csv

    pair = load_mask_pair_from_csv(
        csv_path,
        embryo_id,
        frame_index_src,
        frame_index_tgt,
        data_root=data_root,
    )
    return run_uot_pair(
        pair,
        canonical_cfg=canonical_cfg,
        working_cfg=working_cfg,
        solver_cfg=solver_cfg,
        backend=backend,
        output_frame=output_frame,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UOT on a single mask pair from CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embryo-id", required=True)
    parser.add_argument("--frame-src", type=int, required=True)
    parser.add_argument("--frame-tgt", type=int, required=True)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--mass-mode", type=str, default="uniform")
    args = parser.parse_args()

    solver_cfg = UOTConfig()
    working_cfg = WorkingGridConfig(downsample_factor=args.downsample, mass_mode=MassMode(args.mass_mode))
    result = run_from_csv(
        args.csv,
        args.embryo_id,
        args.frame_src,
        args.frame_tgt,
        solver_cfg=solver_cfg,
        working_cfg=working_cfg,
        output_frame="canonical",
    )
    print("UOT cost:", float(result.cost))

