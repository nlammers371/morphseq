"""Working-grid preparation and lifting for OT.

This module is the ONLY owner of canonical<->work pragmatics:
- crop (pair union bbox)
- pad-to-divisible (bottom/right)
- downsample (sum pooling)
- lifting work-grid outputs back to canonical grid

Hard contract:
- Inputs must already be canonical-grid masks produced by `analyze.utils.coord`
  (meta labels: coord_frame_id="canonical_grid", coord_frame_version=1).
- No canonicalization, no registration, no rotation/scale lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import numpy as np

from analyze.utils.coord.types import CanonicalMaskResult
from analyze.utils.optimal_transport.config import PairFrameGeometry
from analyze.utils.optimal_transport.config import MassMode
from analyze.utils.optimal_transport.density_transforms import (
    enforce_min_mass,
    mask_to_density,
    mask_to_density_uniform,
)
from analyze.utils.optimal_transport.multiscale_sampling import downsample_density
from analyze.utils.optimal_transport.pair_frame import create_pair_frame_geometry
from analyze.utils.optimal_transport.transport_maps import (
    rasterize_mass_to_canonical,
    rasterize_scalar_to_canonical,
    rasterize_velocity_to_canonical,
)


class OutputFrame(str, Enum):
    WORK = "work"
    CANONICAL = "canonical"
    SRC_INPUT = "src_input"


@dataclass(frozen=True)
class WorkingGridConfig:
    downsample_factor: int = 4
    padding_px: int = 8
    crop_policy: str = "union"
    mass_mode: MassMode = MassMode.UNIFORM


@dataclass(frozen=True)
class WorkingGridPair:
    """Canonical<->work mapping plus prepared work-grid densities."""

    coord_frame_id: Literal["canonical_grid"]
    coord_frame_version: int

    canonical_um_per_px: float
    work_um_per_px: float

    pair_frame: PairFrameGeometry

    src_canon_mask: np.ndarray
    tgt_canon_mask: np.ndarray

    # Primary solver inputs: nonnegative float32 "mass" on work grid.
    src_work_density: np.ndarray
    tgt_work_density: np.ndarray

    meta: dict

    @property
    def work_shape_hw(self) -> tuple[int, int]:
        return self.pair_frame.work_shape_hw

    @property
    def canonical_shape_hw(self) -> tuple[int, int]:
        return self.pair_frame.canon_shape_hw

    def lift_work_mass_to_canonical(self, mass_work: np.ndarray) -> np.ndarray:
        """Lift a conserved work-grid mass map to canonical grid (mass-preserving)."""
        if mass_work.shape != self.work_shape_hw:
            raise ValueError(f"mass_work must have shape {self.work_shape_hw}; got {mass_work.shape}")
        return rasterize_mass_to_canonical(mass_work.astype(np.float32, copy=False), self.pair_frame)

    def lift_work_scalar_to_canonical(self, scalar_work: np.ndarray) -> np.ndarray:
        """Lift a non-conserved work-grid scalar map to canonical grid."""
        if scalar_work.shape != self.work_shape_hw:
            raise ValueError(f"scalar_work must have shape {self.work_shape_hw}; got {scalar_work.shape}")
        return rasterize_scalar_to_canonical(scalar_work.astype(np.float32, copy=False), self.pair_frame)

    def lift_work_velocity_to_canonical_px_per_step_yx(self, v_work_px_per_step_yx: np.ndarray) -> np.ndarray:
        """Lift work-grid velocity (work px/step) -> canonical velocity (canonical px/step).

        Contract:
        - velocity arrays are (H, W, 2) with component order (dy, dx).
        - solver outputs work px/step.
        - lifting scales magnitudes by downsample_factor to become canonical px/step.
        """
        if v_work_px_per_step_yx.shape != (*self.work_shape_hw, 2):
            raise ValueError(
                f"v_work_px_per_step_yx must have shape {(*self.work_shape_hw, 2)}; got {v_work_px_per_step_yx.shape}"
            )
        # transport_maps.rasterize_velocity_to_canonical(..., convert_to_um=False) scales by s.
        return rasterize_velocity_to_canonical(
            v_work_px_per_step_yx.astype(np.float32, copy=False),
            self.pair_frame,
            convert_to_um=False,
        )


def _require_canonical_mask(mask_res: CanonicalMaskResult, *, role: str) -> None:
    meta = mask_res.meta or {}
    if meta.get("coord_frame_id") != "canonical_grid" or meta.get("coord_frame_version") != 1:
        raise ValueError(
            f"{role} must be a canonical-grid mask (coord_frame_id='canonical_grid', version=1). "
            f"Got coord_frame_id={meta.get('coord_frame_id')!r} coord_frame_version={meta.get('coord_frame_version')!r}."
        )
    if mask_res.mask.ndim != 2:
        raise ValueError(f"{role} mask must be 2D; got shape {mask_res.mask.shape}")
    if tuple(mask_res.mask.shape) != tuple(mask_res.grid.shape_yx):
        raise ValueError(
            f"{role} mask shape {mask_res.mask.shape} does not match grid.shape_yx {mask_res.grid.shape_yx}."
        )


def prepare_working_grid_pair(
    src: CanonicalMaskResult,
    tgt: CanonicalMaskResult,
    cfg: Optional[WorkingGridConfig] = None,
) -> WorkingGridPair:
    """Prepare work-grid densities and geometry from canonical-grid masks."""
    if cfg is None:
        cfg = WorkingGridConfig()

    _require_canonical_mask(src, role="src")
    _require_canonical_mask(tgt, role="tgt")

    if src.mask.shape != tgt.mask.shape:
        raise ValueError(f"Canonical masks must have same shape; got src={src.mask.shape} tgt={tgt.mask.shape}.")
    if float(src.grid.um_per_px) != float(tgt.grid.um_per_px):
        raise ValueError(f"Canonical um_per_px must match; got src={src.grid.um_per_px} tgt={tgt.grid.um_per_px}.")

    if cfg.downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1; got {cfg.downsample_factor}")
    if cfg.padding_px < 0:
        raise ValueError(f"padding_px must be >= 0; got {cfg.padding_px}")

    canon_mask_src = (np.asarray(src.mask) > 0).astype(np.uint8)
    canon_mask_tgt = (np.asarray(tgt.mask) > 0).astype(np.uint8)

    if int(canon_mask_src.sum()) == 0 or int(canon_mask_tgt.sum()) == 0:
        raise ValueError("Empty masks are not supported by working-grid preparation (cannot compute pair bbox).")

    canonical_um_per_px = float(src.grid.um_per_px)
    pair_frame = create_pair_frame_geometry(
        canon_mask_src,
        canon_mask_tgt,
        downsample_factor=int(cfg.downsample_factor),
        padding_px=int(cfg.padding_px),
        px_size_um=canonical_um_per_px,
        crop_policy=str(cfg.crop_policy),
    )

    bbox = pair_frame.pair_crop_box_yx
    src_crop = canon_mask_src[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
    tgt_crop = canon_mask_tgt[bbox.y0:bbox.y1, bbox.x0:bbox.x1]

    pad_h, pad_w = pair_frame.crop_pad_hw
    src_pad = np.pad(src_crop, ((0, pad_h), (0, pad_w)), mode="constant")
    tgt_pad = np.pad(tgt_crop, ((0, pad_h), (0, pad_w)), mode="constant")

    src_density = mask_to_density(src_pad, cfg.mass_mode)
    tgt_density = mask_to_density(tgt_pad, cfg.mass_mode)

    if cfg.mass_mode.name == "DISTANCE_TRANSFORM":
        src_density = enforce_min_mass(src_density, fallback=mask_to_density_uniform(src_pad))
        tgt_density = enforce_min_mass(tgt_density, fallback=mask_to_density_uniform(tgt_pad))

    if pair_frame.downsample_factor > 1:
        src_density = downsample_density(src_density, pair_frame.downsample_factor)
        tgt_density = downsample_density(tgt_density, pair_frame.downsample_factor)

    assert src_density.shape == pair_frame.work_shape_hw
    assert tgt_density.shape == pair_frame.work_shape_hw

    work_um_per_px = canonical_um_per_px * float(pair_frame.downsample_factor)

    meta = {
        "coord_frame_id": "canonical_grid",
        "coord_frame_version": 1,
        "coord_convention": "yx",
        "canonical_um_per_px": canonical_um_per_px,
        "work_um_per_px": work_um_per_px,
        "work_grid": {
            "downsample_factor": pair_frame.downsample_factor,
            "padding_px": cfg.padding_px,
            "crop_policy": cfg.crop_policy,
            "bbox_y0y1x0x1": (bbox.y0, bbox.y1, bbox.x0, bbox.x1),
            "crop_pad_hw": pair_frame.crop_pad_hw,
            "work_shape_hw": pair_frame.work_shape_hw,
        },
        # Store mass sums before any sampling/backends.
        "src_mass_px": float(src_density.sum()),
        "tgt_mass_px": float(tgt_density.sum()),
    }

    return WorkingGridPair(
        coord_frame_id="canonical_grid",
        coord_frame_version=1,
        canonical_um_per_px=canonical_um_per_px,
        work_um_per_px=work_um_per_px,
        pair_frame=pair_frame,
        src_canon_mask=canon_mask_src,
        tgt_canon_mask=canon_mask_tgt,
        src_work_density=src_density,
        tgt_work_density=tgt_density,
        meta=meta,
    )
