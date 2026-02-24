"""Shared working grid for batch OT over N canonical masks.

Workflow:
    results = to_canonical_grid_mask_batch(raw_masks, cfg)
    shared  = prepare_shared_working_grid(results, working_cfg)
    pair    = shared.get_pair(0, 1)
    # or: pair = make_working_grid_pair_from_shared(shared, 0, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Literal, Optional

import numpy as np

from analyze.utils.coord.types import BoxYX, CanonicalMaskResult
from analyze.utils.optimal_transport.config import PairFrameGeometry
from analyze.utils.optimal_transport.density_transforms import (
    enforce_min_mass,
    mask_to_density,
    mask_to_density_uniform,
)
from analyze.utils.optimal_transport.multiscale_sampling import downsample_density
from analyze.utils.optimal_transport.working_grid import (
    WorkingGridConfig,
    WorkingGridPair,
    _require_canonical_mask,
)


@dataclass(frozen=True)
class SharedGridGeometry:
    """Shared crop/downsample geometry for a batch of canonical masks.

    Not pair-specific -- one geometry for N masks.
    """

    canon_shape_hw: tuple[int, int]
    crop_box_yx: BoxYX  # union of all content bboxes + padding
    crop_pad_hw: tuple[int, int]  # pad for downsample divisibility
    downsample_factor: int
    work_shape_hw: tuple[int, int]
    canonical_um_per_px: float
    work_um_per_px: float  # = canonical_um_per_px * downsample_factor


@dataclass(frozen=True)
class SharedWorkingGrid:
    """N canonical masks projected onto one shared work grid.

    All work_densities have shape == geom.work_shape_hw.
    Any (i,j) pair from this grid shares work_shape_hw -> 1 JIT bucket,
    and is dataloader-friendly (uniform tensor shapes).
    """

    geom: SharedGridGeometry
    work_densities: list[np.ndarray]  # len N; each shape == geom.work_shape_hw
    canon_masks: Optional[list[np.ndarray]]  # len N or None (omit to save memory)
    mask_ids: list[str]  # len N; sample/embryo IDs for traceability
    meta: dict

    @property
    def work_shape_hw(self) -> tuple[int, int]:
        return self.geom.work_shape_hw

    @property
    def n_masks(self) -> int:
        return len(self.work_densities)

    # -- Index lookup --------------------------------------------------

    def index_of(self, mask_id: str) -> int:
        """Return the integer index of *mask_id*.  Raises KeyError if absent."""
        try:
            return self.mask_ids.index(mask_id)
        except ValueError:
            raise KeyError(
                f"mask_id {mask_id!r} not in SharedWorkingGrid "
                f"(available: {self.mask_ids})"
            )

    def __contains__(self, mask_id: str) -> bool:
        return mask_id in self.mask_ids

    # -- Pair helpers --------------------------------------------------

    def get_pair(self, src_idx: int, tgt_idx: int) -> WorkingGridPair:
        """Build a WorkingGridPair from two integer indices."""
        return make_working_grid_pair_from_shared(self, src_idx, tgt_idx)

    def get_pair_by_id(
        self, src_id: str, tgt_id: str
    ) -> WorkingGridPair:
        """Build a WorkingGridPair from two mask_id strings.

        Example
        -------
        >>> shared = prepare_shared_working_grid(masks, cfg,
        ...     mask_ids=["ref", "tgt_0", "tgt_1", "tgt_2"])
        >>> pair = shared.get_pair_by_id("ref", "tgt_1")
        """
        return self.get_pair(self.index_of(src_id), self.index_of(tgt_id))


def prepare_shared_working_grid(
    masks: list[CanonicalMaskResult],
    cfg: Optional[WorkingGridConfig] = None,
    *,
    mask_ids: Optional[list[str]] = None,
    keep_canon_masks: bool = False,
) -> SharedWorkingGrid:
    """Prepare a shared working grid from N canonical masks.

    Validation (raises immediately on mismatch -- cohort consistency gate):
      1. All masks pass _require_canonical_mask.
      2. All masks share same .mask.shape (canvas size).
      3. All masks share same .grid.um_per_px (physical resolution).

    Args:
        masks: List of CanonicalMaskResult, all from same canonical config.
        cfg: Working grid config (downsample_factor, padding, mass_mode).
        mask_ids: Optional sample IDs; auto-generated if None.
        keep_canon_masks: If True, store cropped binary masks on the result.

    Returns:
        SharedWorkingGrid with all work_densities sharing identical shape.
    """
    if cfg is None:
        cfg = WorkingGridConfig()
    if len(masks) == 0:
        raise ValueError("At least one mask is required.")
    if mask_ids is not None and len(mask_ids) != len(masks):
        raise ValueError(
            f"mask_ids length ({len(mask_ids)}) must match masks length ({len(masks)})"
        )
    if mask_ids is None:
        mask_ids = [f"mask_{i}" for i in range(len(masks))]

    # --- Validation ---
    for i, m in enumerate(masks):
        _require_canonical_mask(m, role=f"masks[{i}] ({mask_ids[i]})")

    ref_shape = masks[0].mask.shape
    ref_um = float(masks[0].grid.um_per_px)
    for i, m in enumerate(masks):
        if m.mask.shape != ref_shape:
            raise ValueError(
                f"Canvas shape mismatch: masks[0] has shape {ref_shape}, "
                f"masks[{i}] ({mask_ids[i]}) has shape {m.mask.shape}"
            )
        if float(m.grid.um_per_px) != ref_um:
            raise ValueError(
                f"um_per_px mismatch: masks[0] has {ref_um}, "
                f"masks[{i}] ({mask_ids[i]}) has {float(m.grid.um_per_px)}"
            )

    # --- Compute per-mask bboxes ---
    bboxes: list[BoxYX] = []
    for i, m in enumerate(masks):
        if m.content_bbox_yx is not None:
            bboxes.append(m.content_bbox_yx)
        else:
            # Fallback: scan pixels (legacy results without content_bbox_yx)
            bbox = BoxYX.from_mask(m.mask)
            if bbox is None:
                raise ValueError(
                    f"masks[{i}] ({mask_ids[i]}) is empty (no nonzero pixels)"
                )
            bboxes.append(bbox)

    # --- Union all bboxes + padding + clamp ---
    union_box = reduce(BoxYX.union, bboxes)
    canon_h, canon_w = ref_shape
    crop_box = union_box.pad(cfg.padding_px, cfg.padding_px).clamp(canon_h, canon_w)

    # --- Compute padding for downsample divisibility ---
    crop_h, crop_w = crop_box.h, crop_box.w
    ds = cfg.downsample_factor
    pad_h = (ds - (crop_h % ds)) % ds
    pad_w = (ds - (crop_w % ds)) % ds
    crop_pad_hw = (pad_h, pad_w)

    padded_h = crop_h + pad_h
    padded_w = crop_w + pad_w
    work_h = padded_h // ds
    work_w = padded_w // ds
    work_shape_hw = (work_h, work_w)

    work_um_per_px = ref_um * float(ds)

    geom = SharedGridGeometry(
        canon_shape_hw=ref_shape,
        crop_box_yx=crop_box,
        crop_pad_hw=crop_pad_hw,
        downsample_factor=ds,
        work_shape_hw=work_shape_hw,
        canonical_um_per_px=ref_um,
        work_um_per_px=work_um_per_px,
    )

    # --- Per-mask projection ---
    work_densities: list[np.ndarray] = []
    canon_mask_list: Optional[list[np.ndarray]] = [] if keep_canon_masks else None
    sl = crop_box.to_slices()
    for m in masks:
        canon_mask = (np.asarray(m.mask) > 0).astype(np.uint8)
        cropped = canon_mask[sl]
        padded = np.pad(cropped, ((0, pad_h), (0, pad_w)), mode="constant")
        density = mask_to_density(padded, cfg.mass_mode)

        if cfg.mass_mode.name == "DISTANCE_TRANSFORM":
            density = enforce_min_mass(
                density, fallback=mask_to_density_uniform(padded)
            )

        if ds > 1:
            density = downsample_density(density, ds)

        assert density.shape == work_shape_hw
        work_densities.append(density)

        if canon_mask_list is not None:
            canon_mask_list.append(canon_mask)

    meta = {
        "coord_frame_id": "canonical_grid",
        "coord_frame_version": 1,
        "coord_convention": "yx",
        "canonical_um_per_px": ref_um,
        "work_um_per_px": work_um_per_px,
        "n_masks": len(masks),
        "work_grid": {
            "downsample_factor": ds,
            "padding_px": cfg.padding_px,
            "crop_policy": cfg.crop_policy,
            "crop_box_y0y1x0x1": (crop_box.y0, crop_box.y1, crop_box.x0, crop_box.x1),
            "crop_pad_hw": crop_pad_hw,
            "work_shape_hw": work_shape_hw,
        },
    }

    return SharedWorkingGrid(
        geom=geom,
        work_densities=work_densities,
        canon_masks=canon_mask_list,
        mask_ids=mask_ids,
        meta=meta,
    )


def make_working_grid_pair_from_shared(
    shared: SharedWorkingGrid,
    src_idx: int,
    tgt_idx: int,
) -> WorkingGridPair:
    """Build a WorkingGridPair from two indices into a SharedWorkingGrid.

    References work_densities directly (zero-copy).
    """
    n = shared.n_masks
    if not (0 <= src_idx < n):
        raise IndexError(f"src_idx={src_idx} out of range [0, {n})")
    if not (0 <= tgt_idx < n):
        raise IndexError(f"tgt_idx={tgt_idx} out of range [0, {n})")

    geom = shared.geom
    pair_frame = PairFrameGeometry.from_shared(geom)

    # Canon masks: use stored if available, else reconstruct isn't possible
    # without the original CanonicalMaskResult. We require canon_masks or skip.
    if shared.canon_masks is not None:
        src_canon = shared.canon_masks[src_idx]
        tgt_canon = shared.canon_masks[tgt_idx]
    else:
        # Create empty placeholders -- callers that need canon masks should
        # use keep_canon_masks=True when building the shared grid.
        src_canon = np.zeros(geom.canon_shape_hw, dtype=np.uint8)
        tgt_canon = np.zeros(geom.canon_shape_hw, dtype=np.uint8)

    return WorkingGridPair(
        coord_frame_id="canonical_grid",
        coord_frame_version=1,
        canonical_um_per_px=geom.canonical_um_per_px,
        work_um_per_px=geom.work_um_per_px,
        pair_frame=pair_frame,
        src_canon_mask=src_canon,
        tgt_canon_mask=tgt_canon,
        src_work_density=shared.work_densities[src_idx],
        tgt_work_density=shared.work_densities[tgt_idx],
        meta={
            "coord_frame_id": "canonical_grid",
            "coord_frame_version": 1,
            "coord_convention": "yx",
            "canonical_um_per_px": geom.canonical_um_per_px,
            "work_um_per_px": geom.work_um_per_px,
            "src_mask_id": shared.mask_ids[src_idx],
            "tgt_mask_id": shared.mask_ids[tgt_idx],
            "from_shared_grid": True,
            "work_grid": {
                "downsample_factor": geom.downsample_factor,
                "crop_box_y0y1x0x1": (
                    geom.crop_box_yx.y0,
                    geom.crop_box_yx.y1,
                    geom.crop_box_yx.x0,
                    geom.crop_box_yx.x1,
                ),
                "crop_pad_hw": geom.crop_pad_hw,
                "work_shape_hw": geom.work_shape_hw,
            },
            "src_mass_px": float(shared.work_densities[src_idx].sum()),
            "tgt_mass_px": float(shared.work_densities[tgt_idx].sum()),
        },
    )
