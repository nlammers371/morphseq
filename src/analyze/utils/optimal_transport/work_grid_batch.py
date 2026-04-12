"""Fixed-lattice batch working grid for OT over N canonical masks.

Unlike SharedWorkingGrid (pair-crop path), WorkGridBatch computes
work_full_density as a full-canvas downsample — one per mask, determined
only by (canonical_shape, ds). No batch-dependent geometry.

Workflow:
    batch = prepare_work_grid_batch(canonical_masks, downsample_factor=4)
    pair_pack = pack_pairs(batch, [("ref", "tgt_0"), ("ref", "tgt_1")])
    results = solve_pairs(pair_pack, uot_cfg, backend)

    star_pack = pack_star(batch, ref_ids=["ref"], src_ids=["tgt_0", "tgt_1"])
    results = solve_star(star_pack, uot_cfg, backend)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np

from analyze.utils.coord.types import BoxYX, CanonicalMaskResult
from analyze.utils.optimal_transport.density_transforms import (
    enforce_min_mass,
    mask_to_density,
    mask_to_density_uniform,
)
from analyze.utils.optimal_transport.multiscale_sampling import downsample_density
from analyze.utils.optimal_transport.working_grid import (
    WorkingGridConfig,
    _require_canonical_mask,
)


# ---------------------------------------------------------------------------
# Core batch dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkGridBatch:
    """N canonical masks on a fixed work lattice (full-canvas downsample).

    The lattice is determined only by (canon_shape_hw, ds, canonical_um_per_px)
    — no batch-dependent geometry. Same mask → identical density regardless
    of batch composition.
    """

    work_shape_hw_full: tuple[int, int]  # e.g. (64, 144) for ds=4 on (256,576)
    densities_full: np.ndarray  # (N, Hw, Ww) float32 contiguous
    mask_ids: list[str]
    canonical_um_per_px: float
    work_um_per_px: float
    downsample_factor: int
    canon_shape_hw: tuple[int, int]
    support_bboxes_work: list[BoxYX]  # per-mask tight bbox in work coords
    support_bbox_threshold: float  # threshold used (e.g. 0.0 or 1e-12)
    meta: dict  # work_grid_id, hashes
    # O(1) lookup dict built in __post_init__; not a constructor argument.
    _id_to_idx: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_id_to_idx", {mid: i for i, mid in enumerate(self.mask_ids)}
        )

    @property
    def n_masks(self) -> int:
        return len(self.mask_ids)

    def index_of(self, mask_id: str) -> int:
        """Return the integer index of *mask_id*.  Raises KeyError if absent."""
        try:
            return self._id_to_idx[mask_id]
        except KeyError:
            raise KeyError(
                f"mask_id {mask_id!r} not in WorkGridBatch "
                f"(available: {self.mask_ids})"
            )

    def __contains__(self, mask_id: str) -> bool:
        return mask_id in self._id_to_idx


# ---------------------------------------------------------------------------
# Crop policies
# ---------------------------------------------------------------------------


class CropPolicy(Enum):
    NONE = "none"


@dataclass(frozen=True)
class PerPairUnionCrop:
    margin_cells: int = 2


@dataclass(frozen=True)
class GlobalUnionCrop:
    mask_ids: Optional[list[str]] = None  # None = all
    margin_cells: int = 2


@dataclass(frozen=True)
class PerRefUnionCrop:
    ref_ids: list[str]
    src_ids: list[str]
    margin_cells: int = 2


CropPolicyType = Union[CropPolicy, PerPairUnionCrop, GlobalUnionCrop, PerRefUnionCrop]


# ---------------------------------------------------------------------------
# Pack results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairPack:
    """Packed pairs referencing a WorkGridBatch. No data copy — indices only."""

    src_indices: np.ndarray  # (P,) int — indices into batch.densities_full
    tgt_indices: np.ndarray  # (P,) int
    batch: WorkGridBatch  # reference (no copy)
    crop_boxes_work: Optional[list[BoxYX]]  # per-pair if cropped


@dataclass(frozen=True)
class StarPack:
    """Packed star topology referencing a WorkGridBatch."""

    ref_indices: np.ndarray  # (Nr,) int — into batch.densities_full
    src_indices: np.ndarray  # (Ns,) int — into batch.densities_full
    ref_ids: list[str]
    src_ids: list[str]
    chunk_size: int
    batch: WorkGridBatch  # reference (no copy)
    ref_densities: np.ndarray  # (Nr, Hw, Ww) — small contiguous copy
    crop_box_work: Optional[BoxYX]  # global crop if applied


# ---------------------------------------------------------------------------
# prepare_work_grid_batch
# ---------------------------------------------------------------------------


def prepare_work_grid_batch(
    masks: Union[list[CanonicalMaskResult], np.ndarray],
    downsample_factor: int = 4,
    *,
    mask_ids: Optional[list[str]] = None,
    canon_shape_hw: Optional[tuple[int, int]] = None,
    canonical_um_per_px: Optional[float] = None,
    support_bbox_threshold: float = 0.0,
    mass_mode: Optional["MassMode"] = None,
) -> WorkGridBatch:
    """Prepare a batch of masks on a fixed work lattice.

    Accepts either a list of CanonicalMaskResult or a raw (N, H, W) ndarray
    of canonical binary masks. For raw arrays, canon_shape_hw and
    canonical_um_per_px must be provided.

    Args:
        masks: CanonicalMaskResult list or (N, H, W) ndarray.
        downsample_factor: Downsampling factor.
        mask_ids: Sample IDs; auto-generated if None.
        canon_shape_hw: Required for raw array input.
        canonical_um_per_px: Required for raw array input.
        support_bbox_threshold: Threshold for support bbox computation.
        mass_mode: Mass mode for density computation; defaults to UNIFORM.

    Returns:
        WorkGridBatch with fixed-lattice densities.
    """
    from analyze.utils.optimal_transport.working_grid import WorkingGridConfig

    if mass_mode is None:
        mass_mode = WorkingGridConfig().mass_mode

    # --- Normalize input ---
    if isinstance(masks, np.ndarray):
        if masks.ndim != 3:
            raise ValueError(f"Raw mask array must be 3D (N,H,W), got {masks.ndim}D")
        if canon_shape_hw is None:
            canon_shape_hw = (masks.shape[1], masks.shape[2])
        if canonical_um_per_px is None:
            raise ValueError("canonical_um_per_px required for raw array input")
        raw_masks = [(masks[i] > 0).astype(np.uint8) for i in range(masks.shape[0])]
        n = masks.shape[0]
    elif isinstance(masks, list) and len(masks) > 0:
        if isinstance(masks[0], CanonicalMaskResult):
            # Validate
            for i, m in enumerate(masks):
                _require_canonical_mask(m, role=f"masks[{i}]")
            ref = masks[0]
            if canon_shape_hw is None:
                canon_shape_hw = ref.mask.shape
            if canonical_um_per_px is None:
                canonical_um_per_px = float(ref.grid.um_per_px)
            # Validate consistency
            for i, m in enumerate(masks):
                if m.mask.shape != canon_shape_hw:
                    raise ValueError(
                        f"Shape mismatch: masks[0]={canon_shape_hw}, "
                        f"masks[{i}]={m.mask.shape}"
                    )
                if float(m.grid.um_per_px) != canonical_um_per_px:
                    raise ValueError(
                        f"um_per_px mismatch: masks[0]={canonical_um_per_px}, "
                        f"masks[{i}]={float(m.grid.um_per_px)}"
                    )
            raw_masks = [(np.asarray(m.mask) > 0).astype(np.uint8) for m in masks]
            n = len(masks)
        else:
            raise TypeError(f"Expected CanonicalMaskResult or ndarray, got {type(masks[0])}")
    else:
        raise ValueError("At least one mask is required.")

    ds = downsample_factor
    canon_h, canon_w = canon_shape_hw

    if canon_h % ds != 0 or canon_w % ds != 0:
        raise ValueError(
            f"Canvas shape {canon_shape_hw} must be divisible by ds={ds}"
        )

    work_h = canon_h // ds
    work_w = canon_w // ds
    work_shape_hw_full = (work_h, work_w)

    if mask_ids is None:
        mask_ids = [f"mask_{i}" for i in range(n)]
    if len(mask_ids) != n:
        raise ValueError(f"mask_ids length ({len(mask_ids)}) != masks count ({n})")

    work_um_per_px = canonical_um_per_px * float(ds)

    # --- Per-mask projection (full-canvas, no crop) ---
    densities = np.empty((n, work_h, work_w), dtype=np.float32)
    support_bboxes: list[BoxYX] = []

    for i in range(n):
        canon_mask = raw_masks[i]
        density = mask_to_density(canon_mask, mass_mode)

        if mass_mode.name == "DISTANCE_TRANSFORM":
            density = enforce_min_mass(
                density, fallback=mask_to_density_uniform(canon_mask)
            )

        if ds > 1:
            density = downsample_density(density, ds)

        assert density.shape == work_shape_hw_full
        densities[i] = density.astype(np.float32)

        # Support bbox
        support = density > support_bbox_threshold
        bbox = BoxYX.from_mask(support.astype(np.uint8))
        if bbox is None:
            bbox = BoxYX(0, 0, 0, 0)
        support_bboxes.append(bbox)

    # --- Provenance ---
    _work_grid_id_tuple = (canon_shape_hw, ds, canonical_um_per_px)
    work_grid_id = hashlib.sha256(repr(_work_grid_id_tuple).encode()).hexdigest()[:16]

    _sorted_ids = sorted(mask_ids)
    mask_ids_hash_sorted = hashlib.sha256(
        "\n".join(_sorted_ids).encode()
    ).hexdigest()[:16]
    mask_ids_hash_ordered = hashlib.sha256(
        "\n".join(mask_ids).encode()
    ).hexdigest()[:16]

    meta = {
        "work_grid_id": work_grid_id,
        "mask_ids_hash_sorted": mask_ids_hash_sorted,
        "mask_ids_hash_ordered": mask_ids_hash_ordered,
        "coord_frame_id": "canonical_grid",
        "coord_frame_version": 1,
    }

    return WorkGridBatch(
        work_shape_hw_full=work_shape_hw_full,
        densities_full=densities,
        mask_ids=mask_ids,
        canonical_um_per_px=canonical_um_per_px,
        work_um_per_px=work_um_per_px,
        downsample_factor=ds,
        canon_shape_hw=canon_shape_hw,
        support_bboxes_work=support_bboxes,
        support_bbox_threshold=support_bbox_threshold,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------


def _compute_union_crop_box(
    batch: WorkGridBatch,
    indices: list[int],
    margin_cells: int,
) -> BoxYX:
    """Compute union of support bboxes for given indices + margin, clamped."""
    bboxes = [batch.support_bboxes_work[i] for i in indices]
    union = bboxes[0]
    for b in bboxes[1:]:
        union = union.union(b)
    return union.pad(margin_cells, margin_cells).clamp(*batch.work_shape_hw_full)


# ---------------------------------------------------------------------------
# pack_pairs
# ---------------------------------------------------------------------------


def pack_pairs(
    batch: WorkGridBatch,
    pairs: list[tuple[str, str]],
    crop_policy: Union[CropPolicy, PerPairUnionCrop] = CropPolicy.NONE,
) -> PairPack:
    """Pack pairs as index arrays into a WorkGridBatch.

    No data copy — indices into batch.densities_full.
    """
    src_indices = np.array([batch.index_of(s) for s, _ in pairs], dtype=np.intp)
    tgt_indices = np.array([batch.index_of(t) for _, t in pairs], dtype=np.intp)

    crop_boxes: Optional[list[BoxYX]] = None
    if isinstance(crop_policy, PerPairUnionCrop):
        crop_boxes = []
        for si, ti in zip(src_indices, tgt_indices):
            box = _compute_union_crop_box(
                batch, [int(si), int(ti)], crop_policy.margin_cells
            )
            crop_boxes.append(box)

    return PairPack(
        src_indices=src_indices,
        tgt_indices=tgt_indices,
        batch=batch,
        crop_boxes_work=crop_boxes,
    )


# ---------------------------------------------------------------------------
# pack_star
# ---------------------------------------------------------------------------


def pack_star(
    batch: WorkGridBatch,
    ref_ids: list[str],
    src_ids: list[str],
    chunk_size: int = 32,
    crop_policy: Union[CropPolicy, PerRefUnionCrop, GlobalUnionCrop] = CropPolicy.NONE,
) -> StarPack:
    """Pack star topology (ref × src) referencing a WorkGridBatch.

    ref_densities is a small contiguous copy for speed.
    src data stays as indices — gathered per chunk at solve time.
    """
    ref_indices = np.array([batch.index_of(r) for r in ref_ids], dtype=np.intp)
    src_indices = np.array([batch.index_of(s) for s in src_ids], dtype=np.intp)

    # Small contiguous copy of ref densities
    ref_densities = batch.densities_full[ref_indices].copy()

    crop_box: Optional[BoxYX] = None
    if isinstance(crop_policy, GlobalUnionCrop):
        if crop_policy.mask_ids is not None:
            indices = [batch.index_of(mid) for mid in crop_policy.mask_ids]
        else:
            indices = list(range(batch.n_masks))
        crop_box = _compute_union_crop_box(batch, indices, crop_policy.margin_cells)
    elif isinstance(crop_policy, PerRefUnionCrop):
        all_ids = list(set(crop_policy.ref_ids) | set(crop_policy.src_ids))
        indices = [batch.index_of(mid) for mid in all_ids]
        crop_box = _compute_union_crop_box(batch, indices, crop_policy.margin_cells)

    return StarPack(
        ref_indices=ref_indices,
        src_indices=src_indices,
        ref_ids=ref_ids,
        src_ids=src_ids,
        chunk_size=chunk_size,
        batch=batch,
        ref_densities=ref_densities,
        crop_box_work=crop_box,
    )
