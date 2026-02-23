"""
Phase 0 Step 1: Generate OT maps (fixed WT reference → many targets).

Wraps the existing UOT pipeline to produce per-sample feature maps
on the canonical grid (256×576 at 10 µm/px). Outputs cost_density, 
displacement field, and delta_mass per sample.

Usage:
    from p0_ot_maps import generate_ot_maps
    results = generate_ot_maps(mask_ref, target_masks, config)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from roi_config import (
    FeatureDatasetConfig,
    Phase0FeatureSet,
    PHASE0_CHANNEL_SCHEMAS,
)

logger = logging.getLogger(__name__)


@dataclass
class OTMapResult:
    """Result from a single ref→target OT computation."""
    sample_id: str
    cost_density: np.ndarray       # (H, W) float32 — canonical grid shape
    displacement_yx: np.ndarray    # (H, W, 2) float32 — (v, u) convention
    delta_mass: np.ndarray         # (H, W) float32 — created - destroyed
    aligned_ref_mask: Optional[np.ndarray]    # (H, W) uint8 — exact OT-aligned source mask
    aligned_target_mask: Optional[np.ndarray] # (H, W) uint8 — exact OT-aligned target mask
    total_cost_C: float
    diagnostics: Dict
    alignment_debug: Optional[Dict] = None


def _compute_ot_params_hash(config_dict: dict) -> str:
    """Deterministic hash of OT parameters for provenance."""
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_single_ot(
    mask_ref: np.ndarray,
    mask_target: np.ndarray,
    sample_id: str,
    raw_um_per_px_ref: float,
    raw_um_per_px_tgt: float,
    yolk_ref: Optional[np.ndarray] = None,
    yolk_tgt: Optional[np.ndarray] = None,
    source_id: Optional[str] = None,
    target_id: Optional[str] = None,
    uot_config=None,
    backend=None,
) -> OTMapResult:
    """
    Run unbalanced OT: ref → target, return canonical-grid maps.

    Geometry stage (canonicalization via analyze.utils.coord) runs here, upstream of
    the solver. UOT receives pre-canonical masks and acts as a pure solver.

    Parameters
    ----------
    mask_ref : (H_ref, W_ref) uint8 — reference mask at raw resolution
    mask_target : (H_tgt, W_tgt) uint8 — target mask at raw resolution
    sample_id : str
    raw_um_per_px_ref : float — physical resolution of reference mask
    raw_um_per_px_tgt : float — physical resolution of target mask
    yolk_ref : (H_ref, W_ref) uint8, optional — reference yolk at raw resolution
    yolk_tgt : (H_tgt, W_tgt) uint8, optional — target yolk at raw resolution
    uot_config : UOTConfig, optional — must have use_canonical_grid=False
    backend : UOTBackend, optional

    Returns
    -------
    OTMapResult with maps on canonical grid (256, 576)
    """
    from analyze.utils.optimal_transport import (
        UOTConfig, MassMode,
        WorkingGridConfig, prepare_working_grid_pair,
        run_uot_on_working_grid, lift_work_result_to_canonical,
    )
    from analyze.utils.coord.grids.canonical import CanonicalGridConfig, to_canonical_grid_mask

    # ------------------------------------------------------------------
    # Geometry stage: canonicalize ref and target independently
    # ------------------------------------------------------------------
    CANONICAL_GRID_HW = (256, 576)
    CANONICAL_UM_PER_PX = 10.0

    canonical_cfg = CanonicalGridConfig(
        reference_um_per_pixel=CANONICAL_UM_PER_PX,
        grid_shape_hw=CANONICAL_GRID_HW,
        align_mode="yolk",
    )

    # Fail fast on mask/yolk shape mismatches. Canonicalization expects
    # embryo and yolk masks to be in the same raw coordinate system.
    if yolk_ref is not None and yolk_ref.shape != mask_ref.shape:
        raise ValueError(
            f"Reference yolk shape {yolk_ref.shape} does not match reference mask shape {mask_ref.shape}."
        )
    if yolk_tgt is not None and yolk_tgt.shape != mask_target.shape:
        raise ValueError(
            f"Target yolk shape {yolk_tgt.shape} does not match target mask shape {mask_target.shape}."
        )

    # Minimal QC for now: binaryize. (If you want morphology QC, add it explicitly upstream.)
    ref_bin = (np.asarray(mask_ref) > 0).astype(np.uint8)
    tgt_bin = (np.asarray(mask_target) > 0).astype(np.uint8)

    src_canon_result = to_canonical_grid_mask(
        ref_bin,
        um_per_px=float(raw_um_per_px_ref),
        yolk_mask=(yolk_ref.astype(np.uint8) if yolk_ref is not None else None),
        cfg=canonical_cfg,
    )
    tgt_canon_result = to_canonical_grid_mask(
        tgt_bin,
        um_per_px=float(raw_um_per_px_tgt),
        yolk_mask=(yolk_tgt.astype(np.uint8) if yolk_tgt is not None else None),
        cfg=canonical_cfg,
    )

    # ------------------------------------------------------------------
    # Solver stage: UOT is a pure consumer of canonical masks
    # ------------------------------------------------------------------
    working_cfg = WorkingGridConfig(
        downsample_factor=1,
        padding_px=16,
        mass_mode=MassMode.UNIFORM,
    )

    if uot_config is None:
        uot_config = UOTConfig(
            epsilon=1e-4,
            marginal_relaxation=10.0,
            max_support_points=3000,
            store_coupling=True,
            random_seed=42,
            metric="sqeuclidean",
            coord_scale=1.0 / max(CANONICAL_GRID_HW),
        )

    if backend is None:
        # Prefer OTT (JAX) by default; POT may pull in torch.
        try:
            from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend

            backend = OTTBackend()
        except Exception:
            from analyze.utils.optimal_transport.backends.pot_backend import POTBackend

            backend = POTBackend()

    pair_work = prepare_working_grid_pair(src_canon_result, tgt_canon_result, working_cfg)
    result_work = run_uot_on_working_grid(pair_work, config=uot_config, backend=backend)
    result = lift_work_result_to_canonical(result_work, pair_work)

    # Extract maps (canonical-shaped)
    cost_density = result.cost_src_canon if result.cost_src_canon is not None else np.zeros(CANONICAL_GRID_HW, dtype=np.float32)
    displacement_yx = result.velocity_canon_px_per_step_yx  # (H, W, 2)
    delta_mass = result.mass_created_canon - result.mass_destroyed_canon

    # Verify canonical shape
    expected_hw = CANONICAL_GRID_HW
    bad_shapes = []
    if cost_density.shape != expected_hw:
        bad_shapes.append(f"cost_density={cost_density.shape}")
    if displacement_yx.shape != (expected_hw[0], expected_hw[1], 2):
        bad_shapes.append(f"displacement_yx={displacement_yx.shape}")
    if delta_mass.shape != expected_hw:
        bad_shapes.append(f"delta_mass={delta_mass.shape}")
    if bad_shapes:
        raise ValueError(
            f"OT output shapes are not canonical (expected {expected_hw}): {', '.join(bad_shapes)}."
        )

    src_align = (src_canon_result.meta or {}).get("align_meta", {}) or {}
    tgt_align = (tgt_canon_result.meta or {}).get("align_meta", {}) or {}
    metrics = {}
    if isinstance(result.diagnostics, dict):
        metrics = result.diagnostics.get("metrics", {}) or {}

    src_mask_aligned = src_canon_result.mask.astype(np.uint8)
    tgt_mask_aligned = tgt_canon_result.mask.astype(np.uint8)
    overlap_iou = np.nan
    src_bool = src_mask_aligned > 0
    tgt_bool = tgt_mask_aligned > 0
    union = float(np.logical_or(src_bool, tgt_bool).sum())
    inter = float(np.logical_and(src_bool, tgt_bool).sum())
    overlap_iou = inter / union if union > 0 else np.nan

    # Bbox/pad info from the working-grid pair meta
    wg_meta = pair_work.meta if pair_work.meta else {}
    wg_block = wg_meta.get("work_grid", {}) if isinstance(wg_meta, dict) else {}
    bbox_y0y1x0x1 = wg_block.get("bbox_y0y1x0x1", (np.nan, np.nan, np.nan, np.nan))
    pad_hw = wg_block.get("crop_pad_hw", (np.nan, np.nan))
    bbox_y0, bbox_y1, bbox_x0, bbox_x1 = bbox_y0y1x0x1
    pad_h, pad_w = pad_hw

    alignment_debug = {
        "sample_id": sample_id,
        "source_id": source_id,
        "target_id": target_id,
        "src_rotation_deg": src_align.get("rotation_deg"),
        "src_flip": src_align.get("flip"),
        "src_retained_ratio": src_align.get("retained_ratio"),
        "src_anchor_shift_x": (
            src_align.get("anchor_shift_xy", (np.nan, np.nan))[0]
            if src_align.get("anchor_shift_xy") is not None else np.nan
        ),
        "src_anchor_shift_y": (
            src_align.get("anchor_shift_xy", (np.nan, np.nan))[1]
            if src_align.get("anchor_shift_xy") is not None else np.nan
        ),
        "src_head_y_final": (
            src_align.get("head_yx_final", (np.nan, np.nan))[0]
            if src_align.get("head_yx_final") is not None else np.nan
        ),
        "src_head_x_final": (
            src_align.get("head_yx_final", (np.nan, np.nan))[1]
            if src_align.get("head_yx_final") is not None else np.nan
        ),
        "src_back_y_final": (
            src_align.get("back_yx_final", (np.nan, np.nan))[0]
            if src_align.get("back_yx_final") is not None else np.nan
        ),
        "src_back_x_final": (
            src_align.get("back_yx_final", (np.nan, np.nan))[1]
            if src_align.get("back_yx_final") is not None else np.nan
        ),
        "tgt_rotation_deg": tgt_align.get("rotation_deg"),
        "tgt_flip": tgt_align.get("flip"),
        "tgt_retained_ratio": tgt_align.get("retained_ratio"),
        "tgt_anchor_shift_x": (
            tgt_align.get("anchor_shift_xy", (np.nan, np.nan))[0]
            if tgt_align.get("anchor_shift_xy") is not None else np.nan
        ),
        "tgt_anchor_shift_y": (
            tgt_align.get("anchor_shift_xy", (np.nan, np.nan))[1]
            if tgt_align.get("anchor_shift_xy") is not None else np.nan
        ),
        "tgt_head_y_final": (
            tgt_align.get("head_yx_final", (np.nan, np.nan))[0]
            if tgt_align.get("head_yx_final") is not None else np.nan
        ),
        "tgt_head_x_final": (
            tgt_align.get("head_yx_final", (np.nan, np.nan))[1]
            if tgt_align.get("head_yx_final") is not None else np.nan
        ),
        "tgt_back_y_final": (
            tgt_align.get("back_yx_final", (np.nan, np.nan))[0]
            if tgt_align.get("back_yx_final") is not None else np.nan
        ),
        "tgt_back_x_final": (
            tgt_align.get("back_yx_final", (np.nan, np.nan))[1]
            if tgt_align.get("back_yx_final") is not None else np.nan
        ),
        "pair_bbox_y0": bbox_y0,
        "pair_bbox_y1": bbox_y1,
        "pair_bbox_x0": bbox_x0,
        "pair_bbox_x1": bbox_x1,
        "pair_pad_h": pad_h,
        "pair_pad_w": pad_w,
        "total_cost_C": float(result.cost),
        "mass_delta_crop": metrics.get("mass_delta_crop"),
        "mass_ratio_crop": metrics.get("mass_ratio_crop"),
        "overlap_iou_src_tgt": overlap_iou,
    }

    return OTMapResult(
        sample_id=sample_id,
        cost_density=cost_density.astype(np.float32),
        displacement_yx=displacement_yx.astype(np.float32),
        delta_mass=delta_mass.astype(np.float32),
        aligned_ref_mask=src_mask_aligned,
        aligned_target_mask=tgt_mask_aligned,
        total_cost_C=float(result.cost),
        diagnostics=result.diagnostics or {},
        alignment_debug=alignment_debug,
    )


def generate_ot_maps(
    mask_ref: np.ndarray,
    target_masks: List[np.ndarray],
    sample_ids: List[str],
    raw_um_per_px_ref: float,
    raw_um_per_px_targets: np.ndarray,
    yolk_ref: Optional[np.ndarray] = None,
    yolk_targets: Optional[List[np.ndarray]] = None,
    feature_set: Phase0FeatureSet = Phase0FeatureSet.V0_COST,
    uot_config=None,
    backend=None,
    source_id: Optional[str] = None,
    target_ids: Optional[List[str]] = None,
    return_aligned_masks: bool = False,
    collect_debug: bool = True,
    strict_debug_ids: bool = False,
    return_debug_df: bool = False,
):
    """
    Run OT for all targets against fixed reference, return feature array X and total_cost_C.

    Masks are provided at raw resolution; UOT pipeline aligns to canonical grid.

    Parameters
    ----------
    mask_ref : (H_ref, W_ref) uint8 — reference mask at raw resolution
    target_masks : list of (H_i, W_i) uint8, length N — raw resolution
    sample_ids : list of str, length N
    raw_um_per_px_ref : float — physical resolution of reference mask
    raw_um_per_px_targets : (N,) array — physical resolution per target
    feature_set : Phase0FeatureSet

    Returns
    -------
    If return_aligned_masks=False:
        X : (N, 256, 576, C) float32 — feature maps per sample on canonical grid
        total_cost_C : (N,) float32 — total OT cost per sample
    If return_aligned_masks=True:
        X, total_cost_C, aligned_ref_mask, aligned_target_masks
        aligned_ref_mask : (256, 576) uint8
        aligned_target_masks : (N, 256, 576) uint8
    If return_debug_df=True:
        returns an additional DataFrame with per-sample alignment diagnostics.
    """
    N = len(target_masks)
    assert len(sample_ids) == N
    assert len(raw_um_per_px_targets) == N
    if yolk_targets is not None:
        assert len(yolk_targets) == N
    if target_ids is not None:
        assert len(target_ids) == N

    if collect_debug and (source_id is None or target_ids is None):
        msg = (
            "Alignment debug capture is enabled, but source_id/target_ids were not fully provided. "
            "Debug rows will miss explicit source/target IDs. Pass source_id + target_ids "
            "(for example embryo_id|frame_index) to make downstream debugging traceable."
        )
        if strict_debug_ids:
            raise ValueError(msg)
        logger.warning(msg)
    
    # Canonical grid shape
    H, W = 256, 576
    channel_schemas = PHASE0_CHANNEL_SCHEMAS[feature_set]
    C = len(channel_schemas)

    X = np.zeros((N, H, W, C), dtype=np.float32)
    total_cost_C = np.zeros(N, dtype=np.float32)
    aligned_ref_mask = None
    aligned_target_masks = np.zeros((N, H, W), dtype=np.uint8) if return_aligned_masks else None
    debug_rows = []

    for i, (mask_tgt, sid) in enumerate(zip(target_masks, sample_ids)):
        logger.info(f"OT map {i+1}/{N}: {sid}")

        ot_result = run_single_ot(
            mask_ref, mask_tgt, sid,
            raw_um_per_px_ref=raw_um_per_px_ref,
            raw_um_per_px_tgt=raw_um_per_px_targets[i],
            yolk_ref=yolk_ref,
            yolk_tgt=yolk_targets[i] if yolk_targets is not None else None,
            source_id=source_id,
            target_id=target_ids[i] if target_ids is not None else None,
            uot_config=uot_config, backend=backend,
        )

        total_cost_C[i] = ot_result.total_cost_C

        if return_aligned_masks:
            if ot_result.aligned_ref_mask is None or ot_result.aligned_target_mask is None:
                raise ValueError(
                    "run_single_ot did not return aligned masks. "
                    "Expected aligned_ref_mask and aligned_target_mask for QC overlays."
                )
            if ot_result.aligned_ref_mask.shape != (H, W):
                raise ValueError(
                    f"aligned_ref_mask shape mismatch: {ot_result.aligned_ref_mask.shape} vs {(H, W)}"
                )
            if ot_result.aligned_target_mask.shape != (H, W):
                raise ValueError(
                    f"aligned_target_mask shape mismatch: {ot_result.aligned_target_mask.shape} vs {(H, W)}"
                )

            if aligned_ref_mask is None:
                aligned_ref_mask = ot_result.aligned_ref_mask.astype(np.uint8)
            else:
                if not np.array_equal(aligned_ref_mask, ot_result.aligned_ref_mask):
                    logger.warning(
                        "Aligned reference mask changed across samples for the same reference. "
                        "Using first sample's aligned reference mask."
                    )

            aligned_target_masks[i] = ot_result.aligned_target_mask.astype(np.uint8)

        if collect_debug and ot_result.alignment_debug is not None:
            debug_rows.append(ot_result.alignment_debug)

        if feature_set == Phase0FeatureSet.V0_COST:
            X[i, :, :, 0] = ot_result.cost_density
        elif feature_set == Phase0FeatureSet.V1_DYNAMICS:
            X[i, :, :, 0] = ot_result.cost_density
            X[i, :, :, 1] = ot_result.displacement_yx[:, :, 1]  # disp_u (x)
            X[i, :, :, 2] = ot_result.displacement_yx[:, :, 0]  # disp_v (y)
            X[i, :, :, 3] = np.sqrt(
                ot_result.displacement_yx[:, :, 0]**2
                + ot_result.displacement_yx[:, :, 1]**2
            )  # disp_mag
            X[i, :, :, 4] = ot_result.delta_mass

    logger.info(f"Generated OT maps: X shape={X.shape}, mean cost={total_cost_C.mean():.4f}")
    debug_df = pd.DataFrame(debug_rows) if return_debug_df else None

    if return_aligned_masks and return_debug_df:
        if aligned_ref_mask is None:
            raise ValueError("Aligned reference mask was never populated.")
        return X, total_cost_C, aligned_ref_mask, aligned_target_masks, debug_df
    if return_aligned_masks:
        if aligned_ref_mask is None:
            raise ValueError("Aligned reference mask was never populated.")
        return X, total_cost_C, aligned_ref_mask, aligned_target_masks
    if return_debug_df:
        return X, total_cost_C, debug_df
    return X, total_cost_C


__all__ = [
    "OTMapResult",
    "run_single_ot",
    "generate_ot_maps",
]
