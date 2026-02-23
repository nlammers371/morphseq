"""Explicit registration utilities ("Stage 2" registration).

Registration is always explicit. Nothing here is called implicitly by UOT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .transforms import GridTransform, TransformChain
from .types import RegisterResult


@dataclass
class RegisterConfig:
    mode: str = "rotate_only"  # "rotate_only" | "rotate_then_pivot_translate"
    angle_step_deg: float = 1.0
    min_iou_absolute: float = 0.25
    min_iou_improvement: float = 0.02


def _mask_com(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask > 0.5)
    if ys.size == 0:
        h, w = mask.shape[:2]
        return (h / 2.0, w / 2.0)
    return (float(ys.mean()), float(xs.mean()))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = a > 0.5
    bb = b > 0.5
    intersection = float(np.logical_and(ab, bb).sum())
    union = float(np.logical_or(ab, bb).sum())
    return intersection / (union + 1e-6)


def _apply_pivot_rotation(mask: np.ndarray, pivot_yx: tuple[float, float], angle_deg: float) -> np.ndarray:
    if cv2 is None:
        raise ImportError("cv2 is required for registration.")
    cy, cx = float(pivot_yx[0]), float(pivot_yx[1])
    h, w = mask.shape[:2]
    theta = np.radians(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    M = np.float32(
        [
            [cos_t, -sin_t, cx * (1 - cos_t) + cy * sin_t],
            [sin_t, cos_t, cy * (1 - cos_t) - cx * sin_t],
        ]
    )
    return cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST)


def _apply_translation(mask: np.ndarray, dyx: tuple[float, float]) -> np.ndarray:
    if cv2 is None:
        raise ImportError("cv2 is required for registration.")
    dy, dx = float(dyx[0]), float(dyx[1])
    h, w = mask.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST)


def register_to_fixed(
    moving: np.ndarray,
    fixed: np.ndarray,
    *,
    cfg: Optional[RegisterConfig] = None,
    apply: bool = False,
    moving_pivot_yx: Optional[Tuple[float, float]] = None,
    fixed_pivot_yx: Optional[Tuple[float, float]] = None,
) -> RegisterResult:
    """Register `moving` into `fixed` coordinate frame (explicit)."""
    cfg = cfg or RegisterConfig()
    if moving.shape != fixed.shape:
        raise ValueError(f"Shape mismatch: moving={moving.shape} fixed={fixed.shape}")
    if not np.isfinite(cfg.angle_step_deg) or cfg.angle_step_deg <= 0:
        raise ValueError("angle_step_deg must be > 0")
    if cfg.mode not in ("rotate_only", "rotate_then_pivot_translate"):
        raise ValueError(f"Unknown mode={cfg.mode!r}")

    h, w = moving.shape[:2]

    if moving.sum() == 0 or fixed.sum() == 0:
        identity = TransformChain.identity(shape_yx=(h, w), interp="nearest")
        meta = {
            "coord_frame_id": "canonical_grid",
            "coord_frame_version": 1,
            "coord_convention": "yx",
            "register_to_fixed": {"applied": False, "reason": "empty_mask", "moving_in_fixed": bool(apply)},
        }
        return RegisterResult(transform=identity, applied=False, meta=meta, moving_in_fixed=moving.copy() if apply else None)

    tgt_pivot = moving_pivot_yx if moving_pivot_yx is not None else _mask_com(moving)
    src_pivot = fixed_pivot_yx if fixed_pivot_yx is not None else _mask_com(fixed)

    iou_before = _iou(moving, fixed)
    angles = np.arange(-180.0, 180.0 + cfg.angle_step_deg, cfg.angle_step_deg)
    best_angle = 0.0
    best_iou = -1.0
    iou_at_0 = iou_before
    for a in angles:
        rotated = _apply_pivot_rotation(moving, tgt_pivot, float(a))
        iou_val = _iou(rotated, fixed)
        if abs(float(a)) < 0.5:
            iou_at_0 = iou_val
        if iou_val > best_iou:
            best_iou = iou_val
            best_angle = float(a)

    apply_rot = (best_iou >= cfg.min_iou_absolute) and (best_iou >= iou_at_0 + cfg.min_iou_improvement)
    if apply_rot:
        angle_out = best_angle
        rot_applied = True
    else:
        angle_out = 0.0
        rot_applied = False

    transforms: list[GridTransform] = []
    if rot_applied:
        # Store rotation matrix used in warpAffine.
        cy, cx = float(tgt_pivot[0]), float(tgt_pivot[1])
        theta = np.radians(float(angle_out))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        M = np.array(
            [
                [cos_t, -sin_t, cx * (1 - cos_t) + cy * sin_t],
                [sin_t, cos_t, cy * (1 - cos_t) - cx * sin_t],
            ],
            dtype=np.float64,
        )
        transforms.append(
            GridTransform(
                name="pivot_rotate",
                affine_2x3=M,
                in_shape_yx=(h, w),
                out_shape_yx=(h, w),
                interp="nearest",
                params={"angle_deg": float(angle_out), "pivot_yx": (float(cy), float(cx)), "affine_convention": "opencv_xy"},
            )
        )

    translate_dyx = (0.0, 0.0)
    mode_effective = cfg.mode
    if mode_effective == "rotate_then_pivot_translate" and rot_applied:
        translate_dyx = (float(src_pivot[0] - tgt_pivot[0]), float(src_pivot[1] - tgt_pivot[1]))
        M_t = np.array([[1.0, 0.0, float(translate_dyx[1])], [0.0, 1.0, float(translate_dyx[0])]], dtype=np.float64)
        transforms.append(
            GridTransform(
                name="translate",
                affine_2x3=M_t,
                in_shape_yx=(h, w),
                out_shape_yx=(h, w),
                interp="nearest",
                params={"translate_dyx": translate_dyx, "affine_convention": "opencv_xy"},
            )
        )

    chain = TransformChain(transforms=transforms) if transforms else TransformChain.identity(shape_yx=(h, w), interp="nearest")

    moving_in_fixed = None
    if apply:
        out = moving
        if rot_applied:
            out = _apply_pivot_rotation(out, tgt_pivot, angle_out)
        if mode_effective == "rotate_then_pivot_translate" and rot_applied:
            out = _apply_translation(out, translate_dyx)
        moving_in_fixed = (out > 0.5).astype(np.uint8)

    meta = {
        "coord_frame_id": "canonical_grid",
        "coord_frame_version": 1,
        "coord_convention": "yx",
        "register_to_fixed": {
            "applied": bool(rot_applied),
            "mode": mode_effective,
            "angle_deg": float(angle_out),
            "best_angle_deg": float(best_angle),
            "best_iou": float(best_iou),
            "iou_before": float(iou_before),
            "iou_after": float(_iou(moving_in_fixed if moving_in_fixed is not None else moving, fixed)),
            "moving_pivot_yx": (float(tgt_pivot[0]), float(tgt_pivot[1])),
            "fixed_pivot_yx": (float(src_pivot[0]), float(src_pivot[1])),
            "translate_dyx": translate_dyx,
            "moving_in_fixed": bool(apply),
        },
    }
    return RegisterResult(transform=chain, applied=bool(rot_applied), meta=meta, moving_in_fixed=moving_in_fixed)

