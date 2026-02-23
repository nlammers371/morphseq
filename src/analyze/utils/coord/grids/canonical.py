"""Canonical grid mapping utilities ("Stage 1" canonicalization).

Public convenience verbs:
- `to_canonical_grid_mask`
- `to_canonical_grid_image`
- `to_canonical_grid_frame`

The internal engine is `CanonicalGridMapper`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from scipy import ndimage

from ..types import (
    CanonicalFrameResult,
    CanonicalGrid,
    CanonicalImageResult,
    CanonicalMaskResult,
    Frame,
)
from ..transforms import GridTransform, TransformChain


@dataclass
class CanonicalGridConfig:
    """Legacy-style config for canonical grid standardization."""

    reference_um_per_pixel: float = 10.0
    grid_shape_hw: tuple[int, int] = (256, 576)
    padding_um: float = 50.0
    align_mode: str = "yolk"  # "yolk" | "centroid" | "none"
    allow_flip: bool = True
    anchor_mode: str = "yolk_anchor"  # "yolk_anchor" | "com_center"
    anchor_frac_yx: tuple[float, float] = (0.50, 0.50)
    clipping_threshold: float = 0.98
    error_on_clip: bool = False
    downsample_factor: int = 1
    assume_canonical: bool = False


class CanonicalAligner:
    """Ported canonical aligner (PCA+scale+flip+anchor), independent of UOT."""

    def __init__(
        self,
        target_shape_hw: tuple[int, int] = (256, 576),
        target_um_per_pixel: float = 10.0,
        allow_flip: bool = True,
        anchor_mode: str = "yolk_anchor",
        anchor_frac_yx: tuple[float, float] = (0.50, 0.50),
        clipping_threshold: float = 0.98,
        error_on_clip: bool = False,
        yolk_weight: float = 1.0,
        back_weight: float = 1.0,
        back_sample_radius_k: float = 1.75,
        yolk_pivot_angle_range_deg: float = 180.0,
        yolk_pivot_angle_step_deg: float = 1.0,
    ) -> None:
        if cv2 is None:
            raise ImportError("cv2 is required for CanonicalAligner.")
        self.H, self.W = target_shape_hw
        self.target_res = float(target_um_per_pixel)
        self.allow_flip = bool(allow_flip)
        self.anchor_mode = str(anchor_mode)
        self.anchor_frac_yx = anchor_frac_yx
        self.clipping_threshold = float(clipping_threshold)
        self.error_on_clip = bool(error_on_clip)
        self.yolk_weight = float(yolk_weight)
        self.back_weight = float(back_weight)
        self.back_sample_radius_k = float(back_sample_radius_k)
        self.yolk_pivot_angle_range_deg = float(yolk_pivot_angle_range_deg)
        self.yolk_pivot_angle_step_deg = float(yolk_pivot_angle_step_deg)
        self._last_back_debug: Optional[dict] = None

        self.is_landscape = self.W >= self.H
        y_frac, x_frac = anchor_frac_yx
        self.anchor_point_xy = (self.W * float(x_frac), self.H * float(y_frac))

    @classmethod
    def from_config(cls, config: CanonicalGridConfig) -> "CanonicalAligner":
        return cls(
            target_shape_hw=config.grid_shape_hw,
            target_um_per_pixel=config.reference_um_per_pixel,
            allow_flip=config.allow_flip,
            anchor_mode=config.anchor_mode,
            anchor_frac_yx=config.anchor_frac_yx,
            clipping_threshold=config.clipping_threshold,
            error_on_clip=config.error_on_clip,
        )

    def _center_of_mass(self, mask: np.ndarray) -> tuple[float, float]:
        if mask is None or np.sum(mask) == 0:
            return (self.H / 2.0, self.W / 2.0)
        cy, cx = ndimage.center_of_mass(mask)
        return float(cy), float(cx)

    def _pca_angle_deg(self, mask: np.ndarray) -> tuple[float, tuple[float, float], bool]:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return 0.0, (self.W / 2.0, self.H / 2.0), False
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        mean = coords.mean(axis=0)
        coords_centered = coords - mean
        cov = np.cov(coords_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvals[0] <= 0:
            return 0.0, (float(mean[0]), float(mean[1])), False
        angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        return angle, (float(mean[0]), float(mean[1])), True

    def _warp(self, mask: np.ndarray, M: np.ndarray, *, out_shape_hw: Optional[tuple[int, int]] = None) -> np.ndarray:
        h, w = out_shape_hw if out_shape_hw is not None else (self.H, self.W)
        return cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST)

    def _bbox(self, mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        ys, xs = np.where(mask > 0.5)
        if ys.size == 0:
            return None
        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    def _project_point_to_mask_in_disk(
        self,
        mask: np.ndarray,
        yx: tuple[float, float],
        disk_center_yx: tuple[float, float],
        disk_radius: float,
    ) -> tuple[float, float]:
        y, x = float(yx[0]), float(yx[1])
        iy = int(np.clip(round(y), 0, mask.shape[0] - 1))
        ix = int(np.clip(round(x), 0, mask.shape[1] - 1))
        if mask[iy, ix] > 0.5:
            return float(iy), float(ix)

        ys, xs = np.where(mask > 0.5)
        if ys.size == 0:
            return y, x

        dy_disk = ys.astype(np.float64) - disk_center_yx[0]
        dx_disk = xs.astype(np.float64) - disk_center_yx[1]
        in_disk = (dy_disk**2 + dx_disk**2) <= disk_radius**2
        if not in_disk.any():
            d2 = (ys.astype(np.float64) - y) ** 2 + (xs.astype(np.float64) - x) ** 2
            idx = int(np.argmin(d2))
            return float(ys[idx]), float(xs[idx])

        ys_disk = ys[in_disk]
        xs_disk = xs[in_disk]
        d2 = (ys_disk.astype(np.float64) - y) ** 2 + (xs_disk.astype(np.float64) - x) ** 2
        idx = int(np.argmin(d2))
        return float(ys_disk[idx]), float(xs_disk[idx])

    def _compute_back_direction(
        self,
        mask: np.ndarray,
        yolk_mask: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        back_debug: dict = {}

        if yolk_mask is None or np.sum(yolk_mask) == 0:
            warnings.warn(
                "No yolk mask available for back-direction computation. Returning mask COM as fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            fallback = self._center_of_mass(mask)
            back_debug["selected"] = "no_yolk_fallback"
            back_debug["back_yx"] = (float(fallback[0]), float(fallback[1]))
            self._last_back_debug = back_debug
            return fallback

        yolk_com_y, yolk_com_x = self._center_of_mass(yolk_mask)
        back_debug["yolk_com_yx"] = (float(yolk_com_y), float(yolk_com_x))

        yolk_area = float(yolk_mask.sum())
        r_yolk = np.sqrt(yolk_area / np.pi) if yolk_area > 0 else 0.0
        r_sample = float(self.back_sample_radius_k) * float(r_yolk)
        back_debug["r_yolk_px"] = float(r_yolk)
        back_debug["r_sample_px"] = float(r_sample)

        ys, xs = np.where(mask > 0.5)
        if ys.size == 0:
            fallback = (yolk_com_y, yolk_com_x)
            back_debug["selected"] = "empty_mask"
            back_debug["back_yx"] = (float(fallback[0]), float(fallback[1]))
            self._last_back_debug = back_debug
            return fallback

        dy = ys.astype(np.float64) - float(yolk_com_y)
        dx = xs.astype(np.float64) - float(yolk_com_x)
        in_disk = (dy**2 + dx**2) <= (r_sample**2)
        n_pixels_in_disk = int(in_disk.sum())
        back_debug["n_pixels_in_disk"] = n_pixels_in_disk

        if n_pixels_in_disk == 0:
            warnings.warn(
                f"No embryo-mask pixels within sampling disk (r_sample={r_sample:.1f}px). Using yolk COM.",
                RuntimeWarning,
                stacklevel=2,
            )
            back_debug["selected"] = "empty_disk"
            back_debug["back_yx"] = (float(yolk_com_y), float(yolk_com_x))
            self._last_back_debug = back_debug
            return (yolk_com_y, yolk_com_x)

        if n_pixels_in_disk < 50:
            warnings.warn(
                f"Only {n_pixels_in_disk} embryo-mask pixels in sampling disk; result may be noisy.",
                RuntimeWarning,
                stacklevel=2,
            )

        back_centroid_y = float(ys[in_disk].mean())
        back_centroid_x = float(xs[in_disk].mean())
        back_debug["raw_back_centroid_yx"] = (back_centroid_y, back_centroid_x)

        back_y, back_x = self._project_point_to_mask_in_disk(
            mask,
            (back_centroid_y, back_centroid_x),
            disk_center_yx=(yolk_com_y, yolk_com_x),
            disk_radius=r_sample,
        )
        back_debug["selected"] = "yolk_surrounding_centroid"
        back_debug["back_yx"] = (float(back_y), float(back_x))
        self._last_back_debug = back_debug
        return (back_y, back_x)

    def _coarse_candidate_select(
        self,
        mask: np.ndarray,
        yolk: Optional[np.ndarray],
        rotation_needed: float,
        scale: float,
        cx: float,
        cy: float,
        *,
        use_yolk: bool,
    ) -> tuple[float, bool, tuple[float, float], tuple[float, float]]:
        """Return (final_rotation_deg, flip, yolk_yx, back_yx).

        Two-step orientation selection (landscape + yolk present):

        Step 1 — AP axis (yolk left/right):
            Among all 4 candidates (rot_add ∈ {0,180} × flip ∈ {F,T}), pick the
            one with the smallest yolk_yx[1] (yolk as far left as possible).
            This fixes the head→tail direction on the horizontal axis.

        Step 2 — DV axis (back up/down):
            Compute the back-direction vector: back_vec_y = back_yx[0] - yolk_yx[0].
            back_vec_y < 0 means back is ABOVE yolk → dorsal-up (correct).
            back_vec_y > 0 means back is BELOW yolk → needs a vertical flip.
            A vertical flip = rot+180 composed with an L-R flip, so the partner
            candidate is (rot_add ± 180, not flip).  We look it up from the
            already-computed candidate table — no extra warp needed.

        Fallback (no yolk): original diagonal scoring unchanged.
        """
        # Build candidate table keyed by (rot_add, do_flip)
        candidate_table: dict[tuple[int, bool], tuple[np.ndarray, Optional[np.ndarray], tuple[float, float], tuple[float, float]]] = {}
        rot_options = [0, 180]
        flip_options = [False, True] if self.allow_flip else [False]

        for rot_add in rot_options:
            for do_flip in flip_options:
                M = cv2.getRotationMatrix2D((cx, cy), rotation_needed + rot_add, scale)
                M[0, 2] += (self.W / 2) - cx
                M[1, 2] += (self.H / 2) - cy
                mask_w = self._warp(mask, M)
                yolk_w = self._warp(yolk, M) if yolk is not None else None
                if do_flip:
                    mask_w = cv2.flip(mask_w, 1)
                    if yolk_w is not None:
                        yolk_w = cv2.flip(yolk_w, 1)

                yolk_feature_mask = (
                    yolk_w
                    if (use_yolk and yolk_w is not None and yolk_w.sum() > 0)
                    else mask_w
                )
                yolk_yx = self._center_of_mass(yolk_feature_mask)
                back_yx = self._compute_back_direction(mask_w, yolk_mask=yolk_w if use_yolk else None)
                candidate_table[(rot_add, do_flip)] = (mask_w, yolk_w, yolk_yx, back_yx)

        has_yolk = use_yolk and yolk is not None and any(
            yolk_w is not None and yolk_w.sum() > 0
            for (_, yolk_w, _, _) in candidate_table.values()
        )

        if has_yolk:
            # --- Step 1: pick candidate with yolk farthest LEFT (smallest yolk_x) ---
            best_key = min(candidate_table.keys(), key=lambda k: candidate_table[k][2][1])
            _, _, best_yolk_yx, best_back_yx = candidate_table[best_key]
            best_rot, best_flip = best_key

            # --- Step 2: enforce back ABOVE yolk (back_vec_y < 0) ---
            # back_vec_y = back_yx[0] - yolk_yx[0]; > 0 means back is below yolk
            back_vec_y = best_back_yx[0] - best_yolk_yx[0]
            if back_vec_y > 0:
                # Vertical flip = rot+180 + toggle L-R flip.  Look up the partner.
                partner_key = ((best_rot + 180) % 360, not best_flip)
                if partner_key in candidate_table:
                    _, _, best_yolk_yx, best_back_yx = candidate_table[partner_key]
                    best_rot, best_flip = partner_key
        else:
            # Fallback (no yolk): original diagonal scoring
            best_score = None
            best_rot, best_flip = 0, False
            best_yolk_yx, best_back_yx = (0.0, 0.0), (0.0, 0.0)
            for (rot_add, do_flip), (_, _, yolk_yx, back_yx) in candidate_table.items():
                yolk_cost = yolk_yx[1] + yolk_yx[0]
                back_score = back_yx[1] + back_yx[0]
                score = (self.back_weight * back_score) - (self.yolk_weight * yolk_cost)
                if best_score is None or score > best_score:
                    best_score = score
                    best_rot, best_flip = rot_add, do_flip
                    best_yolk_yx, best_back_yx = yolk_yx, back_yx

        final_rotation = rotation_needed + best_rot
        return float(final_rotation), bool(best_flip), (float(best_yolk_yx[0]), float(best_yolk_yx[1])), (float(best_back_yx[0]), float(best_back_yx[1]))

    def _apply_anchor_shift(
        self,
        aligned_mask: np.ndarray,
        aligned_yolk: Optional[np.ndarray],
        *,
        use_yolk: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray], float, float, bool, bool, np.ndarray]:
        """Shift so feature COM lands at self.anchor_point_xy.

        Returns shifted_mask, shifted_yolk, shift_x, shift_y, clamped, fit_impossible, M_shift.
        """
        if self.anchor_mode == "yolk_anchor" and use_yolk and aligned_yolk is not None and aligned_yolk.sum() > 0:
            feat_mask = aligned_yolk
        else:
            feat_mask = aligned_mask

        cur_cy, cur_cx = self._center_of_mass(feat_mask)
        desired_shift_x = self.anchor_point_xy[0] - cur_cx
        desired_shift_y = self.anchor_point_xy[1] - cur_cy

        bbox = self._bbox(aligned_mask)
        shift_x = desired_shift_x
        shift_y = desired_shift_y
        clamped = False
        fit_impossible = False
        if bbox is not None:
            min_y, max_y, min_x, max_x = bbox
            min_shift_x = -min_x
            max_shift_x = (self.W - 1) - max_x
            min_shift_y = -min_y
            max_shift_y = (self.H - 1) - max_y
            if min_shift_x <= max_shift_x:
                shift_x = float(np.clip(desired_shift_x, min_shift_x, max_shift_x))
            else:
                fit_impossible = True
            if min_shift_y <= max_shift_y:
                shift_y = float(np.clip(desired_shift_y, min_shift_y, max_shift_y))
            else:
                fit_impossible = True
            clamped = (shift_x != desired_shift_x) or (shift_y != desired_shift_y)

        M_shift = np.float32([[1, 0, float(shift_x)], [0, 1, float(shift_y)]])
        shifted_mask = self._warp(aligned_mask, M_shift)
        shifted_yolk = self._warp(aligned_yolk, M_shift) if aligned_yolk is not None else None
        return shifted_mask, shifted_yolk, float(shift_x), float(shift_y), bool(clamped), bool(fit_impossible), M_shift

    def _validate_output_mask(
        self,
        final_mask: np.ndarray,
        original_mask: np.ndarray,
        *,
        scale: float,
        final_rotation: float,
        best_flip: bool,
        shift_x: float,
        shift_y: float,
        fit_impossible: bool,
        retained_ratio: float,
    ) -> None:
        if final_mask.sum() == 0:
            raise RuntimeError(
                f"CanonicalAligner produced EMPTY mask. input_px={original_mask.sum()} retained_ratio={retained_ratio:.4f} "
                f"scale={scale:.3f} rotation={final_rotation:.1f} flip={best_flip} shift=({shift_x:.1f},{shift_y:.1f}) "
                f"fit_impossible={fit_impossible}"
            )
        touches = (
            bool(final_mask[0, :].any())
            or bool(final_mask[-1, :].any())
            or bool(final_mask[:, 0].any())
            or bool(final_mask[:, -1].any())
        )
        if touches:
            raise RuntimeError(
                "CanonicalAligner produced mask touching grid edge; check orientation/scale/anchor."
            )

    def generic_canonical_alignment(
        self,
        mask: np.ndarray,
        original_um_per_px: float,
        *,
        use_pca: bool = True,
        return_debug: bool = False,
    ) -> tuple[np.ndarray, dict, TransformChain]:
        if mask is None or mask.sum() == 0:
            empty = np.zeros((self.H, self.W), dtype=np.uint8)
            chain = TransformChain.identity(shape_yx=(self.H, self.W), interp="nearest")
            return empty, {"error": "empty_mask"}, chain

        scale = float(original_um_per_px) / float(self.target_res)
        angle_deg, centroid_xy, pca_used = self._pca_angle_deg(mask) if use_pca else (0.0, (0.0, 0.0), False)
        cx, cy = centroid_xy
        target_angle = 0.0 if self.is_landscape else 90.0
        rotation_needed = float(angle_deg) - float(target_angle)

        candidates = []
        rot_options = [0, 180]
        flip_options = [False, True] if self.allow_flip else [False]
        for rot_add in rot_options:
            for do_flip in flip_options:
                M = cv2.getRotationMatrix2D((cx, cy), rotation_needed + rot_add, scale)
                M[0, 2] += (self.W / 2) - cx
                M[1, 2] += (self.H / 2) - cy
                mask_w = self._warp(mask, M)
                if do_flip:
                    mask_w = cv2.flip(mask_w, 1)
                mask_com_yx = self._center_of_mass(mask_w)
                score = -(mask_com_yx[0] + mask_com_yx[1])
                candidates.append((score, rot_add, do_flip))

        _best_score, best_rot, best_flip = max(candidates, key=lambda x: x[0])
        final_rotation = rotation_needed + best_rot

        M_final = cv2.getRotationMatrix2D((cx, cy), final_rotation, scale)
        M_final[0, 2] += (self.W / 2) - cx
        M_final[1, 2] += (self.H / 2) - cy
        aligned_mask = self._warp(mask, M_final)
        if best_flip:
            aligned_mask = cv2.flip(aligned_mask, 1)

        aligned_mask_pre_shift = aligned_mask.copy()
        aligned_angle_deg, _, _ = self._pca_angle_deg(aligned_mask_pre_shift)

        aligned_mask, _aligned_yolk, shift_x, shift_y, clamped, fit_impossible, M_shift = self._apply_anchor_shift(
            aligned_mask,
            None,
            use_yolk=False,
        )

        expected_area = float(mask.sum()) * (scale ** 2)
        final_area = float(aligned_mask.sum())
        retained_ratio = final_area / max(expected_area, 1e-6)
        clipped = retained_ratio < self.clipping_threshold
        if clipped:
            message = f"generic_canonical_alignment clipped mask: retained_ratio={retained_ratio:.4f} (threshold={self.clipping_threshold:.4f})"
            if self.error_on_clip:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        final_mask = (aligned_mask > 0.5).astype(np.uint8)
        self._validate_output_mask(
            final_mask,
            mask,
            scale=scale,
            final_rotation=float(final_rotation),
            best_flip=bool(best_flip),
            shift_x=float(shift_x),
            shift_y=float(shift_y),
            fit_impossible=bool(fit_impossible),
            retained_ratio=float(retained_ratio),
        )

        meta = {
            "pca_angle_deg": float(angle_deg),
            "rotation_needed_deg": float(rotation_needed),
            "rotation_deg": float(final_rotation),
            "flip": bool(best_flip),
            "scale": float(scale),
            "pca_used": bool(pca_used),
            "aligned_pca_angle_deg": float(aligned_angle_deg),
            "anchor_xy": (float(self.anchor_point_xy[0]), float(self.anchor_point_xy[1])),
            "anchor_shift_xy": (float(shift_x), float(shift_y)),
            "anchor_shift_clamped": bool(clamped),
            "fit_impossible": bool(fit_impossible),
            "retained_ratio": float(retained_ratio),
            "clipped": bool(clipped),
            "yolk_used": False,
            "anchor_yx": (float(self.anchor_point_xy[1]), float(self.anchor_point_xy[0])),
            "anchor_shift_dyx": (float(shift_y), float(shift_x)),
        }
        if return_debug:
            meta["debug"] = {"aligned_mask_pre_shift": aligned_mask_pre_shift}

        chain = _build_stage1_chain(
            grid_shape_yx=(self.H, self.W),
            M_final=M_final,
            best_flip=bool(best_flip),
            M_shift=M_shift,
        )
        return final_mask, meta, chain

    def embryo_canonical_alignment(
        self,
        mask: np.ndarray,
        original_um_per_px: float,
        *,
        yolk: Optional[np.ndarray] = None,
        use_pca: bool = True,
        return_debug: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray], dict, TransformChain]:
        if mask is None or mask.sum() == 0:
            empty = np.zeros((self.H, self.W), dtype=np.uint8)
            chain = TransformChain.identity(shape_yx=(self.H, self.W), interp="nearest")
            return empty, None, {"error": "empty_mask", "yolk_com_yx": None}, chain

        if yolk is None or (hasattr(yolk, "sum") and yolk.sum() == 0):
            warnings.warn(
                "embryo_canonical_alignment: yolk mask is None or empty. Falling back to generic_canonical_alignment().",
                RuntimeWarning,
                stacklevel=2,
            )
            canonical_mask, meta, chain = self.generic_canonical_alignment(mask, original_um_per_px, use_pca=use_pca, return_debug=return_debug)
            meta["yolk_used"] = False
            meta["yolk_com_yx"] = None
            return canonical_mask, None, meta, chain

        scale = float(original_um_per_px) / float(self.target_res)
        angle_deg, centroid_xy, pca_used = self._pca_angle_deg(mask) if use_pca else (0.0, (0.0, 0.0), False)
        cx, cy = centroid_xy
        target_angle = 0.0 if self.is_landscape else 90.0
        rotation_needed = float(angle_deg) - float(target_angle)

        final_rotation, best_flip, best_yolk_yx, best_back_yx = self._coarse_candidate_select(
            mask, yolk, rotation_needed, scale, cx, cy, use_yolk=True
        )

        M_final = cv2.getRotationMatrix2D((cx, cy), final_rotation, scale)
        M_final[0, 2] += (self.W / 2) - cx
        M_final[1, 2] += (self.H / 2) - cy
        aligned_mask = self._warp(mask, M_final)
        aligned_yolk = self._warp(yolk, M_final)
        if best_flip:
            aligned_mask = cv2.flip(aligned_mask, 1)
            aligned_yolk = cv2.flip(aligned_yolk, 1)

        aligned_mask_pre_shift = aligned_mask.copy()
        aligned_yolk_pre_shift = aligned_yolk.copy()
        aligned_angle_deg, _, _ = self._pca_angle_deg(aligned_mask_pre_shift)

        aligned_mask, aligned_yolk, shift_x, shift_y, clamped, fit_impossible, M_shift = self._apply_anchor_shift(
            aligned_mask,
            aligned_yolk,
            use_yolk=True,
        )

        expected_area = float(mask.sum()) * (scale ** 2)
        final_area = float(aligned_mask.sum())
        retained_ratio = final_area / max(expected_area, 1e-6)
        clipped = retained_ratio < self.clipping_threshold
        if clipped:
            message = f"embryo_canonical_alignment clipped mask: retained_ratio={retained_ratio:.4f} (threshold={self.clipping_threshold:.4f})"
            if self.error_on_clip:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        final_mask = (aligned_mask > 0.5).astype(np.uint8)
        self._validate_output_mask(
            final_mask,
            mask,
            scale=scale,
            final_rotation=float(final_rotation),
            best_flip=bool(best_flip),
            shift_x=float(shift_x),
            shift_y=float(shift_y),
            fit_impossible=bool(fit_impossible),
            retained_ratio=float(retained_ratio),
        )
        final_yolk_mask = (aligned_yolk > 0.5).astype(np.uint8) if aligned_yolk is not None else None

        yolk_feature_mask = aligned_yolk if (aligned_yolk is not None and aligned_yolk.sum() > 0) else aligned_mask
        final_yolk_yx = self._center_of_mass(yolk_feature_mask)
        final_back_yx = self._compute_back_direction(aligned_mask, yolk_mask=aligned_yolk)

        meta = {
            "pca_angle_deg": float(angle_deg),
            "rotation_needed_deg": float(rotation_needed),
            "rotation_deg": float(final_rotation),
            "flip": bool(best_flip),
            "scale": float(scale),
            "pca_used": bool(pca_used),
            "aligned_pca_angle_deg": float(aligned_angle_deg),
            "anchor_xy": (float(self.anchor_point_xy[0]), float(self.anchor_point_xy[1])),
            "anchor_shift_xy": (float(shift_x), float(shift_y)),
            "anchor_shift_clamped": bool(clamped),
            "fit_impossible": bool(fit_impossible),
            "yolk_yx_pre_shift": (float(best_yolk_yx[0]), float(best_yolk_yx[1])),
            "back_yx_pre_shift": (float(best_back_yx[0]), float(best_back_yx[1])),
            "yolk_yx_final": (float(final_yolk_yx[0]), float(final_yolk_yx[1])),
            "back_yx_final": (float(final_back_yx[0]), float(final_back_yx[1])),
            "retained_ratio": float(retained_ratio),
            "clipped": bool(clipped),
            "yolk_used": True,
            "yolk_com_yx": (float(final_yolk_yx[0]), float(final_yolk_yx[1])),
            "anchor_yx": (float(self.anchor_point_xy[1]), float(self.anchor_point_xy[0])),
            "anchor_shift_dyx": (float(shift_y), float(shift_x)),
        }
        if return_debug:
            meta["debug"] = {
                "aligned_mask_pre_shift": aligned_mask_pre_shift,
                "aligned_yolk_pre_shift": aligned_yolk_pre_shift,
                "back_direction": copy.deepcopy(self._last_back_debug),
            }

        chain = _build_stage1_chain(
            grid_shape_yx=(self.H, self.W),
            M_final=M_final,
            best_flip=bool(best_flip),
            M_shift=M_shift,
        )
        return final_mask, final_yolk_mask, meta, chain

    def align(
        self,
        mask: np.ndarray,
        yolk: Optional[np.ndarray],
        original_um_per_px: float,
        *,
        use_pca: bool = True,
        use_yolk: bool = True,
        reference_mask: Optional[np.ndarray] = None,
        return_debug: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
        """Deprecated legacy wrapper. `reference_mask` is forbidden."""
        warnings.warn(
            "CanonicalAligner.align() is deprecated. Prefer to_canonical_grid_* or embryo_canonical_alignment().",
            DeprecationWarning,
            stacklevel=2,
        )
        if reference_mask is not None:
            raise ValueError(
                "align(reference_mask=...) is not supported. Use analyze.utils.coord.register.register_to_fixed(...) explicitly."
            )
        if not use_yolk:
            canonical_mask, meta, _chain = self.generic_canonical_alignment(mask, original_um_per_px, use_pca=use_pca, return_debug=return_debug)
            return canonical_mask, None, meta
        canonical_mask, canonical_yolk, meta, _chain = self.embryo_canonical_alignment(mask, original_um_per_px, yolk=yolk, use_pca=use_pca, return_debug=return_debug)
        return canonical_mask, canonical_yolk, meta


def _build_stage1_chain(
    *,
    grid_shape_yx: tuple[int, int],
    M_final: np.ndarray,
    best_flip: bool,
    M_shift: np.ndarray,
) -> TransformChain:
    h, w = grid_shape_yx
    transforms = [
        GridTransform(
            name="rotate_scale_center",
            affine_2x3=np.asarray(M_final, dtype=np.float64),
            in_shape_yx=grid_shape_yx,
            out_shape_yx=grid_shape_yx,
            interp="nearest",
            params={"affine_convention": "opencv_xy"},
        )
    ]
    if best_flip:
        transforms.append(
            GridTransform(
                name="flip_x",
                affine_2x3=np.array([[ -1.0, 0.0, float(w - 1)], [0.0, 1.0, 0.0]], dtype=np.float64),
                in_shape_yx=grid_shape_yx,
                out_shape_yx=grid_shape_yx,
                interp="nearest",
                params={"affine_convention": "opencv_xy"},
            )
        )
    transforms.append(
        GridTransform(
            name="translate_anchor",
            affine_2x3=np.asarray(M_shift, dtype=np.float64),
            in_shape_yx=grid_shape_yx,
            out_shape_yx=grid_shape_yx,
            interp="nearest",
            params={"affine_convention": "opencv_xy"},
        )
    )
    return TransformChain(transforms=transforms)


class CanonicalGridMapper:
    """Internal engine; `to_canonical_grid_*` are stable convenience wrappers."""

    def __init__(self, *, grid: CanonicalGrid, cfg: CanonicalGridConfig):
        self.grid = grid
        self.cfg = cfg
        self.aligner = CanonicalAligner.from_config(cfg)

    @staticmethod
    def _default_grid(cfg: CanonicalGridConfig) -> CanonicalGrid:
        h, w = cfg.grid_shape_hw
        y_frac, x_frac = cfg.anchor_frac_yx
        anchor_y = float(h) * float(y_frac)
        anchor_x = float(w) * float(x_frac)
        return CanonicalGrid(
            um_per_px=float(cfg.reference_um_per_pixel),
            shape_yx=(int(h), int(w)),
            anchor_mode=str(cfg.anchor_mode),
            anchor_yx=(anchor_y, anchor_x),
        )

    def to_canonical_mask(
        self,
        mask: np.ndarray,
        *,
        um_per_px: float,
        yolk_mask: Optional[np.ndarray] = None,
    ) -> CanonicalMaskResult:
        if self.cfg.assume_canonical:
            chain = TransformChain.identity(shape_yx=self.grid.shape_yx, interp="nearest")
            meta = {
                "coord_frame_id": "canonical_grid",
                "coord_frame_version": 1,
                "coord_convention": "yx",
                "canonical_grid": {
                    "um_per_px": self.grid.um_per_px,
                    "shape_yx": list(self.grid.shape_yx),
                    "anchor_mode": self.grid.anchor_mode,
                    "anchor_yx": list(self.grid.anchor_yx),
                },
                "to_canonical_grid": {"applied": False, "checks": {"assume_canonical": True}, "decisions": {}},
            }
            return CanonicalMaskResult(mask=np.asarray(mask).astype(np.uint8), grid=self.grid, transform_chain=chain, meta=meta)

        use_pca = self.cfg.align_mode != "none"
        use_yolk = self.cfg.align_mode == "yolk"
        if use_yolk:
            can, can_yolk, align_meta, chain = self.aligner.embryo_canonical_alignment(
                mask.astype(bool),
                float(um_per_px),
                yolk=yolk_mask.astype(bool) if yolk_mask is not None else None,
                use_pca=use_pca,
            )
            out_mask = can
            yolk_com = align_meta.get("yolk_com_yx")
        else:
            out_mask, align_meta, chain = self.aligner.generic_canonical_alignment(
                mask.astype(bool),
                float(um_per_px),
                use_pca=use_pca,
            )
            yolk_com = None

        if isinstance(align_meta, dict) and align_meta.get("error"):
            # Empty/failed alignment should not pretend a geometric mapping occurred.
            chain = TransformChain.identity(shape_yx=self.grid.shape_yx, interp="nearest")
            applied = False
            checks = {"error": str(align_meta.get("error"))}
        else:
            applied = True
            checks = {}

        meta = {
            "coord_frame_id": "canonical_grid",
            "coord_frame_version": 1,
            "coord_convention": "yx",
            "canonical_grid": {
                "um_per_px": self.grid.um_per_px,
                "shape_yx": list(self.grid.shape_yx),
                "anchor_mode": self.grid.anchor_mode,
                "anchor_yx": list(self.grid.anchor_yx),
            },
            "to_canonical_grid": {"applied": bool(applied), "checks": checks, "decisions": {}},
            "yolk_com_yx": yolk_com,
            "align_meta": align_meta,
        }
        return CanonicalMaskResult(mask=out_mask.astype(np.uint8), grid=self.grid, transform_chain=chain, meta=meta)


def to_canonical_grid_mask(
    mask: np.ndarray,
    *,
    um_per_px: float,
    yolk_mask: Optional[np.ndarray] = None,
    grid: Optional[CanonicalGrid] = None,
    cfg: Optional[CanonicalGridConfig] = None,
) -> CanonicalMaskResult:
    cfg = cfg or CanonicalGridConfig()
    if grid is None:
        grid = CanonicalGridMapper._default_grid(cfg)
    mapper = CanonicalGridMapper(grid=grid, cfg=cfg)
    return mapper.to_canonical_mask(mask, um_per_px=um_per_px, yolk_mask=yolk_mask)


def to_canonical_grid_image(
    image: np.ndarray,
    *,
    um_per_px: float,
    grid: Optional[CanonicalGrid] = None,
    cfg: Optional[CanonicalGridConfig] = None,
    interpolation: str = "linear",
) -> CanonicalImageResult:
    cfg = cfg or CanonicalGridConfig()
    if grid is None:
        grid = CanonicalGridMapper._default_grid(cfg)
    h_out, w_out = grid.shape_yx
    img = np.asarray(image)

    if cfg.assume_canonical and img.shape[:2] == (h_out, w_out):
        chain = TransformChain.identity(shape_yx=(h_out, w_out), interp="linear")
        meta = {
            "coord_frame_id": "canonical_grid",
            "coord_frame_version": 1,
            "coord_convention": "yx",
            "canonical_grid": {
                "um_per_px": grid.um_per_px,
                "shape_yx": list(grid.shape_yx),
                "anchor_mode": grid.anchor_mode,
                "anchor_yx": list(grid.anchor_yx),
            },
            "to_canonical_grid": {"applied": False, "checks": {"assume_canonical": True}, "decisions": {}},
        }
        return CanonicalImageResult(image=img, grid=grid, transform_chain=chain, meta=meta)

    if cv2 is None:
        raise ImportError("cv2 is required for to_canonical_grid_image.")

    # Image-only mapping is restricted: scale to canonical um/px and center into the canvas.
    scale = float(um_per_px) / float(grid.um_per_px)
    h_in, w_in = img.shape[:2]
    cx, cy = (w_in / 2.0), (h_in / 2.0)
    M = cv2.getRotationMatrix2D((cx, cy), 0.0, scale)
    M[0, 2] += (w_out / 2.0) - cx
    M[1, 2] += (h_out / 2.0) - cy

    flags = cv2.INTER_LINEAR if interpolation == "linear" else cv2.INTER_NEAREST
    out_img = cv2.warpAffine(img.astype(np.float32), M.astype(np.float32), (w_out, h_out), flags=flags)

    chain = TransformChain(
        transforms=[
            GridTransform(
                name="scale_center",
                affine_2x3=np.asarray(M, dtype=np.float64),
                in_shape_yx=(h_in, w_in),
                out_shape_yx=(h_out, w_out),
                interp="linear" if interpolation == "linear" else "nearest",
                params={"scale": float(scale), "affine_convention": "opencv_xy"},
            )
        ]
    )
    meta = {
        "coord_frame_id": "canonical_grid",
        "coord_frame_version": 1,
        "coord_convention": "yx",
        "canonical_grid": {
            "um_per_px": grid.um_per_px,
            "shape_yx": list(grid.shape_yx),
            "anchor_mode": grid.anchor_mode,
            "anchor_yx": list(grid.anchor_yx),
        },
        "to_canonical_grid": {"applied": True, "checks": {"image_only": True}, "decisions": {}},
    }
    return CanonicalImageResult(image=out_img, grid=grid, transform_chain=chain, meta=meta)


def to_canonical_grid_frame(
    frame: Frame,
    *,
    grid: Optional[CanonicalGrid] = None,
    cfg: Optional[CanonicalGridConfig] = None,
) -> CanonicalFrameResult:
    cfg = cfg or CanonicalGridConfig()
    meta_in = frame.meta or {}
    um_per_px = float(frame.um_per_px)
    if not np.isfinite(um_per_px):
        um_meta = meta_in.get("um_per_px")
        if um_meta is None or not np.isfinite(float(um_meta)):
            raise ValueError("Frame.um_per_px is required (or provide frame.meta['um_per_px']).")
        um_per_px = float(um_meta)

    # No-op detection based on labeled frames.
    if meta_in.get("coord_frame_id") == "canonical_grid" and int(meta_in.get("coord_frame_version", 0)) == 1:
        cfg = copy.copy(cfg)
        cfg.assume_canonical = True

    if frame.mask is not None:
        mask_res = to_canonical_grid_mask(
            frame.mask,
            um_per_px=um_per_px,
            yolk_mask=frame.yolk_mask,
            grid=grid,
            cfg=cfg,
        )
        grid = mask_res.grid
        chain = mask_res.transform_chain
        meta_out = dict(mask_res.meta)
        out_mask = mask_res.mask
        out_yolk = None
    else:
        # Image-only canonicalization (no segmentation-driven rotation/anchor decisions).
        if grid is None:
            grid = CanonicalGridMapper._default_grid(cfg)
        img_res = to_canonical_grid_image(frame.image, um_per_px=um_per_px, grid=grid, cfg=cfg) if frame.image is not None else None
        chain = img_res.transform_chain if img_res is not None else TransformChain.identity(shape_yx=grid.shape_yx, interp="nearest")
        meta_out = dict(img_res.meta) if img_res is not None else {
            "coord_frame_id": "canonical_grid",
            "coord_frame_version": 1,
            "coord_convention": "yx",
            "canonical_grid": {
                "um_per_px": grid.um_per_px,
                "shape_yx": list(grid.shape_yx),
                "anchor_mode": grid.anchor_mode,
                "anchor_yx": list(grid.anchor_yx),
            },
            "to_canonical_grid": {"applied": False, "checks": {"identity": True}, "decisions": {}},
        }
        out_mask = None
        out_yolk = None

    out_img = None
    if frame.image is not None:
        # If we have a segmentation-derived chain, apply it; otherwise reuse image-only mapping.
        if frame.mask is not None:
            out_img = TransformChain(
                transforms=[
                    GridTransform(
                        name=t.name,
                        affine_2x3=t.affine_2x3,
                        in_shape_yx=t.in_shape_yx,
                        out_shape_yx=t.out_shape_yx,
                        interp="linear",
                        params=dict(t.params),
                    )
                    for t in chain.transforms
                ]
            ).apply_to_image(np.asarray(frame.image))
        else:
            out_img = to_canonical_grid_image(frame.image, um_per_px=um_per_px, grid=grid, cfg=cfg).image

    out_frame = Frame(
        image=out_img,
        mask=out_mask,
        yolk_mask=out_yolk,
        um_per_px=grid.um_per_px,
        meta=dict(meta_in),
    )
    return CanonicalFrameResult(frame=out_frame, grid=grid, transform_chain=chain, meta=meta_out)
