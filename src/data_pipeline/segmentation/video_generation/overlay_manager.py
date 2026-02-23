"""Overlay rendering for segmentation evaluation videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .mask_decoding import decode_mask
from .models import AnnotationRecord
from .video_config import COLORBLIND_PALETTE, OVERLAY_COLORS, VideoConfig


@dataclass
class OverlayConfig:
    color: tuple[int, int, int]
    thickness: int = 2
    alpha: float = 0.8
    font_scale: float = 0.7
    position: str = "auto"


class OverlayManager:
    """Renders object masks, boxes, labels, and QC overlays."""

    def __init__(self, video_config: VideoConfig):
        self.config = video_config
        self.overlay_configs = {
            "detection": OverlayConfig(
                color=OVERLAY_COLORS["detection"],
                thickness=self.config.BBOX_THICKNESS,
                alpha=self.config.BBOX_ALPHA,
            ),
            "mask": OverlayConfig(
                color=OVERLAY_COLORS["mask"],
                thickness=self.config.MASK_OUTLINE_THICKNESS,
                alpha=self.config.MASK_ALPHA,
            ),
            "metadata": OverlayConfig(
                color=OVERLAY_COLORS["metadata"],
                thickness=2,
                alpha=0.9,
                position="bottom_left",
            ),
            "qc_flags": OverlayConfig(
                color=OVERLAY_COLORS["qc_good"],
                thickness=2,
                alpha=0.9,
                position="top_left",
            ),
        }

    def add_generic_annotations_overlay(
        self,
        frame: np.ndarray,
        annotations: list[AnnotationRecord],
        *,
        source_format: str,
        show_bbox: bool = True,
        show_mask: bool = True,
        show_metrics: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame passed to overlay manager")
        if not annotations:
            return frame

        frame_h, frame_w = frame.shape[:2]

        for ann in annotations:
            color = self._get_annotation_color(ann)
            warning_color = OVERLAY_COLORS["qc_warning"]

            mask = None
            if show_mask and ann.segmentation is not None:
                try:
                    mask = decode_mask(
                        ann.segmentation,
                        source_format=source_format,
                        height=frame_h,
                        width=frame_w,
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Failed to decode mask for annotation {ann.annotation_id}: {exc}"
                    ) from exc

            x1, y1, x2, y2 = self._bbox_to_xyxy(ann, frame_w=frame_w, frame_h=frame_h)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            mask_area = int(np.sum(mask)) if mask is not None else 0
            bbox_area = max(1, w * h)
            fill_ratio = float(mask_area) / float(bbox_area)
            score = ann.score if ann.score is not None else 0.0

            # Keep same quality heuristic used before.
            is_good_quality = mask is None or (0.3 <= fill_ratio <= 0.9 and mask_area >= 500)
            draw_color = color if is_good_quality else warning_color

            if show_mask and mask is not None:
                if mask.shape[:2] != frame.shape[:2]:
                    raise ValueError(
                        f"Mask shape {mask.shape} incompatible with frame {frame.shape}"
                    )
                mask_pixels = mask > 0
                alpha = self.config.MASK_ALPHA
                frame[mask_pixels] = (
                    frame[mask_pixels] * (1 - alpha)
                    + np.asarray(draw_color, dtype=np.float32) * alpha
                ).astype(frame.dtype)

                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(
                    frame,
                    contours,
                    -1,
                    draw_color,
                    self.config.MASK_OUTLINE_THICKNESS,
                )

            if show_bbox and w > 0 and h > 0:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    draw_color,
                    self.config.BBOX_THICKNESS,
                )

            if show_labels:
                label = ann.label or ann.annotation_id
                label = self._format_label(label)
                self._add_text_with_background(
                    frame,
                    label,
                    (x1 + 8, y1 + 24),
                    draw_color,
                    scale=0.9,
                )

            if show_metrics:
                lines = []
                if ann.score is not None:
                    lines.append(f"Score: {score:.2f}")
                if mask is not None:
                    lines.append(f"Fill: {fill_ratio:.2f}")
                    lines.append(f"Area: {mask_area}")

                text_y = y1 + 42
                for line in lines:
                    if text_y + 16 > y2 and y2 > y1:
                        break
                    self._add_text_with_background(
                        frame,
                        line,
                        (x1 + 8, text_y),
                        draw_color,
                        scale=0.5,
                    )
                    text_y += 18

        return frame

    # Backward-compatible API used by older SAM2 scripts.
    def add_sam2_embryos_overlay(
        self,
        frame: np.ndarray,
        embryos_data: dict[str, Any],
        show_bbox: bool = True,
        show_mask: bool = True,
        show_metrics: bool = True,
        show_embryo_id: bool = True,
        min_fill_ratio: float = 0.3,
        max_fill_ratio: float = 0.9,
        min_area_px: int = 500,
    ) -> np.ndarray:
        _ = (show_embryo_id, min_fill_ratio, max_fill_ratio, min_area_px)
        annotations = []
        for emb_id, emb_data in embryos_data.items():
            seg = emb_data.get("segmentation")
            bbox = emb_data.get("bbox")
            if (not bbox) and isinstance(seg, dict):
                bbox = seg.get("bbox")
            annotations.append(
                AnnotationRecord(
                    annotation_id=str(emb_id),
                    segmentation=seg,
                    bbox=list(bbox) if bbox else None,
                    bbox_mode="xyxy_norm",
                    score=emb_data.get("mask_confidence"),
                    label=str(emb_id),
                    metadata={"area": emb_data.get("area")},
                )
            )
        return self.add_generic_annotations_overlay(
            frame,
            annotations,
            source_format="grounded_sam2",
            show_bbox=show_bbox,
            show_mask=show_mask,
            show_metrics=show_metrics,
            show_labels=True,
        )

    def add_qc_flags_overlay(self, frame: np.ndarray, qc_flags: list[str]) -> np.ndarray:
        if not qc_flags:
            return frame

        x_start, y_start = 20, 40
        for idx, flag in enumerate(qc_flags):
            if flag in {"BLUR", "DARK", "CORRUPT"}:
                color = OVERLAY_COLORS["qc_error"]
            elif flag in {"LOW_CONTRAST", "BRIGHT"}:
                color = OVERLAY_COLORS["qc_warning"]
            else:
                color = OVERLAY_COLORS["qc_good"]

            self._add_text_with_background(
                frame,
                f"QC: {flag}",
                (x_start, y_start + idx * 24),
                color,
                scale=0.6,
            )
        return frame

    def _bbox_to_xyxy(self, ann: AnnotationRecord, *, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        if not ann.bbox or len(ann.bbox) != 4:
            return 0, 0, 0, 0

        mode = (ann.bbox_mode or "xyxy_norm").lower()
        x1, y1, x2, y2 = 0, 0, 0, 0

        if mode == "xyxy_norm":
            x1 = int(float(ann.bbox[0]) * frame_w)
            y1 = int(float(ann.bbox[1]) * frame_h)
            x2 = int(float(ann.bbox[2]) * frame_w)
            y2 = int(float(ann.bbox[3]) * frame_h)
        elif mode == "xyxy_abs":
            x1, y1, x2, y2 = [int(round(float(v))) for v in ann.bbox]
        elif mode == "xywh_abs":
            x, y, w, h = [float(v) for v in ann.bbox]
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
        else:
            raise ValueError(f"Unsupported bbox_mode: {ann.bbox_mode}")

        x1 = max(0, min(frame_w - 1, x1)) if frame_w > 0 else 0
        y1 = max(0, min(frame_h - 1, y1)) if frame_h > 0 else 0
        x2 = max(0, min(frame_w - 1, x2)) if frame_w > 0 else 0
        y2 = max(0, min(frame_h - 1, y2)) if frame_h > 0 else 0

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return x1, y1, x2, y2

    def _add_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.7,
    ) -> None:
        x, y = position
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.config.FONT,
            scale,
            self.config.FONT_THICKNESS,
        )

        cv2.rectangle(
            frame,
            (x - 3, y - text_height - 3),
            (x + text_width + 3, y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            self.config.FONT,
            scale,
            color,
            self.config.FONT_THICKNESS,
        )

    def _get_annotation_color(self, ann: AnnotationRecord) -> tuple[int, int, int]:
        colors = list(COLORBLIND_PALETTE.values())
        # Prefer category_id for stability across runs when available.
        if ann.category_id is not None:
            return colors[int(ann.category_id) % len(colors)]
        return colors[hash(ann.annotation_id) % len(colors)]

    def _format_label(self, value: str) -> str:
        m = re_search_embryo_suffix(value)
        if m is not None:
            return f"e{m:02d}"
        parts = value.split("_")
        return parts[-1] if len(parts) > 1 else value


_EMBRYO_SUFFIX = r"_e(\d+)$"


def re_search_embryo_suffix(text: str) -> int | None:
    import re

    match = re.search(_EMBRYO_SUFFIX, text)
    if not match:
        return None
    return int(match.group(1))
