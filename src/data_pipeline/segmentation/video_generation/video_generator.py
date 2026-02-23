"""Generic video generator for segmentation overlays.

Supports multiple annotation sources through adapter parsing:
- GroundedSAM2 nested JSON
- COCO JSON (Detectron2-compatible)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .overlay_manager import OverlayManager
from .results_adapter import load_video_record
from .video_config import VideoConfig


class VideoGenerator:
    """Creates videos with segmentation overlays from canonical frame records."""

    def __init__(self, config: VideoConfig | None = None):
        self.config = config or VideoConfig()
        self.overlay_manager = OverlayManager(self.config)

    def create_eval_video_from_results(
        self,
        *,
        results_json_path: Path,
        experiment_id: str,
        video_id: str,
        output_video_path: Path,
        source_format: str = "auto",
        images_root: Path | None = None,
        show_bbox: bool = False,
        show_mask: bool = True,
        show_metrics: bool = True,
        show_qc_flags: bool = False,
        show_labels: bool = True,
        verbose: bool = True,
    ) -> bool:
        """Render one video from an annotation JSON source."""
        if verbose:
            print(f"Rendering eval video for {video_id} ({source_format})")

        try:
            video_record = load_video_record(
                results_json_path,
                experiment_id=experiment_id,
                video_id=video_id,
                source_format=source_format,
                images_root=images_root,
            )
        except Exception as exc:
            if verbose:
                print(f"Failed to load video record: {exc}")
            return False

        if not video_record.frames:
            if verbose:
                print("No frames found for video")
            return False

        first = next((f for f in video_record.frames if f.image_path.exists()), None)
        if first is None:
            if verbose:
                print("No frame image files found on disk")
            return False

        first_frame = cv2.imread(str(first.image_path))
        if first_frame is None:
            if verbose:
                print(f"Could not read first frame: {first.image_path}")
            return False

        height, width = first_frame.shape[:2]
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.config.CODEC)
        writer = cv2.VideoWriter(str(output_video_path), fourcc, self.config.FPS, (width, height))
        if not writer.isOpened():
            if verbose:
                print(f"Could not open output video: {output_video_path}")
            return False

        frames_written = 0
        frames_with_overlays = 0

        effective_source = source_format
        if source_format == "auto":
            # Infer from first annotation container shape.
            effective_source = (
                "grounded_sam2"
                if any(f.annotations and isinstance(f.annotations[0].segmentation, dict) and "format" in (f.annotations[0].segmentation or {}) for f in video_record.frames)
                else "coco"
            )

        for frame_rec in sorted(video_record.frames, key=lambda f: (f.frame_index if f.frame_index is not None else 0, f.image_id)):
            frame = cv2.imread(str(frame_rec.image_path))
            if frame is None:
                if verbose:
                    print(f"Skipping unreadable frame: {frame_rec.image_path}")
                continue

            frame = self._add_image_id_overlay(frame, frame_rec.image_id)

            if frame_rec.annotations:
                frame = self.overlay_manager.add_generic_annotations_overlay(
                    frame,
                    frame_rec.annotations,
                    source_format=effective_source,
                    show_bbox=show_bbox,
                    show_mask=show_mask,
                    show_metrics=show_metrics,
                    show_labels=show_labels,
                )
                frames_with_overlays += 1

            if show_qc_flags and frame_rec.qc_flags:
                frame = self.overlay_manager.add_qc_flags_overlay(frame, frame_rec.qc_flags)

            writer.write(frame)
            frames_written += 1

        writer.release()

        if verbose:
            print(f"Video written: {output_video_path}")
            print(f"Frames written: {frames_written}; frames with overlays: {frames_with_overlays}")

        return frames_written > 0

    # Backward-compatible method name used across sandbox scripts.
    def create_sam2_eval_video_from_results(
        self,
        results_json_path: Path,
        experiment_id: str,
        video_id: str,
        output_video_path: Path,
        show_bbox: bool = False,
        show_mask: bool = True,
        show_metrics: bool = True,
        show_qc_flags: bool = False,
        verbose: bool = True,
    ) -> bool:
        return self.create_eval_video_from_results(
            results_json_path=results_json_path,
            experiment_id=experiment_id,
            video_id=video_id,
            output_video_path=output_video_path,
            source_format="grounded_sam2",
            show_bbox=show_bbox,
            show_mask=show_mask,
            show_metrics=show_metrics,
            show_qc_flags=show_qc_flags,
            show_labels=True,
            verbose=verbose,
        )

    def _add_image_id_overlay(self, frame: np.ndarray, image_id: str) -> np.ndarray:
        height, width = frame.shape[:2]

        (text_width, text_height), _ = cv2.getTextSize(
            image_id,
            self.config.FONT,
            self.config.FONT_SCALE,
            self.config.FONT_THICKNESS,
        )
        text_x = width - text_width - self.config.IMAGE_ID_MARGIN_RIGHT
        text_y = int(height * self.config.IMAGE_ID_Y_PERCENTAGE)

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + 5),
            self.config.TEXT_BACKGROUND_COLOR,
            -1,
        )
        frame = cv2.addWeighted(
            overlay,
            self.config.TEXT_BACKGROUND_ALPHA,
            frame,
            1 - self.config.TEXT_BACKGROUND_ALPHA,
            0,
        )

        cv2.putText(
            frame,
            image_id,
            (text_x, text_y),
            self.config.FONT,
            self.config.FONT_SCALE,
            self.config.TEXT_COLOR,
            self.config.FONT_THICKNESS,
        )
        return frame

    @staticmethod
    def extract_video_id_from_path(video_path: Path) -> str:
        return video_path.stem

    @staticmethod
    def generate_image_id(video_id: str, frame_number: int) -> str:
        return f"{video_id}_t{str(frame_number).zfill(4)}"
