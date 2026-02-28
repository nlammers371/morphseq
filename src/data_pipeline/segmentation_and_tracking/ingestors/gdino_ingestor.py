from __future__ import annotations

from pathlib import Path

from data_pipeline.segmentation.grounded_sam2.gdino_detection import (
    detect_embryos,
    filter_detections,
    select_seed_frame,
)

from ..raw_types import RawDetection, SeedSelection


def _clamp(val: float, lo: float, hi: float) -> float:
    return float(min(max(val, lo), hi))


def _norm_to_abs_xyxy(box_xyxy_norm: list[float], *, width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy_norm]
    x0a = _clamp(x0 * width, 0.0, float(width))
    x1a = _clamp(x1 * width, 0.0, float(width))
    y0a = _clamp(y0 * height, 0.0, float(height))
    y1a = _clamp(y1 * height, 0.0, float(height))
    return [x0a, y0a, x1a, y1a]


def ingest_detections(
    model,
    image_path: Path,
    *,
    frame_index: int,
    image_id: str,
    image_height_px: int,
    image_width_px: int,
    device: str = "cpu",
    text_prompt: str = "individual embryo",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    confidence_threshold: float = 0.45,
    iou_threshold: float = 0.5,
) -> list[RawDetection]:
    dets = detect_embryos(
        model=model,
        image_path=Path(image_path),
        device=str(device),
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    dets = filter_detections(dets, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold)

    raw: list[RawDetection] = []
    for d in dets:
        box_norm = [float(x) for x in d.get("box_xyxy", [0, 0, 0, 0])]
        conf = float(d.get("confidence", 0.0))
        phrase = d.get("phrase")
        raw.append(
            RawDetection(
                frame_index=int(frame_index),
                image_id=str(image_id),
                box_xyxy_norm=box_norm,
                confidence=conf,
                box_xyxy_abs=_norm_to_abs_xyxy(box_norm, width=int(image_width_px), height=int(image_height_px)),
                image_height_px=int(image_height_px),
                image_width_px=int(image_width_px),
                phrase=str(phrase) if phrase is not None else None,
            )
        )

    # Deterministic ordering to define detection_index.
    raw.sort(key=lambda r: (-float(r.confidence), float(r.box_xyxy_abs[0]), float(r.box_xyxy_abs[1])))
    return raw


def ingest_seed_selection(
    frame_detections: dict[str, list[RawDetection]],
    *,
    experiment_id: str,
    well_id: str,
    video_id: str,
    min_detections: int = 1,
) -> SeedSelection:
    # Convert to the legacy selection input shape: image_id -> list[{"confidence":...}, ...]
    legacy = {
        image_id: [{"confidence": float(d.confidence)} for d in dets]
        for image_id, dets in frame_detections.items()
    }
    seed_image_id = select_seed_frame(legacy, min_detections=min_detections)
    if seed_image_id is None:
        raise ValueError(f"No suitable seed frame found for {experiment_id} {well_id} (min_detections={min_detections}).")

    chosen = frame_detections.get(seed_image_id, [])
    num = int(len(chosen))
    avg = float(sum(d.confidence for d in chosen) / num) if num else 0.0
    # detection_index is the list position after deterministic sort.
    selected_indices = list(range(num))

    # Frame index is stored on detections.
    seed_frame_index = int(chosen[0].frame_index) if chosen else 0
    return SeedSelection(
        experiment_id=str(experiment_id),
        well_id=str(well_id),
        video_id=str(video_id),
        seed_frame_index=seed_frame_index,
        seed_image_id=str(seed_image_id),
        num_detections=num,
        avg_confidence=avg,
        selection_reason="highest_avg_confidence",
        candidate_frames_evaluated=int(len(frame_detections)),
        selected_detection_indices=selected_indices,
        detector_backend="groundingdino",
    )
