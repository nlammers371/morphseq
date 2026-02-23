"""Adapters that map different annotation JSON formats to a canonical render model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import AnnotationRecord, FrameRecord, VideoRecord


def detect_source_format(data: dict[str, Any]) -> str:
    """Detect supported source format from loaded JSON content."""
    if isinstance(data.get("experiments"), dict):
        return "grounded_sam2"
    if isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
        return "coco"
    raise ValueError("Could not detect source format. Supported: grounded_sam2, coco")


def load_results_json(results_json_path: Path) -> dict[str, Any]:
    with open(results_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_experiment_id_from_video(video_id: str) -> str:
    m = re.match(r"^(.+)_([A-H][0-9]{2})$", video_id)
    if m:
        return m.group(1)
    if "_" in video_id:
        return video_id.rsplit("_", 1)[0]
    return video_id


def _derive_video_id_from_image_stem(image_stem: str) -> str | None:
    # Handles common ids like: 20260122_A01_ch00_t0001
    m = re.match(r"^(.+_[A-H][0-9]{2})(?:_ch\d+)?_t\d+$", image_stem)
    if m:
        return m.group(1)
    # Fallback for simpler forms like <video_id>_t0001
    m2 = re.match(r"^(.+)_t\d+$", image_stem)
    return m2.group(1) if m2 else None


def _extract_frame_index(image_id: str, fallback: int) -> int:
    m = re.search(r"_t(\d+)$", image_id)
    if m:
        return int(m.group(1))
    return fallback


def list_videos(
    results_json_path: Path,
    *,
    source_format: str = "auto",
    experiment_id: str | None = None,
) -> list[str]:
    data = load_results_json(results_json_path)
    fmt = detect_source_format(data) if source_format == "auto" else source_format

    if fmt == "grounded_sam2":
        exps = data.get("experiments", {})
        if experiment_id:
            return sorted(list(exps.get(experiment_id, {}).get("videos", {}).keys()))
        all_videos: list[str] = []
        for exp_data in exps.values():
            all_videos.extend(exp_data.get("videos", {}).keys())
        return sorted(set(all_videos))

    if fmt == "coco":
        vids: set[str] = set()
        for image in data.get("images", []):
            image_vid = image.get("video_id")
            if not image_vid:
                stem = Path(str(image.get("file_name", ""))).stem
                image_vid = _derive_video_id_from_image_stem(stem)
            if not image_vid:
                continue
            if experiment_id and _derive_experiment_id_from_video(image_vid) != experiment_id:
                continue
            vids.add(image_vid)
        return sorted(vids)

    raise ValueError(f"Unsupported source_format: {fmt}")


def load_video_record(
    results_json_path: Path,
    *,
    experiment_id: str,
    video_id: str,
    source_format: str = "auto",
    images_root: Path | None = None,
) -> VideoRecord:
    data = load_results_json(results_json_path)
    fmt = detect_source_format(data) if source_format == "auto" else source_format

    if fmt == "grounded_sam2":
        return _load_grounded_sam2_video_record(
            data,
            results_json_path=results_json_path,
            experiment_id=experiment_id,
            video_id=video_id,
        )

    if fmt == "coco":
        return _load_coco_video_record(
            data,
            results_json_path=results_json_path,
            experiment_id=experiment_id,
            video_id=video_id,
            images_root=images_root,
        )

    raise ValueError(f"Unsupported source_format: {fmt}")


def _load_grounded_sam2_video_record(
    data: dict[str, Any],
    *,
    results_json_path: Path,
    experiment_id: str,
    video_id: str,
) -> VideoRecord:
    experiments = data.get("experiments", {})
    if experiment_id not in experiments:
        raise ValueError(f"Experiment not found: {experiment_id}")
    videos = experiments[experiment_id].get("videos", {})
    if video_id not in videos:
        raise ValueError(f"Video not found in experiment {experiment_id}: {video_id}")

    image_dict = videos[video_id].get("image_ids", {})
    sam2_root = results_json_path.parent.parent
    video_dir = sam2_root / "raw_data_organized" / experiment_id / "images" / video_id

    frames: list[FrameRecord] = []
    for i, image_id in enumerate(sorted(image_dict.keys())):
        image_data = image_dict.get(image_id, {})
        image_path = video_dir / f"{image_id}.jpg"

        annotations: list[AnnotationRecord] = []
        embryos = image_data.get("embryos", {})
        for emb_id, emb_data in embryos.items():
            segmentation = emb_data.get("segmentation")
            bbox = emb_data.get("bbox")
            if (not bbox) and isinstance(segmentation, dict):
                bbox = segmentation.get("bbox")
            annotations.append(
                AnnotationRecord(
                    annotation_id=str(emb_id),
                    segmentation=segmentation,
                    bbox=list(bbox) if bbox else None,
                    bbox_mode="xyxy_norm",
                    score=emb_data.get("mask_confidence"),
                    label=str(emb_id),
                    metadata={
                        "area": emb_data.get("area"),
                        "snip_id": emb_data.get("snip_id"),
                    },
                )
            )

        frame = FrameRecord(
            image_id=image_id,
            image_path=image_path,
            frame_index=_extract_frame_index(image_id, i),
            annotations=annotations,
            qc_flags=list(image_data.get("qc_flags", [])),
        )
        frames.append(frame)

    return VideoRecord(experiment_id=experiment_id, video_id=video_id, frames=frames)


def _resolve_image_path(file_name: str, *, results_json_path: Path, images_root: Path | None) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    if images_root is not None:
        return images_root / p
    return results_json_path.parent / p


def _load_coco_video_record(
    data: dict[str, Any],
    *,
    results_json_path: Path,
    experiment_id: str,
    video_id: str,
    images_root: Path | None,
) -> VideoRecord:
    categories = {c.get("id"): c.get("name", f"cat_{c.get('id')}") for c in data.get("categories", [])}

    images = data.get("images", [])
    anns = data.get("annotations", [])

    image_by_id: dict[Any, dict[str, Any]] = {}
    selected_images: list[dict[str, Any]] = []

    for image in images:
        image_id = image.get("id")
        if image_id is None:
            continue
        image_by_id[image_id] = image

        image_vid = image.get("video_id")
        if not image_vid:
            stem = Path(str(image.get("file_name", ""))).stem
            image_vid = _derive_video_id_from_image_stem(stem)

        if image_vid != video_id:
            continue

        image_exp = image.get("experiment_id") or _derive_experiment_id_from_video(str(image_vid))
        if image_exp != experiment_id:
            continue

        selected_images.append(image)

    if not selected_images:
        raise ValueError(
            f"No COCO images found for experiment={experiment_id}, video={video_id}"
        )

    anns_by_image: dict[Any, list[dict[str, Any]]] = {}
    for ann in anns:
        img_id = ann.get("image_id")
        if img_id not in image_by_id:
            continue
        anns_by_image.setdefault(img_id, []).append(ann)

    # Stable sort: frame_index if provided, else by file name
    def _sort_key(img: dict[str, Any]) -> tuple[int, str]:
        fn = str(img.get("file_name", ""))
        idx = img.get("frame_index")
        if isinstance(idx, int):
            return idx, fn
        stem = Path(fn).stem
        fallback_idx = _extract_frame_index(stem, 0)
        return fallback_idx, fn

    selected_images.sort(key=_sort_key)

    frames: list[FrameRecord] = []
    for i, image in enumerate(selected_images):
        image_id_raw = image.get("id")
        file_name = str(image.get("file_name", ""))
        stem = Path(file_name).stem if file_name else str(image_id_raw)
        image_path = _resolve_image_path(file_name, results_json_path=results_json_path, images_root=images_root)

        ann_records: list[AnnotationRecord] = []
        for ann in anns_by_image.get(image_id_raw, []):
            cat_id = ann.get("category_id")
            ann_records.append(
                AnnotationRecord(
                    annotation_id=str(ann.get("id", f"ann_{len(ann_records)}")),
                    segmentation=ann.get("segmentation"),
                    bbox=list(ann.get("bbox", [])) if ann.get("bbox") is not None else None,
                    bbox_mode="xywh_abs",
                    score=ann.get("score"),
                    label=categories.get(cat_id, str(cat_id) if cat_id is not None else "annotation"),
                    category_id=cat_id,
                    metadata={
                        "iscrowd": ann.get("iscrowd", 0),
                        "area": ann.get("area"),
                    },
                )
            )

        frame = FrameRecord(
            image_id=stem,
            image_path=image_path,
            frame_index=int(image.get("frame_index", _extract_frame_index(stem, i))),
            annotations=ann_records,
            qc_flags=[],
        )
        frames.append(frame)

    return VideoRecord(experiment_id=experiment_id, video_id=video_id, frames=frames)
