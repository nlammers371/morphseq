from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.io.frame_snapshot_hash import SNAPSHOT_COLUMNS_ORDER, compute_frame_snapshot_hash
from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.segmentation import REQUIRED_COLUMNS_SEGMENTATION_TRACKING
from data_pipeline.segmentation.grounded_sam2.frame_organization_for_sam2 import prepare_bidirectional_propagation, cleanup_frame_directory
from data_pipeline.segmentation.grounded_sam2.gdino_detection import (
    load_groundingdino_model,
    detect_embryos,
    filter_detections,
    select_seed_frame,
    convert_boxes_to_sam2_format,
)
from data_pipeline.segmentation.grounded_sam2.propagation import load_sam2_model, propagate_bidirectional, encode_mask_to_rle


def _load_frame_manifest(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_validated_sentinel(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n")


def segment_and_track_well(
    *,
    frame_manifest_parquet: Path,
    experiment_id: str,
    well_id: str,
    channel_id: str,
    output_root: Path,
    gdino_config_path: Path,
    gdino_weights_path: Path,
    sam2_config_path: Path,
    sam2_checkpoint_path: Path,
    device: str,
    run_id: str | None = None,
    text_prompt: str = "individual embryo",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    confidence_threshold: float = 0.45,
    iou_threshold: float = 0.5,
) -> Path:
    """
    Run GroundingDINO detection + SAM2 bidirectional propagation for a single (well_id, channel_id).

    Writes per-well segmentation_tracking.csv + .validated sentinel.
    """
    df = _load_frame_manifest(frame_manifest_parquet)
    if df.empty:
        raise ValueError(f"Empty frame_manifest: {frame_manifest_parquet}")

    sub = df[
        (df["experiment_id"].astype(str) == str(experiment_id))
        & (df["well_id"].astype(str) == str(well_id))
        & (df["channel_id"].astype(str) == str(channel_id))
    ].copy()
    if sub.empty:
        raise ValueError(f"No manifest rows for experiment_id={experiment_id} well_id={well_id} channel_id={channel_id}")

    sub["frame_index"] = pd.to_numeric(sub["frame_index"], errors="raise").astype(int)
    sub = sub.sort_values(["frame_index"], kind="mergesort").reset_index(drop=True)

    # Snapshot hash is computed from the actual frames we will run.
    snap = sub.loc[:, SNAPSHOT_COLUMNS_ORDER].copy()
    frame_snapshot_hash = compute_frame_snapshot_hash(snap)

    # Model init.
    gdino_model = load_groundingdino_model(
        config_path=str(gdino_config_path),
        weights_path=str(gdino_weights_path),
        device=str(device),
    )
    sam2_predictor = load_sam2_model(
        config_path=str(sam2_config_path),
        checkpoint_path=str(sam2_checkpoint_path),
        device=str(device),
    )

    # Detect per frame.
    frame_detections: dict[str, list[dict[str, Any]]] = {}
    for r in sub.itertuples(index=False):
        image_id = str(getattr(r, "image_id"))
        img_path = Path(str(getattr(r, "source_image_path")))
        dets = detect_embryos(
            model=gdino_model,
            image_path=img_path,
            text_prompt=text_prompt,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
        )
        dets = filter_detections(
            dets,
            confidence_threshold=float(confidence_threshold),
            iou_threshold=float(iou_threshold),
        )
        frame_detections[image_id] = dets

    seed_image_id = select_seed_frame(frame_detections)
    if not seed_image_id:
        raise ValueError("Could not select a seed frame (no detections met threshold)")

    seed_row = sub[sub["image_id"].astype(str) == str(seed_image_id)]
    if seed_row.empty:
        raise RuntimeError(f"Seed image_id not found in manifest slice: {seed_image_id}")
    seed_row = seed_row.iloc[0]
    seed_frame_index = int(seed_row["frame_index"])

    image_w = int(seed_row["image_width_px"])
    image_h = int(seed_row["image_height_px"])

    seed_dets = frame_detections[str(seed_image_id)]
    if not seed_dets:
        raise ValueError(f"Seed frame {seed_image_id} had zero detections after filtering")

    seed_boxes = convert_boxes_to_sam2_format(seed_dets, image_width=image_w, image_height=image_h)
    embryo_ids = [f"embryo_{i}" for i in range(int(seed_boxes.shape[0]))]

    frame_paths = [Path(p) for p in sub["source_image_path"].astype(str).tolist()]
    forward_dir, backward_dir = prepare_bidirectional_propagation(frame_paths, seed_frame_idx=seed_frame_index)
    try:
        results = propagate_bidirectional(
            predictor=sam2_predictor,
            forward_dir=forward_dir,
            backward_dir=backward_dir,
            seed_boxes=seed_boxes,
            seed_frame_idx=seed_frame_index,
            embryo_ids=embryo_ids,
            verbose=False,
        )
    finally:
        cleanup_frame_directory(forward_dir)
        if backward_dir is not None:
            cleanup_frame_directory(backward_dir)

    # Build segmentation_tracking rows.
    run_id = run_id or uuid.uuid4().hex[:8]
    well_index = str(sub["well_index"].iloc[0])

    # Map frame_index -> manifest row (for image_id and physical fields).
    by_frame: dict[int, dict[str, Any]] = {
        int(r.frame_index): {
            "image_id": str(r.image_id),
            "source_image_path": str(r.source_image_path),
            "source_micrometers_per_pixel": float(r.source_micrometers_per_pixel),
            "image_width_px": int(r.image_width_px),
            "image_height_px": int(r.image_height_px),
            "time_int": int(r.time_int),
        }
        for r in sub.itertuples(index=False)
    }

    out_rows: list[dict[str, Any]] = []
    for frame_idx, frame_data in sorted(results.items(), key=lambda kv: int(kv[0])):
        if int(frame_idx) not in by_frame:
            continue
        meta = by_frame[int(frame_idx)]
        image_id = meta["image_id"]

        for instance_id, emb_data in frame_data.items():
            mask = emb_data["mask"]
            bbox = emb_data.get("bbox") or [0, 0, 0, 0]
            area_px = float(emb_data.get("area", 0.0))
            conf = float(emb_data.get("confidence", 0.0))

            embryo_id = f"{well_id}_{str(instance_id)}"

            # RLE dict -> JSON string for CSV.
            rle_dict = encode_mask_to_rle(mask.astype(np.uint8))
            rle_json = json.dumps(rle_dict)

            cx = (float(bbox[0]) + float(bbox[2])) / 2.0 if len(bbox) >= 4 else 0.0
            cy = (float(bbox[1]) + float(bbox[3])) / 2.0 if len(bbox) >= 4 else 0.0

            out_rows.append(
                {
                    "schema_version": 2,
                    "experiment_id": str(experiment_id),
                    "video_id": str(well_id),
                    "well_id": str(well_id),
                    "well_index": str(well_index),
                    "image_id": str(image_id),
                    "embryo_id": str(embryo_id),
                    "instance_id": str(instance_id),
                    "snip_id": f"{embryo_id}_{channel_id}_f{int(frame_idx):04d}",
                    "frame_index": int(frame_idx),
                    "time_int": int(meta["time_int"]),
                    "channel_id": str(channel_id),
                    "source_micrometers_per_pixel": float(meta["source_micrometers_per_pixel"]),
                    "image_width_px": int(meta["image_width_px"]),
                    "image_height_px": int(meta["image_height_px"]),
                    "frame_snapshot_hash": str(frame_snapshot_hash),
                    "mask_rle": rle_json,
                    "embryo_mask_rle": rle_json,
                    "area_px": float(area_px),
                    "bbox_x_min": float(bbox[0]) if len(bbox) > 0 else 0.0,
                    "bbox_y_min": float(bbox[1]) if len(bbox) > 1 else 0.0,
                    "bbox_x_max": float(bbox[2]) if len(bbox) > 2 else 0.0,
                    "bbox_y_max": float(bbox[3]) if len(bbox) > 3 else 0.0,
                    "mask_confidence": float(conf),
                    "centroid_x_px": float(cx),
                    "centroid_y_px": float(cy),
                    "is_seed_frame": bool(int(frame_idx) == seed_frame_index),
                    "source_image_path": str(meta["source_image_path"]),
                    # Leave empty: per-snip PNGs are materialized in snip processing from RLE.
                    "exported_mask_path": "",
                    "embryo_mask_path": "",
                }
            )

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        raise ValueError("No segmentation_tracking rows were produced")

    validate_dataframe_schema(out_df, REQUIRED_COLUMNS_SEGMENTATION_TRACKING, "segmentation_tracking")
    if out_df["snip_id"].duplicated().any():
        dups = out_df.loc[out_df["snip_id"].duplicated(), "snip_id"].astype(str).head(10).tolist()
        raise ValueError(f"snip_id is not unique; examples: {dups}")

    per_well_contracts = output_root / experiment_id / "per_well" / well_id / "contracts"
    per_well_contracts.mkdir(parents=True, exist_ok=True)
    out_csv = per_well_contracts / "segmentation_tracking.csv"
    out_df.to_csv(out_csv, index=False)
    _write_validated_sentinel(per_well_contracts / ".segmentation_tracking.validated")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-manifest", type=Path, required=True)
    ap.add_argument("--experiment-id", type=str, required=True)
    ap.add_argument("--well-id", type=str, required=True)
    ap.add_argument("--channel-id", type=str, default="BF")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--run-id", type=str, default=None)

    ap.add_argument("--gdino-config", type=Path, required=True)
    ap.add_argument("--gdino-weights", type=Path, required=True)
    ap.add_argument("--sam2-config", type=Path, required=True)
    ap.add_argument("--sam2-checkpoint", type=Path, required=True)
    args = ap.parse_args()

    segment_and_track_well(
        frame_manifest_parquet=args.frame_manifest,
        experiment_id=args.experiment_id,
        well_id=args.well_id,
        channel_id=args.channel_id,
        output_root=args.output_root,
        gdino_config_path=args.gdino_config,
        gdino_weights_path=args.gdino_weights,
        sam2_config_path=args.sam2_config,
        sam2_checkpoint_path=args.sam2_checkpoint,
        device=args.device,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()

