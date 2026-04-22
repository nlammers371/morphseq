from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.utils.cuda_diagnostics import require_cuda_or_raise
from data_pipeline.models.paths import resolve_segmentation_and_tracking_model_paths
from data_pipeline.models.groundingdino import load_groundingdino_model as load_gdino_model
from data_pipeline.models.sam2 import load_sam2_video_predictor
from data_pipeline.segmentation.backends import load_segmentation_backends_config
from data_pipeline.segmentation.grounded_sam2.gdino_detection import convert_boxes_to_sam2_format
from data_pipeline.segmentation.grounded_sam2.propagation import propagate_bidirectional
from data_pipeline.segmentation.grounded_sam2.frame_organization_for_sam2 import sam2_frame_context
from data_pipeline.segmentation.grounded_sam2.csv_formatter import extract_well_index

from ..normalize_context import NormalizeContext
from ..ingestors import get_detector_ingestor, get_tracker_ingestor
from ..normalizers import (
    normalize_frame_detections,
    normalize_seed_selection,
    normalize_track_instances,
    normalize_mask_rle,
    build_segmentation_tracking_contract,
)
from data_pipeline.segmentation.video_generation.video_config import VideoConfig
from ..qc.render_overlays import render_overlays_from_segmentation_tracking
from ..qc.materialize_raw_frames import materialize_raw_frames
from ..qc.render_raw_video import render_raw_video, RawVideoConfig


def _resolve_rel_to_data_root(path_str: str, *, data_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (Path(data_root) / p)


def run_segmentation_and_tracking(
    *,
    frame_contract_csv: Path,
    experiment_id: str,
    well_id: str,
    output_root: Path,
    pipeline_config: dict[str, Any],
    device: str = "cuda",
    run_id: str | None = None,
    verbose: bool = False,
) -> dict[str, Path]:
    """
    Per-well Phase 3 runner.

    Writes per-well shards under:
      {output_root}/segmentation_and_tracking/{experiment}/per_well/{well}/
    """
    output_root = Path(output_root)
    run_id = run_id or uuid.uuid4().hex[:8]

    if str(device).lower().startswith("cuda"):
        # Fail early with an actionable error instead of silently falling back
        # (and producing different masks).
        require_cuda_or_raise()

    sat_cfg = (pipeline_config.get("segmentation_and_tracking") or {})
    channel_id = str(sat_cfg.get("channel_id", "BF"))

    # Manifest `well_id` is often experiment-qualified (e.g. "20240418_A01").
    # Use that value for filtering and storage so we match contracts.
    requested_well_id = str(well_id)
    if not requested_well_id.startswith(f"{experiment_id}_"):
        candidate = f"{experiment_id}_{requested_well_id}"
    else:
        candidate = requested_well_id

    # Use the experiment-qualified well_id for storage to match frame_contract keys.
    shard_dir = output_root / "segmentation_and_tracking" / str(experiment_id) / "per_well" / str(candidate)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Canonical per-well output roots (real files).
    contracts_dir = shard_dir / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = shard_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest and filter to BF frames for this well.
    manifest = pd.read_csv(frame_contract_csv)
    well_id_col = manifest["well_id"].astype(str)
    wanted = {str(requested_well_id), candidate}
    well_df = manifest[
        (manifest["experiment_id"].astype(str) == str(experiment_id))
        & (well_id_col.isin(wanted))
        & (manifest["channel_id"].astype(str) == str(channel_id))
    ].copy()
    if len(well_df) == 0:
        raise ValueError(f"No frames found for experiment={experiment_id} well={well_id} channel_id={channel_id}.")

    well_df["time_int"] = well_df["time_int"].astype(int)
    well_df = well_df.sort_values(["time_int", "image_id"]).reset_index(drop=True)
    max_frames = sat_cfg.get("max_frames")
    if max_frames is not None:
        well_df = well_df.head(int(max_frames)).reset_index(drop=True)

    manifest_well_id = str(well_df["well_id"].iloc[0])
    # Canonical for contracts: match frame_contract well_id.
    canonical_well_id = manifest_well_id
    # Human-facing shorthand used in some IDs.
    well_slug = canonical_well_id.split("_")[-1]
    video_id = f"{experiment_id}_{well_slug}"
    # frame_contract well_index may be a string like "A01" (legacy). Normalize to 1-96 index.
    well_index_raw = str(well_df["well_index"].iloc[0])
    try:
        well_index = int(well_index_raw)
    except ValueError:
        well_index = int(extract_well_index(well_index_raw))

    # Mask head folder naming: embryo_mask, yolk_mask, etc.
    mask_type = str(sat_cfg.get("mask_type", "embryo"))
    mask_head = f"{mask_type}_mask"
    masks_root_dir = shard_dir / "masks"
    masks_root_dir.mkdir(parents=True, exist_ok=True)
    mask_head_dir = masks_root_dir / mask_head
    mask_head_dir.mkdir(parents=True, exist_ok=True)

    # Pre-resolve frame paths once (used by raw_frames, detections, and SAM2).
    frames_for_qc: list[tuple[str, Path]] = []
    stitched_abs_by_image_id: dict[str, Path] = {}
    stitched_rel_by_image_id: dict[str, str] = {}
    time_int_by_image_id: dict[str, int] = {}
    img_hw_by_image_id: dict[str, tuple[int, int]] = {}
    for _, row in well_df.iterrows():
        image_id = str(row["image_id"])
        stitched_rel = str(row["stitched_image_path"])
        stitched_abs = _resolve_rel_to_data_root(stitched_rel, data_root=output_root)
        stitched_abs_by_image_id[image_id] = stitched_abs
        stitched_rel_by_image_id[image_id] = stitched_rel
        time_int_by_image_id[image_id] = int(row["time_int"])
        img_hw_by_image_id[image_id] = (int(row["image_height_px"]), int(row["image_width_px"]))
        frames_for_qc.append((image_id, stitched_abs))

    # Optional: persist raw frames + a SAM2 sequential view for debugging / later reuse.
    raw_frames_cfg = (sat_cfg.get("raw_frames") or {})
    if bool(raw_frames_cfg.get("enabled", True)):
        materialize_raw_frames(
            frames=frames_for_qc,
            out_dir=artifacts_dir,
            mode=str(raw_frames_cfg.get("mode", "symlink")),
            write_sam2_seq_dir=bool(raw_frames_cfg.get("write_sam2_seq_dir", True)),
        )

    # Optional: raw video (no masks) for quick QC.
    raw_video_cfg = (sat_cfg.get("raw_video") or {})
    if bool(raw_video_cfg.get("enabled", False)):
        well_slug = str(canonical_well_id).split("_")[-1]
        render_raw_video(
            frames=frames_for_qc,
            out_dir=artifacts_dir / "raw_video",
            out_name=f"{well_slug}_raw.mp4",
            cfg=RawVideoConfig(
                fps=int(raw_video_cfg.get("fps", 10)),
                codec=str(raw_video_cfg.get("codec", "mp4v")),
                write_video=bool(raw_video_cfg.get("write_video", True)),
                write_frames=bool(raw_video_cfg.get("write_frames", False)),
            ),
        )

    # Resolve model paths from config (single models_root under output_root).
    model_paths = resolve_segmentation_and_tracking_model_paths(config=pipeline_config, data_root=output_root)

    backends = load_segmentation_backends_config(pipeline_config)
    det_ing = get_detector_ingestor(backends.detector_backend)
    trk_ing = get_tracker_ingestor(backends.tracker_backend)

    det_ctx = NormalizeContext(
        source_backend=backends.detector_backend,
        source_model=model_paths.groundingdino.model_name,
        model_release=model_paths.groundingdino.model_release,
        run_id=run_id,
    )
    trk_ctx = NormalizeContext(
        source_backend=backends.tracker_backend,
        source_model=model_paths.sam2.model_name,
        model_release=model_paths.sam2.model_release,
        run_id=run_id,
    )

    # Load detector model once.
    det_model = load_gdino_model(
        repo_dir=model_paths.groundingdino.repo_dir,
        config_path=model_paths.groundingdino.config_path,
        weights_path=model_paths.groundingdino.weights_path,
        device=device,
    )

    raw_dets_all = []
    dets_by_image: dict[str, list] = {}
    for image_id, stitched_abs in frames_for_qc:
        time_int = int(time_int_by_image_id[image_id])
        img_h, img_w = img_hw_by_image_id[image_id]

        raw = det_ing.ingest_detections(
            det_model,
            stitched_abs,
            time_int=time_int,
            image_id=image_id,
            image_height_px=img_h,
            image_width_px=img_w,
            device=device,
            text_prompt=str(sat_cfg.get("text_prompt", "individual embryo")),
            box_threshold=float(sat_cfg.get("box_threshold", 0.35)),
            text_threshold=float(sat_cfg.get("text_threshold", 0.25)),
            confidence_threshold=float(sat_cfg.get("confidence_threshold", 0.45)),
            iou_threshold=float(sat_cfg.get("iou_threshold", 0.5)),
        )
        dets_by_image[image_id] = raw
        raw_dets_all.extend(raw)

    det_ctx.stamp(raw_dets_all)
    det_df = normalize_frame_detections(raw_dets_all, experiment_id=experiment_id, well_id=canonical_well_id, video_id=video_id)
    det_df.to_parquet(contracts_dir / "frame_detections.parquet", index=False)

    # Seed selection.
    seed = det_ing.ingest_seed_selection(
        dets_by_image,
        experiment_id=experiment_id,
        well_id=canonical_well_id,
        video_id=video_id,
        min_detections=int(sat_cfg.get("min_detections", 1)),
    )
    seed.detector_backend = det_ctx.source_backend
    seed.run_id = det_ctx.run_id
    seed_df = normalize_seed_selection([seed])
    seed_df.to_parquet(contracts_dir / "seed_selection.parquet", index=False)

    # Build SAM2 frame sequence (chronological) and map seq_idx -> image_id/time_int.
    frame_paths = []
    seq_index_to_time_int: dict[int, int] = {}
    image_id_by_seq_index: dict[int, str] = {}
    source_image_path_by_image_id: dict[str, str] = {}
    for seq_idx, (image_id, stitched_abs) in enumerate(frames_for_qc):
        frame_paths.append(stitched_abs)
        time_int = int(time_int_by_image_id[image_id])
        seq_index_to_time_int[seq_idx] = time_int
        image_id_by_seq_index[seq_idx] = image_id
        source_image_path_by_image_id[image_id] = stitched_rel_by_image_id[image_id]

    # Determine seed sequence index by matching seed_image_id.
    seed_seq_idx = None
    for seq_idx, image_id in image_id_by_seq_index.items():
        if image_id == seed.seed_image_id:
            seed_seq_idx = int(seq_idx)
            break
    if seed_seq_idx is None:
        raise ValueError(f"Seed image_id not found in well frames: {seed.seed_image_id}")

    # Convert seed detections for the seed frame to SAM2 prompt boxes.
    seed_row = well_df[well_df["image_id"].astype(str) == str(seed.seed_image_id)].iloc[0]
    img_h = int(seed_row["image_height_px"])
    img_w = int(seed_row["image_width_px"])
    seed_raw_dets = dets_by_image.get(seed.seed_image_id, [])
    seed_boxes = convert_boxes_to_sam2_format(
        [{"box_xyxy": d.box_xyxy_norm, "confidence": d.confidence} for d in seed_raw_dets],
        img_h,
        img_w,
    )

    # Load tracker model once.
    predictor = load_sam2_video_predictor(
        sam2_models_root=model_paths.sam2.models_root,
        config_path=model_paths.sam2.config_path,
        checkpoint_path=model_paths.sam2.checkpoint_path,
        device=device,
    )

    # Run SAM2 native bidirectional on a single sequential frame directory.
    with sam2_frame_context(frame_paths) as frame_dir:
        sam2_results = propagate_bidirectional(
            predictor=predictor,
            frame_dir=Path(frame_dir),
            seed_boxes=seed_boxes,
            seed_frame_idx=int(seed_seq_idx),
            embryo_ids=None,
            verbose=verbose,
        )

    # Remap SAM2 seq indices to manifest time_int.
    remapped: dict[int, dict[str, dict]] = {}
    image_id_by_time_int: dict[int, str] = {}
    for seq_idx, frame_data in sam2_results.items():
        time_int = seq_index_to_time_int.get(int(seq_idx))
        if time_int is None:
            continue
        remapped[int(time_int)] = frame_data
        image_id_by_time_int[int(time_int)] = image_id_by_seq_index[int(seq_idx)]

    raw_tracks = trk_ing.ingest_propagation(
        remapped,
        image_id_by_time_int=image_id_by_time_int,
        seed_time_int=int(seed.seed_time_int),
        seed_image_id=str(seed.seed_image_id),
        well_id=str(canonical_well_id),
        channel_id=str(channel_id),
    )
    trk_ctx.stamp(raw_tracks)
    tracks_df = normalize_track_instances(
        raw_tracks,
        experiment_id=experiment_id,
        well_id=canonical_well_id,
        well_index=well_index,
        video_id=video_id,
    )
    tracks_df.to_parquet(contracts_dir / "embryo_track_instances.parquet", index=False)

    exported_mask_rel_prefix = str((masks_root_dir.relative_to(output_root)).as_posix())
    raw_masks = trk_ing.tracks_to_raw_masks(
        raw_tracks,
        source_image_path_by_image_id=source_image_path_by_image_id,
        exported_mask_dir=mask_head_dir,
        exported_mask_rel_prefix=exported_mask_rel_prefix,
        mask_type=mask_type,
    )
    trk_ctx.stamp(raw_masks)
    masks_df = normalize_mask_rle(
        raw_masks,
        experiment_id=experiment_id,
        well_id=canonical_well_id,
        video_id=video_id,
        channel_id=str(channel_id),
    )
    masks_df.to_parquet(contracts_dir / "embryo_mask_rle.parquet", index=False)

    final_df = build_segmentation_tracking_contract(masks_df, well_index=well_index)
    segmentation_tracking_csv = contracts_dir / "segmentation_tracking.csv"
    final_df.to_csv(segmentation_tracking_csv, index=False)

    # Optional QC overlays (frames + MP4) for sanity-checking masks.
    overlay_cfg = (sat_cfg.get("qc_overlay") or {})
    if bool(overlay_cfg.get("enabled", True)):
        render_mask_type = overlay_cfg.get("render_mask_type")
        if render_mask_type is None:
            render_mask_type = mask_type
        render_mask_head = f"{str(render_mask_type)}_mask"

        qc_dir = artifacts_dir / "overlays" / render_mask_head
        qc_video_cfg = VideoConfig()
        qc_video_cfg.MASK_ALPHA = float(overlay_cfg.get("alpha", 0.45))
        qc_video_cfg.FPS = int(overlay_cfg.get("fps", 10))
        qc_video_cfg.WRITE_FRAMES = bool(overlay_cfg.get("write_frames", True))
        qc_video_cfg.WRITE_VIDEO = bool(overlay_cfg.get("write_video", True))
        render_cfg = (overlay_cfg.get("render") or {})
        qc_video_cfg.OUTPUT_SCALE = float(render_cfg.get("scale", 1.0))
        qc_video_cfg.LABEL_FONT_SCALE = float(render_cfg.get("label_font_scale", 0.8))
        qc_video_cfg.LABEL_THICKNESS = int(render_cfg.get("label_thickness", 2))
        qc_video_cfg.BANNER_FONT_SCALE = float(render_cfg.get("banner_font_scale", 1.0))
        qc_video_cfg.BANNER_THICKNESS = int(render_cfg.get("banner_thickness", 2))
        try:
            render_overlays_from_segmentation_tracking(
                segmentation_tracking_csv,
                output_root=output_root,
                out_dir=qc_dir,
                out_name=f"{well_slug}_{render_mask_head}_overlay.mp4",
                cfg=qc_video_cfg,
                mask_type=str(render_mask_type) if render_mask_type is not None else None,
                frame_suffix=f"_{render_mask_head}_overlay",
            )
        except Exception as exc:  # pragma: no cover
            # QC overlays are optional; never fail the core pipeline because of them.
            qc_dir.mkdir(parents=True, exist_ok=True)
            (qc_dir / "qc_overlay_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            if verbose:
                print(f"[qc_overlay] failed: {type(exc).__name__}: {exc}")

    validated = contracts_dir / ".segment_and_track.validated"
    validated.write_text("validated\n")
    return {
        "shard_dir": shard_dir,
        "validated_flag": validated,
    }
