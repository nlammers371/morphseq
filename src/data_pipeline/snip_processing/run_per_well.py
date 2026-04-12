from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skimage.io as io

from data_pipeline.segmentation.video_generation.mask_decoding import decode_coco_segmentation
from data_pipeline.snip_processing.process_snips import process_single_snip
from data_pipeline.snip_processing.validate_snip_manifest import validate_snip_manifest
# Imported for side-effect-free availability; snapshot hash is produced upstream in segmentation_tracking.


@dataclass(frozen=True)
class SnipProcessingConfig:
    # Legacy downstream models expect (H, W) = (576, 256).
    output_shape: tuple[int, int] = (576, 256)
    # Keep explicit target pixel size for determinism across experiments.
    target_pixel_size_um: float = 7.8
    blend_radius_um: float = 30.0
    save_raw_crops: bool = True
    write_manifest_csv: bool = True
    background_n_samples: int = 100
    background_seed: int = 309


def _git_sha_or_unknown() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()[:12]
    except Exception:
        return "unknown"


def _load_table(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_rle(rle: Any) -> dict[str, Any]:
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        raise ValueError("Missing RLE")
    if isinstance(rle, dict):
        return rle
    if isinstance(rle, str):
        s = rle.strip()
        if not s:
            raise ValueError("Empty RLE string")
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            # Some pipelines store the compressed counts as a raw string; without size we can't decode.
            raise ValueError("RLE string is not JSON; expected pycocotools-style dict JSON")
        if not isinstance(obj, dict):
            raise ValueError("Decoded RLE JSON is not a dict")
        return obj
    raise TypeError(f"Unsupported RLE type: {type(rle)}")


def _ensure_mask_png_for_snip(
    *,
    snip_id: str,
    mask_path_value: str | None,
    rle_value: Any,
    out_dir: Path,
) -> Path:
    """
    Return a mask path that exists on disk. Prefer existing path; otherwise decode RLE and write PNG.
    """
    if mask_path_value:
        p = Path(mask_path_value)
        if p.exists():
            return p

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{snip_id}.png"
    if out_path.exists():
        return out_path

    rle = _parse_rle(rle_value)
    # Use size from RLE if available; fall back to required keys.
    size = rle.get("size")
    if not (isinstance(size, (list, tuple)) and len(size) >= 2):
        raise ValueError("RLE dict missing 'size' for decoding")
    h, w = int(size[-2]), int(size[-1])
    mask = decode_coco_segmentation(rle, h, w)
    io.imsave(out_path, (mask > 0).astype(np.uint8) * 255, check_contrast=False)
    return out_path


def _estimate_background_stats_for_channel(
    df: pd.DataFrame,
    *,
    channel_id: str,
    n_samples: int,
    seed: int,
) -> tuple[float, float]:
    # Sample unique frames deterministically.
    frames = (
        df.loc[df["channel_id"].astype(str) == str(channel_id), ["image_id", "source_image_path"]]
        .drop_duplicates()
        .sort_values(["image_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if frames.empty:
        raise ValueError(f"No frames found for channel_id={channel_id}")

    rng = np.random.default_rng(int(seed))
    n_pick = int(min(len(frames), int(n_samples)))
    pick_idx = rng.choice(len(frames), size=n_pick, replace=False)
    pick = frames.iloc[pick_idx].reset_index(drop=True)

    bg_pixels: list[np.ndarray] = []
    for image_id, src_path in pick.itertuples(index=False):
        img = io.imread(src_path)
        if img.ndim == 3:
            img = img[:, :, 0]

        # Union mask across all embryos in this frame.
        frame_rows = df[df["image_id"].astype(str) == str(image_id)]
        union = None
        for r in frame_rows.itertuples(index=False):
            # Prefer canonical RLE; fall back to legacy mask_rle.
            rle_val = getattr(r, "embryo_mask_rle", None)
            if rle_val is None:
                rle_val = getattr(r, "mask_rle", None)
            rle = _parse_rle(rle_val)
            size = rle.get("size")
            if not (isinstance(size, (list, tuple)) and len(size) >= 2):
                continue
            h, w = int(size[-2]), int(size[-1])
            m = decode_coco_segmentation(rle, h, w).astype(bool)
            union = m if union is None else (union | m)

        if union is None:
            continue

        # Resize union to match image if needed (rare; but defensive).
        if union.shape != img.shape[:2]:
            # Simple nearest resize via slicing would be wrong; just skip sample.
            continue

        bg = img[~union]
        if bg.size:
            bg_pixels.append(bg.astype(np.float32).ravel())

    if not bg_pixels:
        return 128.0, 30.0

    all_bg = np.concatenate(bg_pixels)
    return float(np.mean(all_bg)), float(np.std(all_bg))


def run_snip_processing_for_well(
    *,
    segmentation_tracking_path: Path,
    output_root: Path,
    config: SnipProcessingConfig | None = None,
    snip_processing_run_id: str | None = None,
) -> Path:
    config = config or SnipProcessingConfig()

    df = _load_table(segmentation_tracking_path)
    if df.empty:
        raise ValueError(f"Empty segmentation_tracking: {segmentation_tracking_path}")

    # Require v2 snapshot fields.
    required_v2 = [
        "schema_version",
        "channel_id",
        "source_micrometers_per_pixel",
        "image_width_px",
        "image_height_px",
        "frame_snapshot_hash",
        "source_image_path",
        "image_id",
        "well_id",
        "well_index",
        "experiment_id",
        "embryo_id",
        "snip_id",
        "frame_index",
    ]
    missing = [c for c in required_v2 if c not in df.columns]
    if missing:
        raise ValueError(f"segmentation_tracking missing required v2 columns: {missing}")

    exp_id = str(df["experiment_id"].iloc[0])
    well_id = str(df["well_id"].iloc[0])
    well_index = str(df["well_index"].iloc[0])

    run_id = snip_processing_run_id or uuid.uuid4().hex[:8]
    version = _git_sha_or_unknown()

    per_well_root = output_root / exp_id / "per_well" / well_id
    contracts_dir = per_well_root / "contracts"
    processed_dir = per_well_root / "snips" / "processed"
    raw_crops_dir = per_well_root / "snips" / "raw_crops"
    materialized_masks_dir = per_well_root / "snips" / "masks" / "embryo_mask"

    contracts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if config.save_raw_crops:
        raw_crops_dir.mkdir(parents=True, exist_ok=True)

    # Background stats per channel, cached.
    bg_stats: dict[str, tuple[float, float]] = {}
    for ch in sorted(df["channel_id"].astype(str).unique()):
        cache_path = contracts_dir / f"background_stats__{ch}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            bg_stats[ch] = (float(payload["background_mean"]), float(payload["background_std"]))
        else:
            mean, std = _estimate_background_stats_for_channel(
                df,
                channel_id=ch,
                n_samples=config.background_n_samples,
                seed=config.background_seed,
            )
            bg_stats[ch] = (mean, std)
            cache_path.write_text(json.dumps({"background_mean": mean, "background_std": std}, indent=2))

    rows: list[dict[str, Any]] = []
    for r in df.itertuples(index=False):
        channel_id = str(getattr(r, "channel_id"))
        mean, std = bg_stats[channel_id]

        snip_id = str(getattr(r, "snip_id"))
        embryo_id = str(getattr(r, "embryo_id"))
        instance_id = str(getattr(r, "instance_id", embryo_id))
        image_id = str(getattr(r, "image_id"))
        frame_index = int(getattr(r, "frame_index"))

        expected_snip = f"{embryo_id}_{channel_id}_f{frame_index:04d}"
        if snip_id != expected_snip:
            raise ValueError(f"snip_id invariant failed: got {snip_id} expected {expected_snip}")

        source_image_path = str(getattr(r, "source_image_path"))
        um_per_px = float(getattr(r, "source_micrometers_per_pixel"))

        # Prefer canonical mask columns; fall back to legacy names.
        embryo_mask_path_val = getattr(r, "embryo_mask_path", None)
        if embryo_mask_path_val is None:
            embryo_mask_path_val = getattr(r, "exported_mask_path", None)
        embryo_mask_rle_val = getattr(r, "embryo_mask_rle", None)
        if embryo_mask_rle_val is None:
            embryo_mask_rle_val = getattr(r, "mask_rle", None)

        embryo_mask_path = _ensure_mask_png_for_snip(
            snip_id=snip_id,
            mask_path_value=str(embryo_mask_path_val) if embryo_mask_path_val is not None else None,
            rle_value=embryo_mask_rle_val,
            out_dir=materialized_masks_dir,
        )

        yolk_mask_path = None
        if hasattr(r, "yolk_mask_path"):
            yp = getattr(r, "yolk_mask_path")
            if isinstance(yp, str) and yp and Path(yp).exists():
                yolk_mask_path = Path(yp)

        meta = process_single_snip(
            snip_id=snip_id,
            image_path=Path(source_image_path),
            mask_path=Path(embryo_mask_path),
            yolk_mask_path=yolk_mask_path,
            output_shape=tuple(config.output_shape),
            pixel_size_um=um_per_px,
            target_pixel_size_um=float(config.target_pixel_size_um),
            background_mean=float(mean),
            background_std=float(std),
            save_raw_crops=bool(config.save_raw_crops),
            raw_crops_dir=raw_crops_dir if config.save_raw_crops else None,
            processed_dir=processed_dir,
        )

        rows.append(
            {
                "schema_version": 2,
                "snip_id": snip_id,
                "experiment_id": exp_id,
                "well_id": well_id,
                "well_index": well_index,
                "image_id": image_id,
                "frame_index": frame_index,
                "channel_id": channel_id,
                "embryo_id": embryo_id,
                "instance_id": instance_id,
                "source_image_path": source_image_path,
                "embryo_mask_path": str(embryo_mask_path),
                "yolk_mask_path": str(yolk_mask_path) if yolk_mask_path else None,
                "source_micrometers_per_pixel": um_per_px,
                "frame_snapshot_hash": str(getattr(r, "frame_snapshot_hash")),
                "processed_snip_path": meta["processed_path"],
                "raw_crop_path": meta["raw_crop_path"],
                "target_pixel_size_um": float(config.target_pixel_size_um),
                "output_height_px": int(config.output_shape[0]),
                "output_width_px": int(config.output_shape[1]),
                "blend_radius_um": float(config.blend_radius_um),
                "background_mean": float(mean),
                "background_std": float(std),
                "rotation_angle_rad": float(meta["rotation_angle_rad"]),
                "rotation_angle_deg": float(meta["rotation_angle_deg"]),
                "rotation_used_yolk": bool(meta["rotation_used_yolk"]),
                "snip_processing_run_id": run_id,
                "snip_processing_version": version,
            }
        )

    manifest_df = pd.DataFrame(rows)
    validate_snip_manifest(manifest_df)

    out_parquet = contracts_dir / "snip_manifest.parquet"
    out_csv = contracts_dir / "snip_manifest.csv"
    manifest_df.to_parquet(out_parquet, index=False)
    if config.write_manifest_csv:
        manifest_df.to_csv(out_csv, index=False)

    (contracts_dir / ".snip_manifest.validated").write_text("ok\n")
    return out_parquet
