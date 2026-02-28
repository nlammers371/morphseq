from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skimage.io as skio
from skimage.transform import resize

from .io import resolve_from_root, rel_to_root
from .process_snips import process_single_snip


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_gray_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        # best-effort scaling into [0,255]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _mask_to_binary(mask: np.ndarray, *, target_hw: tuple[int, int]) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)
    if (m.shape[0], m.shape[1]) != target_hw:
        m = resize(m.astype(float), target_hw, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        m = (m > 0).astype(np.uint8)
    return m


@dataclass(frozen=True)
class BackgroundStats:
    mean: float
    std: float
    definition: str


def estimate_background_stats_full_frame(
    *,
    rows: pd.DataFrame,
    output_root: Path,
    n_samples: int,
    seed: int,
    max_pixels_per_image: int = 200_000,
) -> BackgroundStats:
    """
    Locked v1 definition:
      background pixels = source stitched image pixels where embryo_mask == 0
      computed in full-frame space (before resize/rotate/crop)
    """
    definition = "full_frame_outside_embryo"
    if len(rows) == 0:
        return BackgroundStats(mean=128.0, std=30.0, definition=definition)

    # Deterministic sampling: sort then RNG.
    rows = rows.sort_values(["image_id", "snip_id"]).reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    n_avail = len(rows)
    if n_samples <= 0:
        n_samples = 1
    replace = n_samples > n_avail
    sample_idx = rng.choice(np.arange(n_avail), size=int(min(n_samples, n_avail) if not replace else n_samples), replace=replace)

    total = 0
    sum_x = 0.0
    sum_x2 = 0.0

    for i in sample_idx.tolist():
        r = rows.iloc[int(i)]
        img_path = resolve_from_root(str(r["source_image_path"]), output_root=output_root)
        mask_path = resolve_from_root(str(r["exported_mask_path"]), output_root=output_root)
        if not img_path.exists() or not mask_path.exists():
            continue

        img = _as_gray_u8(skio.imread(str(img_path)))
        mask = _mask_to_binary(skio.imread(str(mask_path)), target_hw=(img.shape[0], img.shape[1]))
        bg = img[mask == 0]
        if bg.size == 0:
            continue

        if max_pixels_per_image is not None and bg.size > int(max_pixels_per_image):
            pick = rng.choice(np.arange(bg.size), size=int(max_pixels_per_image), replace=False)
            bg = bg[pick]

        bg_f = bg.astype(np.float64)
        total += int(bg_f.size)
        sum_x += float(bg_f.sum())
        sum_x2 += float((bg_f * bg_f).sum())

    if total <= 1:
        return BackgroundStats(mean=128.0, std=30.0, definition=definition)

    mean = sum_x / total
    var = max(0.0, (sum_x2 / total) - (mean * mean))
    std = float(np.sqrt(var))
    return BackgroundStats(mean=float(mean), std=float(std), definition=definition)


def resolve_yolk_mask_path(
    *,
    row: pd.Series,
    output_root: Path,
    experiment_id: str,
    well_id: str,
) -> Path | None:
    """
    Optional yolk mask lookup.
    """
    # Future: if tracking contract ever carries yolk_mask_path, use it.
    if "yolk_mask_path" in row.index:
        v = str(row.get("yolk_mask_path") or "")
        if v:
            p = resolve_from_root(v, output_root=output_root)
            if p.exists():
                return p

    image_id = str(row["image_id"])
    snip_id = str(row["snip_id"])

    candidates = [
        output_root / "segmentation_aux_masks" / str(experiment_id) / "yolk" / f"{image_id}_yolk.png",
        output_root / "segmentation" / str(experiment_id) / "unet_masks" / "yolk" / f"{image_id}_yolk.png",
        output_root / "segmentation_and_tracking" / str(experiment_id) / "per_well" / str(well_id) / "masks" / "yolk_mask" / f"{snip_id}_mask.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def process_snip_row(
    *,
    row: pd.Series,
    output_root: Path,
    experiment_id: str,
    well_id: str,
    processed_dir: Path,
    raw_crops_dir: Path,
    save_raw_crops: bool,
    target_pixel_size_um: float,
    output_shape_hw: tuple[int, int],
    background_mean: float,
    background_std: float,
    blend_radius_um: float,
    yolk_enabled: bool,
) -> dict[str, Any]:
    """
    Process one snip and return manifest fields (including errors without raising).
    """
    snip_id = str(row["snip_id"])
    image_id = str(row["image_id"])
    embryo_id = str(row["embryo_id"])
    mask_type = str(row.get("mask_type", "embryo"))
    frame_index = int(row["frame_index"])

    source_image_abs = resolve_from_root(str(row["source_image_path"]), output_root=output_root)
    embryo_mask_abs = resolve_from_root(str(row["exported_mask_path"]), output_root=output_root)

    yolk_mask_abs = None
    rotation_source = "embryo_only"
    yolk_mask_rel = None
    if yolk_enabled:
        yolk_mask_abs = resolve_yolk_mask_path(row=row, output_root=output_root, experiment_id=experiment_id, well_id=well_id)
        if yolk_mask_abs is not None and yolk_mask_abs.exists():
            yolk_mask_rel = rel_to_root(yolk_mask_abs, output_root=output_root)
            # Best-effort: consider yolk-guided only if mask has any signal.
            try:
                y = skio.imread(str(yolk_mask_abs))
                yb = (np.asarray(y) > 0)
                if bool(np.any(yb)):
                    rotation_source = "yolk_guided"
            except Exception:
                rotation_source = "embryo_only"

    out: dict[str, Any] = {
        "snip_id": snip_id,
        "mask_type": mask_type,
        "experiment_id": str(experiment_id),
        "well_id": str(well_id),
        "well_index": row.get("well_index"),
        "image_id": image_id,
        "embryo_id": embryo_id,
        "frame_index": frame_index,
        "source_image_path": str(row["source_image_path"]),
        "exported_mask_path": str(row["exported_mask_path"]),
        "yolk_mask_path": yolk_mask_rel,
        "processed_snip_path": None,
        "raw_crop_path": None,
        "source_micrometers_per_pixel": float(row["micrometers_per_pixel"]),
        "target_pixel_size_um": float(target_pixel_size_um),
        "output_height_px": int(output_shape_hw[0]),
        "output_width_px": int(output_shape_hw[1]),
        "blend_radius_um": float(blend_radius_um),
        "background_mean": float(background_mean),
        "background_std": float(background_std),
        "background_definition": "full_frame_outside_embryo",
        "rotation_angle_rad": None,
        "rotation_angle_deg": None,
        "rotation_source": rotation_source,
        "processed_file_size_bytes": None,
        "raw_file_size_bytes": None,
        "is_valid": False,
        "error_message": None,
    }

    try:
        if not source_image_abs.exists():
            raise FileNotFoundError(f"source_image_path not found: {source_image_abs}")
        if not embryo_mask_abs.exists():
            raise FileNotFoundError(f"exported_mask_path not found: {embryo_mask_abs}")

        result = process_single_snip(
            snip_id=snip_id,
            image_path=source_image_abs,
            mask_path=embryo_mask_abs,
            yolk_mask_path=yolk_mask_abs,
            output_shape=output_shape_hw,
            pixel_size_um=float(row["micrometers_per_pixel"]),
            target_pixel_size_um=float(target_pixel_size_um),
            background_mean=float(background_mean),
            background_std=float(background_std),
            blend_radius_um=float(blend_radius_um),
            save_raw_crops=bool(save_raw_crops),
            raw_crops_dir=raw_crops_dir,
            processed_dir=processed_dir,
        )
        rot_rad = float(result.get("rotation_angle", 0.0))
        out["rotation_angle_rad"] = rot_rad
        out["rotation_angle_deg"] = float(np.rad2deg(rot_rad))

        processed_path = processed_dir / f"{snip_id}.jpg"
        out["processed_snip_path"] = rel_to_root(processed_path, output_root=output_root)
        if processed_path.exists():
            out["processed_file_size_bytes"] = int(processed_path.stat().st_size)
        if save_raw_crops:
            raw_path = raw_crops_dir / f"{snip_id}.tif"
            if raw_path.exists():
                out["raw_crop_path"] = rel_to_root(raw_path, output_root=output_root)
                out["raw_file_size_bytes"] = int(raw_path.stat().st_size)

        out["is_valid"] = bool(out["processed_file_size_bytes"] and out["processed_file_size_bytes"] > 0)
        return out
    except Exception as exc:
        out["error_message"] = f"{type(exc).__name__}: {exc}"
        out["is_valid"] = False
        return out
