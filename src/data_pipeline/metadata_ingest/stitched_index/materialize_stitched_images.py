"""Materialize stitched images from raw inputs and emit stitched_image_index.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import nd2
import numpy as np
import pandas as pd
import skimage.io as skio


def _parse_selected_wells(selected_wells: Iterable[str] | None) -> set[str]:
    return {str(w).strip() for w in (selected_wells or []) if str(w).strip()}


def _well_to_xy_dir_name(well_index: str) -> str:
    row = ord(well_index[0].upper()) - ord("A")
    col = int(well_index[1:])
    xy_idx = row * 12 + col
    return f"XY{xy_idx:02d}"


def _normalize_extension(extension: str | None) -> str:
    ext = (extension or "jpg").strip().lower()
    if ext.startswith("."):
        ext = ext[1:]
    if ext not in {"jpg", "jpeg", "png", "tif", "tiff"}:
        raise ValueError(f"Unsupported output image extension: {extension}")
    return "jpg" if ext == "jpeg" else ("tif" if ext == "tiff" else ext)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    im = image.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    if im_max <= im_min:
        return np.zeros_like(im, dtype=np.uint8)
    scaled = (im - im_min) / (im_max - im_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _write_image(output_path: Path, image: np.ndarray, overwrite: bool, image_extension: str) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return "existing"

    if image_extension in {"jpg", "png"}:
        image = _to_uint8(image)
    elif image.dtype != np.uint16:
        image = image.astype(np.uint16)

    skio.imsave(output_path, image, check_contrast=False)
    return "materialized"


def _write_placeholder_image(
    output_path: Path, width: int, height: int, value: int, overwrite: bool, image_extension: str
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return "existing"

    if image_extension in {"jpg", "png"}:
        image = np.zeros((max(height, 1), max(width, 1)), dtype=np.uint8)
        image[0, 0] = np.uint8(value % 255)
    else:
        image = np.zeros((max(height, 1), max(width, 1)), dtype=np.uint16)
        image[0, 0] = np.uint16(value % 65535)

    skio.imsave(output_path, image, check_contrast=False)
    return "placeholder"


def _build_yx1_well_to_series_map(mapping_csv: Path | None) -> dict[str, int]:
    if mapping_csv is None or not mapping_csv.exists():
        return {}
    mapping_df = pd.read_csv(mapping_csv)
    out: dict[str, int] = {}
    for _, row in mapping_df.iterrows():
        series_number = row.get("series_number")
        well_index = row.get("well_index")
        if pd.isna(series_number) or pd.isna(well_index):
            continue
        # Convert to 0-based ND2 series index.
        out[str(well_index)] = int(series_number) - 1
    return out


def _select_yx1_channel_index(channel_names: list[str]) -> int:
    lower = [str(name).lower() for name in channel_names]
    if "bf" in lower:
        return lower.index("bf")
    for idx, name in enumerate(lower):
        if any(token in name for token in ("dia", "brightfield", "empty")):
            return idx
    return 0


def _materialize_yx1_image(
    nd: nd2.ND2File,
    dask_arr,
    time_int: int,
    series_index: int,
    channel_index: int,
) -> np.ndarray:
    if dask_arr.ndim == 6:
        stack = dask_arr[time_int, series_index, :, channel_index, :, :].compute()
    elif dask_arr.ndim == 5:
        stack = dask_arr[time_int, series_index, :, :, :].compute()
    else:
        raise ValueError(f"Unexpected ND2 dimensions: {dask_arr.ndim}")

    projected = np.max(stack, axis=0)
    if projected.dtype != np.uint16:
        projected = projected.astype(np.uint16)
    return projected


def _infer_keyence_stack_lookup(raw_images_dir: Path) -> dict[tuple[str, int], list[Path]]:
    lookup: dict[tuple[str, int], list[Path]] = {}
    for path in raw_images_dir.rglob("*CH*.tif"):
        well_index = None
        for part in path.parts:
            if part.startswith("XY") and part[2:].isdigit():
                row = (int(part[2:]) - 1) // 12
                col = (int(part[2:]) - 1) % 12 + 1
                well_index = f"{chr(65 + row)}{col:02d}"
                break
        if well_index is None:
            continue

        match = re.search(r"_(\d+)_Z\d+_CH\d+", path.name)
        if not match:
            continue
        time_raw = int(match.group(1))
        time_int = max(time_raw - 1, 0)

        lookup.setdefault((well_index, time_int), []).append(path)

    for key, values in lookup.items():
        values.sort()
    return lookup


def _materialize_keyence_image(stack_paths: list[Path]) -> np.ndarray:
    frames = [skio.imread(path) for path in stack_paths]
    stack = np.stack(frames, axis=0)
    projected = np.max(stack, axis=0)
    if projected.dtype != np.uint16:
        projected = projected.astype(np.uint16)
    return projected


def materialize_stitched_images(
    experiment: str,
    microscope: str,
    raw_images_dir: Path,
    scope_csv: Path,
    mapping_csv: Path | None,
    output_root: Path,
    output_stitched_index_csv: Path,
    selected_wells: Iterable[str] | None,
    overwrite: bool,
    output_image_extension: str = "jpg",
    done_flag: Path | None = None,
) -> pd.DataFrame:
    """Materialize stitched images for selected wells and emit stitched-image index."""
    scope_df = pd.read_csv(scope_csv)
    selected = _parse_selected_wells(selected_wells)
    if selected:
        scope_df = scope_df[scope_df["well_index"].astype(str).isin(selected)].copy()

    if scope_df.empty:
        raise ValueError(f"No scope rows available for experiment {experiment} after well filtering")

    if "channel_id" not in scope_df.columns:
        scope_df["channel_id"] = scope_df.get("channel", "BF").astype(str)

    scope_df = (
        scope_df.sort_values(["well_index", "channel_id", "time_int"])
        .drop_duplicates(subset=["experiment_id", "well_id", "well_index", "channel_id", "time_int"], keep="first")
        .copy()
    )

    stitched_root = output_root / experiment / "stitched_ff_images"
    data_root = output_root.parent

    yx1_nd2_files = sorted(raw_images_dir.glob("*.nd2")) if microscope == "YX1" else []
    yx1_source = yx1_nd2_files[0] if yx1_nd2_files else raw_images_dir
    keyence_lookup = _infer_keyence_stack_lookup(raw_images_dir) if microscope == "Keyence" else {}
    yx1_series_map = _build_yx1_well_to_series_map(mapping_csv) if microscope == "YX1" else {}
    image_extension = _normalize_extension(output_image_extension)

    nd = None
    dask_arr = None
    channel_index = 0
    if microscope == "YX1" and yx1_nd2_files:
        nd = nd2.ND2File(yx1_nd2_files[0])
        dask_arr = nd.to_dask()
        channel_names = [ch.channel.name for ch in nd.frame_metadata(0).channels]
        channel_index = _select_yx1_channel_index(channel_names)

    rows = []
    try:
        for _, row in scope_df.iterrows():
            well_index = str(row["well_index"])
            channel_id = str(row.get("channel_id", row.get("channel", "BF")))
            well_id = str(row["well_id"])
            time_int = int(row["time_int"])
            frame_index = int(row.get("frame_index", time_int))

            image_id = row.get("image_id")
            if pd.isna(image_id) or not str(image_id):
                image_id = f"{well_id}_{channel_id}_t{frame_index:04d}"
            image_id = str(image_id)

            output_path = stitched_root / well_index / channel_id / f"{image_id}.{image_extension}"
            width = int(float(row.get("image_width_px", 512)))
            height = int(float(row.get("image_height_px", 512)))

            status = "placeholder"
            source_artifact: Path
            source_kind: str

            if microscope == "YX1" and nd is not None and dask_arr is not None:
                series_index = yx1_series_map.get(well_index)
                if series_index is not None:
                    image = _materialize_yx1_image(
                        nd=nd,
                        dask_arr=dask_arr,
                        time_int=time_int,
                        series_index=series_index,
                        channel_index=channel_index,
                    )
                    status = _write_image(output_path, image, overwrite=overwrite, image_extension=image_extension)
                else:
                    status = _write_placeholder_image(
                        output_path,
                        width,
                        height,
                        frame_index,
                        overwrite=overwrite,
                        image_extension=image_extension,
                    )
                source_artifact = yx1_source
                source_kind = "yx1_nd2_zstack"
            elif microscope == "Keyence":
                stack_paths = keyence_lookup.get((well_index, time_int), [])
                if stack_paths:
                    image = _materialize_keyence_image(stack_paths)
                    status = _write_image(output_path, image, overwrite=overwrite, image_extension=image_extension)
                    source_artifact = stack_paths[0]
                else:
                    status = _write_placeholder_image(
                        output_path,
                        width,
                        height,
                        frame_index,
                        overwrite=overwrite,
                        image_extension=image_extension,
                    )
                    source_artifact = raw_images_dir / _well_to_xy_dir_name(well_index)
                source_kind = "keyence_tiff_zstack"
            else:
                status = _write_placeholder_image(
                    output_path,
                    width,
                    height,
                    frame_index,
                    overwrite=overwrite,
                    image_extension=image_extension,
                )
                source_artifact = raw_images_dir
                source_kind = "placeholder"

            stitched_rel = output_path.relative_to(data_root)
            rows.append(
                {
                    "experiment_id": experiment,
                    "microscope_id": microscope,
                    "well_id": well_id,
                    "well_index": well_index,
                    "channel_id": channel_id,
                    "time_int": time_int,
                    "frame_index": frame_index,
                    "image_id": image_id,
                    "stitched_image_path": str(stitched_rel),
                    "materialization_status": status,
                    "source_artifact_path": str(source_artifact),
                    "source_artifact_kind": source_kind,
                    "image_width_px": width,
                    "image_height_px": height,
                }
            )
    finally:
        if nd is not None:
            nd.close()

    stitched_index_df = pd.DataFrame(rows)
    output_stitched_index_csv.parent.mkdir(parents=True, exist_ok=True)
    stitched_index_df.to_csv(output_stitched_index_csv, index=False)

    if done_flag is not None:
        done_flag.parent.mkdir(parents=True, exist_ok=True)
        done_flag.write_text("done\n")

    return stitched_index_df
