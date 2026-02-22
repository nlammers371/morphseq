"""Materialize stitched-image placeholders and emit stitched_image_index.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

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


def _infer_keyence_source_lookup(raw_images_dir: Path) -> dict[tuple[str, int], Path]:
    lookup: dict[tuple[str, int], Path] = {}
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

        key = (well_index, time_int)
        if key not in lookup:
            lookup[key] = path
    return lookup


def _write_placeholder_image(output_path: Path, width: int, height: int, value: int, overwrite: bool) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return "existing"

    image = np.zeros((max(height, 1), max(width, 1)), dtype=np.uint16)
    image[0, 0] = np.uint16(value % 65535)
    skio.imsave(output_path, image, check_contrast=False)
    return "materialized"


def materialize_stitched_images(
    experiment: str,
    microscope: str,
    raw_images_dir: Path,
    scope_csv: Path,
    output_root: Path,
    output_stitched_index_csv: Path,
    selected_wells: Iterable[str] | None,
    overwrite: bool,
    done_flag: Path | None = None,
) -> pd.DataFrame:
    """Materialize images for selected wells and emit stitched_image_index.csv."""
    scope_df = pd.read_csv(scope_csv)
    selected = _parse_selected_wells(selected_wells)
    if selected:
        scope_df = scope_df[scope_df["well_index"].astype(str).isin(selected)].copy()

    if scope_df.empty:
        raise ValueError(f"No scope rows available for experiment {experiment} after well filtering")

    if "channel_id" not in scope_df.columns:
        scope_df["channel_id"] = scope_df.get("channel", "BF").astype(str)

    scope_df = (
        scope_df.sort_values(["well_index", "channel_id", "time_int"])  # deterministic
        .drop_duplicates(subset=["experiment_id", "well_id", "well_index", "channel_id", "time_int"], keep="first")
        .copy()
    )

    stitched_root = output_root / experiment / "stitched_ff_images"
    data_root = output_root.parent

    yx1_nd2_files = sorted(raw_images_dir.glob("*.nd2")) if microscope == "YX1" else []
    yx1_source = yx1_nd2_files[0] if yx1_nd2_files else raw_images_dir

    keyence_lookup = _infer_keyence_source_lookup(raw_images_dir) if microscope == "Keyence" else {}

    rows = []
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

        output_path = stitched_root / well_index / channel_id / f"{image_id}.tif"

        width = int(float(row.get("image_width_px", 512)))
        height = int(float(row.get("image_height_px", 512)))
        status = _write_placeholder_image(output_path, width, height, frame_index, overwrite=overwrite)

        if microscope == "Keyence":
            source_artifact = keyence_lookup.get((well_index, time_int), raw_images_dir / _well_to_xy_dir_name(well_index))
            source_kind = "keyence_tiff_stack"
        else:
            source_artifact = yx1_source
            source_kind = "nd2"

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

    stitched_index_df = pd.DataFrame(rows)
    output_stitched_index_csv.parent.mkdir(parents=True, exist_ok=True)
    stitched_index_df.to_csv(output_stitched_index_csv, index=False)

    if done_flag is not None:
        done_flag.parent.mkdir(parents=True, exist_ok=True)
        done_flag.write_text("done\n")

    return stitched_index_df
