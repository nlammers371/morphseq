from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from data_pipeline.schemas.snip_processing import REQUIRED_COLUMNS_SNIP_MANIFEST


def validate_snip_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS_SNIP_MANIFEST if c not in df.columns]
    if missing:
        raise ValueError(f"snip_manifest missing required columns: {missing}")

    if df.empty:
        raise ValueError("snip_manifest is empty")

    if df["snip_id"].isna().any():
        raise ValueError("snip_id contains nulls")

    if df["snip_id"].duplicated().any():
        dups = df.loc[df["snip_id"].duplicated(), "snip_id"].astype(str).head(10).tolist()
        raise ValueError(f"snip_id is not unique; examples: {dups}")

    # Required non-null fields (nullable columns excluded).
    required_non_null = [
        "schema_version",
        "experiment_id",
        "well_id",
        "well_index",
        "image_id",
        "frame_index",
        "channel_id",
        "embryo_id",
        "instance_id",
        "source_image_path",
        "embryo_mask_path",
        "source_micrometers_per_pixel",
        "frame_snapshot_hash",
        "processed_snip_path",
        "target_pixel_size_um",
        "output_height_px",
        "output_width_px",
        "blend_radius_um",
        "background_mean",
        "background_std",
        "rotation_angle_rad",
        "rotation_angle_deg",
        "rotation_used_yolk",
        "snip_processing_run_id",
        "snip_processing_version",
    ]
    for c in required_non_null:
        if df[c].isna().any():
            raise ValueError(f"snip_manifest column '{c}' contains nulls")

    # snip_id invariant.
    frame_index = pd.to_numeric(df["frame_index"], errors="raise").astype(int)
    expected = (
        df["embryo_id"].astype(str)
        + "_"
        + df["channel_id"].astype(str)
        + "_f"
        + frame_index.map(lambda x: f"{x:04d}")
    )
    bad = df["snip_id"].astype(str) != expected
    if bad.any():
        examples = df.loc[bad, ["snip_id", "embryo_id", "channel_id", "frame_index"]].head(10).to_dict("records")
        raise ValueError(f"snip_id invariant failed; examples: {examples}")

    # File existence checks.
    def _check_paths(col: str, *, allow_null: bool) -> None:
        series = df[col]
        if allow_null:
            series = series.dropna()
        else:
            if series.isna().any():
                raise ValueError(f"{col} contains nulls")

        missing_paths: list[str] = []
        for p in series.astype(str):
            if not p:
                missing_paths.append(p)
                continue
            if not Path(p).exists():
                missing_paths.append(p)
            if len(missing_paths) >= 10:
                break
        if missing_paths:
            raise ValueError(f"{col} has missing files; examples: {missing_paths[:10]}")

    _check_paths("source_image_path", allow_null=False)
    _check_paths("embryo_mask_path", allow_null=False)
    _check_paths("processed_snip_path", allow_null=False)
    _check_paths("raw_crop_path", allow_null=True)
    _check_paths("yolk_mask_path", allow_null=True)

