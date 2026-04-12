from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.frame_manifest import REQUIRED_COLUMNS_FRAME_MANIFEST, UNIQUE_KEY_FRAME_MANIFEST
from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA


def _stitched_ff_path(
    *,
    built_image_data_root: Path,
    experiment_id: str,
    well_index: str,
    channel_id: str,
    frame_index: int,
) -> Path:
    """
    Current stitched FF layout:
      built_image_data/{exp}/stitched_ff_images/{well}/{channel}/{well}_{channel}_t{frame:04d}.tif

    Note: on-disk filenames may still use `_t####`. Our canonical image_id uses `_f####`.
    """
    return (
        built_image_data_root
        / experiment_id
        / "stitched_ff_images"
        / well_index
        / channel_id
        / f"{well_index}_{channel_id}_t{int(frame_index):04d}.tif"
    )


def build_frame_manifest(
    *,
    scope_metadata_mapped_csv: Path,
    built_image_data_root: Path,
    output_parquet: Path,
    channel_id: str | None = "BF",
) -> pd.DataFrame:
    scope_df = pd.read_csv(scope_metadata_mapped_csv)
    # This is "mapped", but it is still a superset of the base scope schema.
    validate_dataframe_schema(scope_df, REQUIRED_COLUMNS_SCOPE_METADATA, "scope_metadata_mapped.csv")

    df = scope_df.copy()
    if channel_id is not None:
        df = df[df["channel"].astype(str) == str(channel_id)].copy()
    if df.empty:
        raise ValueError(f"No rows left after channel filter channel_id={channel_id}")

    df["microscope_id"] = df["microscope_id"].astype(str)
    df["well_index"] = df["well_index"].astype(str)
    df["well_id"] = df["well_id"].astype(str)

    df["channel_id"] = df["channel"].astype(str)
    fi = pd.to_numeric(df["frame_index"], errors="raise").astype(int)
    df["frame_index"] = fi
    df["time_int"] = fi

    # Canonical image_id uses frame suffix `_f`.
    df["image_id"] = (
        df["well_id"].astype(str)
        + "_"
        + df["channel_id"].astype(str)
        + "_f"
        + df["frame_index"].map(lambda x: f"{int(x):04d}")
    )

    exp_id = str(df["experiment_id"].iloc[0])
    df["source_image_path"] = [
        str(
            _stitched_ff_path(
                built_image_data_root=built_image_data_root,
                experiment_id=exp_id,
                well_index=str(w),
                channel_id=str(ch),
                frame_index=int(f),
            ).resolve()
        )
        for w, ch, f in zip(df["well_index"], df["channel_id"], df["frame_index"])
    ]

    df["source_micrometers_per_pixel"] = pd.to_numeric(df["micrometers_per_pixel"], errors="raise").astype(float)
    df["image_width_px"] = pd.to_numeric(df["image_width_px"], errors="raise").astype(int)
    df["image_height_px"] = pd.to_numeric(df["image_height_px"], errors="raise").astype(int)

    out = df.loc[:, REQUIRED_COLUMNS_FRAME_MANIFEST].copy()
    validate_dataframe_schema(out, REQUIRED_COLUMNS_FRAME_MANIFEST, "frame_manifest")

    if out.duplicated(UNIQUE_KEY_FRAME_MANIFEST).any():
        examples = (
            out.loc[out.duplicated(UNIQUE_KEY_FRAME_MANIFEST), UNIQUE_KEY_FRAME_MANIFEST]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"frame_manifest uniqueness failed; examples: {examples}")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, index=False)
    (output_parquet.parent / ".frame_manifest.validated").write_text("ok\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope-metadata-mapped-csv", type=Path, required=True)
    ap.add_argument("--built-image-data-root", type=Path, required=True)
    ap.add_argument("--output-parquet", type=Path, required=True)
    ap.add_argument("--channel-id", type=str, default="BF")
    args = ap.parse_args()

    build_frame_manifest(
        scope_metadata_mapped_csv=args.scope_metadata_mapped_csv,
        built_image_data_root=args.built_image_data_root,
        output_parquet=args.output_parquet,
        channel_id=args.channel_id,
    )


if __name__ == "__main__":
    main()

