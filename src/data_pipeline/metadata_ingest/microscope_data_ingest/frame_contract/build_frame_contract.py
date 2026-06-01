from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.frame_contract import REQUIRED_COLUMNS_FRAME_CONTRACT, UNIQUE_KEY_FRAME_CONTRACT
from data_pipeline.shared.identifiers import build_image_id
from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA




def _stitched_ff_path(
    *,
    built_image_data_root: Path,
    experiment_id: str,
    well_index: str,
    channel_id: str,
    time_int: int,
) -> Path:
    return (
        built_image_data_root
        / experiment_id
        / "stitched_ff_images"
        / well_index
        / channel_id
        / f"{well_index}_{channel_id}_t{int(time_int):04d}.jpg"
    )


def build_frame_contract(
    *,
    scope_metadata_mapped_csv: Path,
    built_image_data_root: Path,
    output_csv: Path,
    channel_id: str | None = "BF",
) -> pd.DataFrame:
    scope_df = pd.read_csv(scope_metadata_mapped_csv)
    validate_dataframe_schema(scope_df, REQUIRED_COLUMNS_SCOPE_METADATA, "scope_metadata_mapped.csv")

    df = scope_df.copy()
    if channel_id is not None and "channel" in df.columns:
        df = df[df["channel"].astype(str) == str(channel_id)].copy()
    if df.empty:
        raise ValueError(f"No rows left after channel filter channel_id={channel_id}")

    exp_id = str(df["experiment_id"].iloc[0])
    df["microscope_id"] = df["microscope_id"].astype(str)
    df["well_index"] = df["well_index"].astype(str)
    df["well_id"] = df["well_id"].astype(str)

    df["channel_id"] = df["channel"].astype(str)
    df["channel_name_raw"] = df["channel"].astype(str)

    fi = pd.to_numeric(df["time_int"], errors="raise").astype(int)
    df["time_int"] = fi

    df["image_id"] = [
        build_image_id(exp_id, str(w), str(ch), int(t))
        for w, ch, t in zip(df["well_id"], df["channel_id"], df["time_int"])
    ]

    df["stitched_image_path"] = [
        str(
            _stitched_ff_path(
                built_image_data_root=built_image_data_root,
                experiment_id=exp_id,
                well_index=str(w),
                channel_id=str(ch),
                time_int=int(f),
            ).resolve()
        )
        for w, ch, f in zip(df["well_index"], df["channel_id"], df["time_int"])
    ]
    df["source_image_path"] = df["stitched_image_path"]

    df["micrometers_per_pixel"] = pd.to_numeric(df["micrometers_per_pixel"], errors="raise").astype(float)
    df["frame_interval_s"] = pd.to_numeric(df["frame_interval_s"], errors="raise").astype(float)
    df["experiment_time_s"] = pd.to_numeric(df["experiment_time_s"], errors="coerce")
    df["absolute_start_time"] = df["absolute_start_time"].astype(str)
    df["image_width_px"] = pd.to_numeric(df["image_width_px"], errors="raise").astype(int)
    df["image_height_px"] = pd.to_numeric(df["image_height_px"], errors="raise").astype(int)
    df["objective_magnification"] = df["objective_magnification"].astype(str)

    out = df.loc[:, REQUIRED_COLUMNS_FRAME_CONTRACT].copy()
    validate_dataframe_schema(out, REQUIRED_COLUMNS_FRAME_CONTRACT, "frame_contract.csv")
    if out.duplicated(list(UNIQUE_KEY_FRAME_CONTRACT)).any():
        examples = (
            out.loc[out.duplicated(list(UNIQUE_KEY_FRAME_CONTRACT)), list(UNIQUE_KEY_FRAME_CONTRACT)]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"frame_contract uniqueness failed; examples: {examples}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    (output_csv.parent / ".frame_contract.validated").write_text("ok\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope-metadata-mapped-csv", type=Path, required=True)
    ap.add_argument("--built-image-data-root", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--channel-id", type=str, default="BF")
    args = ap.parse_args()

    build_frame_contract(
        scope_metadata_mapped_csv=args.scope_metadata_mapped_csv,
        built_image_data_root=args.built_image_data_root,
        output_csv=args.output_csv,
        channel_id=args.channel_id,
    )


if __name__ == "__main__":
    main()
