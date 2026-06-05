from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.frame_contract import REQUIRED_COLUMNS_FRAME_CONTRACT, UNIQUE_KEY_FRAME_CONTRACT
from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA


def build_frame_contract(
    *,
    scope_metadata_mapped_csv: Path,
    stitched_inventory_csv: Path,
    output_csv: Path,
    channel_id: str | None = "BF",
) -> pd.DataFrame:
    # Load and validate stitched inventory — this is the source of truth for
    # which wells were actually built in this run.
    inventory_df = pd.read_csv(stitched_inventory_csv)
    inventory_wells = set(inventory_df["well_index"].astype(str))
    inventory_image_format = {
        str(r["well_index"]): str(r["image_format"])
        for _, r in inventory_df.iterrows()
    }
    inventory_image_dir = {
        str(r["well_index"]): Path(str(r["stitched_image_dir"]))
        for _, r in inventory_df.iterrows()
    }

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

    # Restrict scope metadata to inventory wells (all currently built wells).
    df = df[df["well_index"].isin(inventory_wells)].copy()

    if df.empty:
        raise ValueError("No scope metadata rows matched the stitched inventory wells.")

    df["channel_id"] = df["channel"].astype(str)
    df["channel_name_raw"] = df["channel"].astype(str)

    fi = pd.to_numeric(df["frame_index"], errors="raise").astype(int)
    df["frame_index"] = fi
    df["time_int"] = fi

    df["image_id"] = (
        df["well_id"].astype(str)
        + "_"
        + df["channel_id"].astype(str)
        + "_f"
        + df["frame_index"].map(lambda x: f"{int(x):04d}")
    )

    def _stitched_path(well_index: str, channel_id: str, frame_index: int) -> str:
        img_dir = inventory_image_dir[well_index]
        fmt = inventory_image_format[well_index]
        return str((img_dir / channel_id / f"{well_index}_{channel_id}_t{int(frame_index):04d}.{fmt}").resolve())

    df["stitched_image_path"] = [
        _stitched_path(str(w), str(ch), int(f))
        for w, ch, f in zip(df["well_index"], df["channel_id"], df["frame_index"])
    ]
    df["source_image_path"] = df["stitched_image_path"]

    df["micrometers_per_pixel"] = pd.to_numeric(df["micrometers_per_pixel"], errors="raise").astype(float)
    df["frame_interval_s"] = pd.to_numeric(df["frame_interval_s"], errors="raise").astype(float)
    df["experiment_time_s"] = pd.to_numeric(df["experiment_time_s"], errors="coerce")
    df["absolute_start_time"] = df["absolute_start_time"].astype(str)
    df["image_width_px"] = pd.to_numeric(df["image_width_px"], errors="raise").astype(int)
    df["image_height_px"] = pd.to_numeric(df["image_height_px"], errors="raise").astype(int)
    df["objective_magnification"] = (
        pd.to_numeric(df["objective_magnification"], errors="coerce")
        .fillna(df["objective_magnification"].astype(str).str.extract(r"(\d+(?:\.\d+)?)x", expand=False).pipe(pd.to_numeric, errors="coerce"))
    )

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
    ap.add_argument("--stitched-inventory", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--channel-id", type=str, default="BF")
    args = ap.parse_args()

    build_frame_contract(
        scope_metadata_mapped_csv=args.scope_metadata_mapped_csv,
        stitched_inventory_csv=args.stitched_inventory,
        output_csv=args.output_csv,
        channel_id=args.channel_id,
    )


if __name__ == "__main__":
    main()
