"""Compute per-snip predicted developmental stage (HPF) using temperature-rate formula."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.feature_extraction.stage_inference import predict_stage_hpf
from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.frame_contract import REQUIRED_COLUMNS_FRAME_CONTRACT
from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
from data_pipeline.schemas.stage_predictions import REQUIRED_COLUMNS_STAGE_PREDICTIONS


def _coalesce_column(df: pd.DataFrame, candidates: list[str], *, required: bool = True) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            series = df[name]
            if not series.isna().all():
                return series
    if required:
        raise ValueError(f"None of the expected columns are present with values: {candidates}")
    return pd.Series(index=df.index, dtype="object")


def _pipeline_version() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[4]
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return sha[:12]
    except Exception:
        return "unknown"


def compute_stage_predictions_well(
    *,
    output_root: Path,
    experiment_id: str,
    well_id: str,
    segmentation_tracking_csv: Path,
    frame_contract_csv: Path,
    plate_metadata_csv: Path,
) -> dict[str, Path]:
    output_root = Path(output_root)
    exp = str(experiment_id)
    well = str(well_id)

    out_dir = output_root / "computed_features" / exp / "per_well" / well / "contracts"
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = pd.read_csv(segmentation_tracking_csv)
    if seg.empty:
        raise ValueError(f"Empty segmentation_tracking: {segmentation_tracking_csv}")
    # Keep embryo masks only if multiple mask heads are present.
    if "mask_type" in seg.columns:
        seg = seg[seg["mask_type"].astype(str) == "embryo"].copy()
    if seg.empty:
        raise ValueError(f"No embryo rows in segmentation_tracking: {segmentation_tracking_csv}")

    fm = pd.read_csv(frame_contract_csv)
    validate_dataframe_schema(fm, REQUIRED_COLUMNS_FRAME_CONTRACT, "frame_contract")
    fm = fm[fm["experiment_id"].astype(str) == exp].copy()
    fm = fm[fm["well_id"].astype(str) == well].copy()
    if fm.empty:
        raise ValueError(f"No frame_contract rows for experiment={exp} well_id={well}.")

    plate = pd.read_csv(plate_metadata_csv)
    validate_dataframe_schema(plate, REQUIRED_COLUMNS_PLATE_METADATA, "plate_metadata")
    plate = plate[(plate["experiment_id"].astype(str) == exp) & (plate["well_id"].astype(str) == well)].copy()
    if plate.empty:
        raise ValueError(f"No plate_metadata rows for experiment={exp} well_id={well}.")
    if len(plate) != 1:
        raise ValueError(f"Expected 1 plate_metadata row for {exp}/{well}, found {len(plate)}.")

    start_age_hpf = float(plate["start_age_hpf"].iloc[0])
    temperature = float(plate["temperature"].iloc[0])

    # Join seg -> frame_contract to get elapsed_time_s.
    seg["image_id"] = seg["image_id"].astype(str)
    fm["image_id"] = fm["image_id"].astype(str)
    merged = seg.merge(
        fm[["image_id", "elapsed_time_s", "well_index"]].copy(),
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    if merged["elapsed_time_s"].isna().any():
        preview = merged.loc[merged["elapsed_time_s"].isna(), ["snip_id", "image_id"]].head(10).to_dict(orient="records")
        raise ValueError(f"Missing elapsed_time_s after join to frame_contract (preview): {preview}")
    merged_well_index = _coalesce_column(merged, ["well_index", "well_index_x", "well_index_y"]).astype(str)

    # Compute predictions.
    merged["elapsed_time_s"] = merged["elapsed_time_s"].astype(float)
    developmental_rate = 0.055 * temperature - 0.57

    predicted = merged["elapsed_time_s"].map(lambda t: predict_stage_hpf(start_age_hpf, float(t), temperature))
    out = pd.DataFrame(
        {
            "experiment_id": exp,
            "well_id": well,
            "well_index": merged_well_index,
            "image_id": merged["image_id"].astype(str),
            "embryo_id": merged["embryo_id"].astype(str),
            "time_int": merged["time_int"].astype(int),
            "snip_id": merged["snip_id"].astype(str),
            "elapsed_time_s": merged["elapsed_time_s"].astype(float),
            "start_age_hpf": float(start_age_hpf),
            "temperature": float(temperature),
            "developmental_rate_hpf_per_h": float(developmental_rate),
            "predicted_stage_hpf": predicted.astype(float),
            "stage_confidence": 1.0,
            "stage_model": "kimmel1995_temp_rate",
            "pipeline_version": _pipeline_version(),
        }
    )

    validate_dataframe_schema(out, REQUIRED_COLUMNS_STAGE_PREDICTIONS, "stage_predictions")

    # Deterministic order.
    out = out.sort_values(["time_int", "image_id", "embryo_id", "snip_id"]).reset_index(drop=True)

    out_pq = out_dir / "stage_predictions.parquet"
    out_csv = out_dir / "stage_predictions.csv"
    out.to_parquet(out_pq, index=False)
    out.to_csv(out_csv, index=False)

    flag = out_dir / ".stage_predictions.computed"
    flag.write_text("computed\n")
    return {"parquet": out_pq, "csv": out_csv, "computed_flag": flag}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--well-id", required=True)
    p.add_argument("--segmentation-tracking-csv", type=Path, required=True)
    p.add_argument("--frame-contract-csv", type=Path, required=True)
    p.add_argument("--plate-metadata-csv", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    compute_stage_predictions_well(
        output_root=args.output_root,
        experiment_id=str(args.experiment),
        well_id=str(args.well_id),
        segmentation_tracking_csv=args.segmentation_tracking_csv,
        frame_contract_csv=args.frame_contract_csv,
        plate_metadata_csv=args.plate_metadata_csv,
    )


if __name__ == "__main__":
    main()
