"""
Auxiliary mask materialization harness.

This stage owns the auxiliary-mask output contract for now. Long term, the
implementation should move into the main `data_pipeline/segmentation` layer so
the pipeline stays fully self-contained, while this wrapper keeps the run
interface thin and explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.feature_extraction.io.loaders import load_frame_contract
from data_pipeline.schemas.auxiliary_masks import REQUIRED_COLUMNS_AUXILIARY_MASKS
from .paths import (
    AUXILIARY_MASK_FAMILIES,
    auxiliary_mask_path,
    auxiliary_mask_sentinel_path,
    auxiliary_mask_subdir,
)


AUXILIARY_MASK_SCHEMA_VERSION = 1
AUXILIARY_MASK_VERSION = "placeholder_zero_masks_v1"


def _write_zero_mask(path: Path, height_px: int, width_px: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    mask = np.zeros((height_px, width_px), dtype=np.uint8)
    io.imsave(path, mask, check_contrast=False)


def materialize_auxiliary_masks(
    frame_contract: Path,
    output_root: Path,
    output_manifest_csv: Path,
) -> pd.DataFrame:
    frame_df = load_frame_contract(frame_contract)
    data_root = output_root.parent.parent
    experiment_id = output_root.name

    manifest_rows = []
    for _, row in frame_df.iterrows():
        row_experiment_id = str(row["experiment_id"])
        image_id = str(row["image_id"])
        well_id = str(row["well_id"])
        well_index = str(row["well_index"])
        frame_index = int(row["frame_index"])
        height_px = int(row["image_height_px"])
        width_px = int(row["image_width_px"])
        created_placeholder = False

        row_payload = {
            "schema_version": AUXILIARY_MASK_SCHEMA_VERSION,
            "experiment_id": row_experiment_id,
            "well_id": well_id,
            "well_index": well_index,
            "frame_index": frame_index,
            "image_id": image_id,
            "source_image_path": str(row["source_image_path"]),
            "source_micrometers_per_pixel": float(row["micrometers_per_pixel"]),
            "image_width_px": width_px,
            "image_height_px": height_px,
            "auxiliary_mask_version": AUXILIARY_MASK_VERSION,
        }

        for family in AUXILIARY_MASK_FAMILIES:
            mask_path = auxiliary_mask_path(data_root, row_experiment_id, family, image_id)
            if not mask_path.exists():
                _write_zero_mask(mask_path, height_px, width_px)
                created_placeholder = True
            row_payload[f"{family}_mask_path"] = str(mask_path)

        row_payload["materialization_status"] = (
            "placeholder_zero_masks" if created_placeholder else "preserved_existing_masks"
        )
        manifest_rows.append(row_payload)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df = manifest_df.loc[:, REQUIRED_COLUMNS_AUXILIARY_MASKS].copy()
    validate_dataframe_schema(
        manifest_df,
        REQUIRED_COLUMNS_AUXILIARY_MASKS,
        "auxiliary_masks.csv",
    )
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_manifest_csv, index=False)
    auxiliary_mask_sentinel_path(data_root, experiment_id).write_text("ok\n")
    return manifest_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-contract", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--output-manifest-csv", type=Path, required=True)
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    data_root = args.output_root.parent.parent
    for family in AUXILIARY_MASK_FAMILIES:
        auxiliary_mask_subdir(data_root, args.output_root.name, family).mkdir(parents=True, exist_ok=True)

    materialize_auxiliary_masks(
        frame_contract=args.frame_contract,
        output_root=args.output_root,
        output_manifest_csv=args.output_manifest_csv,
    )


if __name__ == "__main__":
    main()
