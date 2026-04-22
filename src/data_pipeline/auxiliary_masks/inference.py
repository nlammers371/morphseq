from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from skimage import io

from data_pipeline.feature_extraction.io.loaders import load_frame_contract
from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.auxiliary_masks import REQUIRED_COLUMNS_AUXILIARY_MASKS
from src.functions.core_utils_segmentation import Dataset, FishModel


AUXILIARY_MASK_SCHEMA_VERSION = 1
AUXILIARY_MASK_VERSION = "unet_auxiliary_masks_v1"
AUXILIARY_MASK_FAMILY_SPECS = {
    "via": {"checkpoint_name": "via_v1_0100", "n_classes": 1},
    "yolk": {"checkpoint_name": "yolk_v1_0050", "n_classes": 1},
    "focus": {"checkpoint_name": "focus_v0_0100", "n_classes": 1},
    "bubble": {"checkpoint_name": "bubble_v0_0100", "n_classes": 1},
}


def _resolve_frame_column(frame_df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in frame_df.columns:
        return preferred
    if fallback in frame_df.columns:
        return fallback
    raise KeyError(f"frame_contract is missing both {preferred!r} and {fallback!r}")


def _load_family_model(checkpoint_path: Path, n_classes: int, device: torch.device) -> FishModel:
    model = FishModel("FPN", "resnet34", in_channels=3, out_classes=n_classes)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
        state = {
            key.removeprefix("model."): value if key.startswith("model.") else value
            for key, value in state.items()
        }
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def _save_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(path, mask.astype(np.uint8), check_contrast=False)


def _predict_family_masks(
    *,
    frame_df: pd.DataFrame,
    data_root: Path,
    family: str,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    device: torch.device,
) -> dict[str, str]:
    spec = AUXILIARY_MASK_FAMILY_SPECS[family]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {family}: {checkpoint_path}")

    image_path_column = _resolve_frame_column(frame_df, "source_image_path", "stitched_image_path")
    microns_column = _resolve_frame_column(
        frame_df,
        "source_micrometers_per_pixel",
        "micrometers_per_pixel",
    )

    def _resolve_source_path(value: object) -> str:
        path = Path(str(value))
        if path.is_absolute():
            return str(path)
        return str(data_root / path)

    source_paths = [_resolve_source_path(p) for p in frame_df[image_path_column].tolist()]
    image_lookup = {_resolve_source_path(row[image_path_column]): str(row["image_id"]) for _, row in frame_df.iterrows()}

    dataset = Dataset(
        root=str(output_dir),
        filenames=source_paths,
        out_dims=(576, 320),
        num_classes=spec["n_classes"],
        predict_only_flag=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = _load_family_model(checkpoint_path, spec["n_classes"], device)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = logits.sigmoid()
            pr_max = torch.max(probs, axis=1)

            lb_predicted = pr_max.indices + 2
            lb_predicted[pr_max.values < 0.5] = 1
            lb_predicted = lb_predicted / (spec["n_classes"] + 1) * 255
            lb_predicted = np.asarray(lb_predicted.cpu()).astype(np.uint8)

            batch_paths = batch["path"]
            if isinstance(batch_paths, (str, Path)):
                batch_paths = [batch_paths]

            for idx, sample_path in enumerate(batch_paths):
                sample_path = str(sample_path)
                image_id = image_lookup.get(sample_path)
                if image_id is None:
                    raise KeyError(f"No image_id found for source_image_path={sample_path}")

                out_path = output_dir / f"{image_id}_{family}.png"
                if out_path.exists() and not overwrite:
                    written[image_id] = str(out_path)
                    continue

                _save_binary_mask(out_path, lb_predicted[idx])
                written[image_id] = str(out_path)

    return written


def run_auxiliary_mask_inference(
    *,
    frame_contract: Path,
    model_root: Path,
    output_root: Path,
    output_manifest_csv: Path,
    well_id: str | None = None,
    batch_size: int = 64,
    num_workers: int = 1,
    overwrite: bool = False,
) -> pd.DataFrame:
    frame_df = load_frame_contract(frame_contract)
    data_root = frame_contract.parent.parent.parent

    if well_id is not None:
        frame_df = frame_df.loc[frame_df["well_id"].astype(str) == str(well_id)].copy()

    if frame_df.empty:
        raise ValueError("frame_contract.csv is empty; cannot materialize auxiliary masks")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_rows = []

    family_paths: dict[str, dict[str, str]] = {}
    for family, spec in AUXILIARY_MASK_FAMILY_SPECS.items():
        checkpoint_path = model_root / spec["checkpoint_name"]
        family_output_dir = output_root / family
        family_paths[family] = _predict_family_masks(
            frame_df=frame_df,
            data_root=data_root,
            family=family,
            checkpoint_path=checkpoint_path,
            output_dir=family_output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            overwrite=overwrite,
            device=device,
        )

    for _, row in frame_df.iterrows():
        image_id = str(row["image_id"])
        source_image_value = row[_resolve_frame_column(frame_df, "source_image_path", "stitched_image_path")]
        source_image_path = Path(str(source_image_value))
        if not source_image_path.is_absolute():
            source_image_path = data_root / source_image_path

        source_mpp_value = row[
            _resolve_frame_column(frame_df, "source_micrometers_per_pixel", "micrometers_per_pixel")
        ]
        manifest_rows.append(
            {
                "schema_version": AUXILIARY_MASK_SCHEMA_VERSION,
                "experiment_id": str(row["experiment_id"]),
                "well_id": str(row["well_id"]),
                "well_index": str(row["well_index"]),
                "time_int": int(row["time_int"]),
                "image_id": image_id,
                "source_image_path": str(source_image_path),
                "source_micrometers_per_pixel": float(source_mpp_value),
                "image_width_px": int(row["image_width_px"]),
                "image_height_px": int(row["image_height_px"]),
                "via_mask_path": family_paths["via"][image_id],
                "yolk_mask_path": family_paths["yolk"][image_id],
                "focus_mask_path": family_paths["focus"][image_id],
                "bubble_mask_path": family_paths["bubble"][image_id],
                "auxiliary_mask_version": AUXILIARY_MASK_VERSION,
                "materialization_status": "generated",
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).loc[:, REQUIRED_COLUMNS_AUXILIARY_MASKS].copy()
    validate_dataframe_schema(manifest_df, REQUIRED_COLUMNS_AUXILIARY_MASKS, "auxiliary_masks.csv")
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_manifest_csv, index=False)
    (output_root / "contracts" / ".auxiliary_masks.validated").write_text("ok\n")
    return manifest_df
