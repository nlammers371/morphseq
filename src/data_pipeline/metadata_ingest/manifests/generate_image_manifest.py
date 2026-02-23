"""
Image Manifest Generation

This module generates experiment_image_manifest.json, the single source of truth
for per-well, per-channel frame ordering.

MVP Requirements:
- Read scope_and_plate_metadata.csv
- Scan built_image_data/{exp}/stitched_ff_images/
- Validate channel normalization (BF required)
- Sort frames by time_int
- Write validated JSON

Schema: Uses schemas/image_manifest.py
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Import schemas
from src.data_pipeline.schemas.image_manifest import (
    REQUIRED_EXPERIMENT_FIELDS,
    REQUIRED_WELL_FIELDS,
    REQUIRED_CHANNEL_FIELDS,
    REQUIRED_FRAME_FIELDS,
)
from src.data_pipeline.schemas.channel_normalization import (
    CHANNEL_NORMALIZATION_MAP,
    VALID_CHANNEL_NAMES,
    BRIGHTFIELD_CHANNELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def normalize_channel_name(raw_channel_name: str) -> str:
    """
    Normalize channel name using standard mapping.

    Args:
        raw_channel_name: Raw channel name from microscope

    Returns:
        Normalized channel name (e.g., 'BF', 'GFP')

    Raises:
        ValueError: If channel name cannot be normalized
    """
    # Direct match
    if raw_channel_name in CHANNEL_NORMALIZATION_MAP:
        return CHANNEL_NORMALIZATION_MAP[raw_channel_name]

    # Case-insensitive match
    for raw, normalized in CHANNEL_NORMALIZATION_MAP.items():
        if raw.lower() == raw_channel_name.lower():
            return normalized

    # Already normalized
    if raw_channel_name in VALID_CHANNEL_NAMES:
        return raw_channel_name

    # Give up
    raise ValueError(
        f"Unknown channel name: '{raw_channel_name}'. "
        f"Valid names: {VALID_CHANNEL_NAMES}. "
        f"Add to CHANNEL_NORMALIZATION_MAP if this is a new microscope channel."
    )


def scan_stitched_images(
    stitched_root: Path,
    exp_name: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scan stitched_ff_images directory and build frame inventory.

    Directory structure:
        built_image_data/{exp}/stitched_ff_images/{well}/{channel}/{well}_{channel}_t{time_int:04d}.tif

    Returns:
        Dict mapping well_id -> list of frame dicts with:
            - channel_name
            - raw_channel_name
            - time_int
            - file_path (relative to data root)
            - image_width_px
            - image_height_px
    """
    stitched_dir = stitched_root / exp_name / "stitched_ff_images"

    if not stitched_dir.exists():
        raise FileNotFoundError(f"Stitched images not found: {stitched_dir}")

    well_frames = {}

    # Scan wells
    for well_dir in sorted(stitched_dir.iterdir()):
        if not well_dir.is_dir():
            continue

        well_name = well_dir.name  # e.g., 'A01'
        well_id = f"{exp_name}_{well_name}"

        # Scan channels
        for channel_dir in sorted(well_dir.iterdir()):
            if not channel_dir.is_dir():
                continue

            channel_name = channel_dir.name  # Should already be normalized (e.g., 'BF')

            # Validate channel name
            try:
                normalized = normalize_channel_name(channel_name)
            except ValueError as e:
                log.warning("Skipping invalid channel in %s: %s", well_dir.name, e)
                continue

            # Scan frames
            for frame_file in sorted(channel_dir.glob("*.tif")):
                # Parse filename: {well}_{channel}_t{time_int:04d}.tif
                stem = frame_file.stem
                parts = stem.split("_t")

                if len(parts) != 2:
                    log.warning("Unexpected frame filename: %s", frame_file.name)
                    continue

                try:
                    time_int = int(parts[1])
                except ValueError:
                    log.warning("Could not parse time_int from: %s", frame_file.name)
                    continue

                # Get image dimensions (read from file if needed)
                # For MVP, we'll populate from metadata CSV; placeholder here
                image_width_px = None
                image_height_px = None

                # Build frame dict
                frame_info = {
                    "channel_name": normalized,
                    "raw_channel_name": channel_name,
                    "time_int": time_int,
                    "file_path": str(frame_file.relative_to(stitched_root.parent)),
                    "image_width_px": image_width_px,
                    "image_height_px": image_height_px,
                }

                if well_id not in well_frames:
                    well_frames[well_id] = []

                well_frames[well_id].append(frame_info)

    log.info("Scanned %d wells with frames", len(well_frames))
    return well_frames


def generate_image_manifest(
    metadata_root: Path,
    stitched_root: Path,
    output_root: Path,
    exp_name: str,
    microscope_id: str,
):
    """
    Generate experiment_image_manifest.json.

    Args:
        metadata_root: Path to experiment_metadata/
        stitched_root: Path to built_image_data/
        output_root: Path to experiment_metadata/
        exp_name: Experiment name
        microscope_id: Microscope identifier (e.g., 'YX1', 'Keyence')

    Output:
        experiment_metadata/{exp_name}/experiment_image_manifest.json
    """
    log.info("Generating image manifest for %s", exp_name)

    # Read scope_and_plate_metadata.csv
    metadata_csv = metadata_root / exp_name / "scope_and_plate_metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_csv}")

    metadata_df = pd.read_csv(metadata_csv)
    log.info("Loaded metadata: %d rows", len(metadata_df))

    # Scan stitched images
    well_frames = scan_stitched_images(stitched_root, exp_name)

    # Build manifest structure
    manifest = {
        "experiment_id": exp_name,
        "microscope_id": microscope_id,
        "created_at": datetime.now().isoformat(),
        "total_wells": 0,
        "total_channels": 0,
        "total_frames": 0,
        "wells": []
    }

    all_channels = set()
    total_frames = 0

    # Group frames by well
    for well_id, frames in sorted(well_frames.items()):
        well_index = well_id.split("_")[-1]  # e.g., 'A01'

        # Get well metadata
        well_meta = metadata_df[metadata_df["well_id"] == well_id]

        if well_meta.empty:
            log.warning("No metadata for well %s, using defaults", well_id)
            genotype = "unknown"
            treatment = "unknown"
            embryos_per_well = 1
        else:
            # Use first row (should be consistent per well)
            row = well_meta.iloc[0]
            genotype = str(row.get("genotype", "unknown"))
            treatment = str(row.get("treatment", "control"))
            embryos_per_well = int(row.get("embryos_per_well", 1))

        # Group frames by channel
        channels_dict = {}
        for frame_info in frames:
            channel_name = frame_info["channel_name"]
            if channel_name not in channels_dict:
                channels_dict[channel_name] = {
                    "channel_name": channel_name,
                    "raw_channel_name": frame_info["raw_channel_name"],
                    "frames": []
                }

            # Match metadata for this frame
            frame_meta = metadata_df[
                (metadata_df["well_id"] == well_id) &
                (metadata_df["time_int"] == frame_info["time_int"])
            ]

            if not frame_meta.empty:
                meta_row = frame_meta.iloc[0]
                experiment_time_s = float(meta_row.get("experiment_time_s", 0))
                image_width_px = int(meta_row.get("image_width_px", 0))
                image_height_px = int(meta_row.get("image_height_px", 0))
                micrometers_per_pixel = float(meta_row.get("micrometers_per_pixel", 0))
            else:
                experiment_time_s = 0
                image_width_px = 0
                image_height_px = 0
                micrometers_per_pixel = 0

            # Build frame dict
            frame_dict = {
                "frame_index": frame_info["time_int"],
                "time_int": frame_info["time_int"],
                "experiment_time_s": experiment_time_s,
                "image_id": f"{well_id}_{channel_name}_t{frame_info['time_int']:04d}",
                "file_path": frame_info["file_path"],
                "image_width_px": image_width_px,
                "image_height_px": image_height_px,
                "micrometers_per_pixel": micrometers_per_pixel,
            }

            channels_dict[channel_name]["frames"].append(frame_dict)
            all_channels.add(channel_name)
            total_frames += 1

        # Sort frames by time_int within each channel
        for channel_data in channels_dict.values():
            channel_data["frames"].sort(key=lambda f: f["time_int"])

        # Build well dict
        well_dict = {
            "well_id": well_id,
            "well_index": well_index,
            "embryos_per_well": embryos_per_well,
            "genotype": genotype,
            "treatment": treatment,
            "channels": list(channels_dict.values())
        }

        manifest["wells"].append(well_dict)

    # Update totals
    manifest["total_wells"] = len(manifest["wells"])
    manifest["total_channels"] = len(all_channels)
    manifest["total_frames"] = total_frames

    # Validate manifest
    _validate_manifest(manifest)

    # Write JSON
    output_dir = output_root / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_image_manifest.json"

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Wrote manifest: %s", output_path)
    log.info("  Wells: %d, Channels: %d, Frames: %d",
             manifest["total_wells"], manifest["total_channels"], manifest["total_frames"])

    return output_path


def _validate_manifest(manifest: dict):
    """
    Validate manifest against schema.

    Raises:
        ValueError: If validation fails
    """
    # Check experiment fields
    for field in REQUIRED_EXPERIMENT_FIELDS:
        if field not in manifest:
            raise ValueError(f"Missing experiment field: {field}")

    # Check wells
    if not manifest["wells"]:
        raise ValueError("Manifest contains no wells")

    for well in manifest["wells"]:
        # Check well fields
        for field in REQUIRED_WELL_FIELDS:
            if field not in well:
                raise ValueError(f"Missing well field in {well.get('well_id', 'unknown')}: {field}")

        # Check channels
        if not well["channels"]:
            raise ValueError(f"Well {well['well_id']} has no channels")

        # Validate BF channel exists
        channel_names = [ch["channel_name"] for ch in well["channels"]]
        if not any(ch in BRIGHTFIELD_CHANNELS for ch in channel_names):
            raise ValueError(f"Well {well['well_id']} missing brightfield channel")

        for channel in well["channels"]:
            # Check channel fields
            for field in REQUIRED_CHANNEL_FIELDS:
                if field not in channel:
                    raise ValueError(
                        f"Missing channel field in {well['well_id']}/{channel.get('channel_name', 'unknown')}: {field}"
                    )

            # Check frames
            if not channel["frames"]:
                raise ValueError(f"Channel {channel['channel_name']} in {well['well_id']} has no frames")

            for frame in channel["frames"]:
                # Check frame fields
                for field in REQUIRED_FRAME_FIELDS:
                    if field not in frame:
                        raise ValueError(
                            f"Missing frame field in {well['well_id']}/{channel['channel_name']}: {field}"
                        )

    log.info("Manifest validation passed")


if __name__ == "__main__":
    # Example usage (not executed in pipeline)
    from pathlib import Path

    data_root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
    metadata_root = data_root / "experiment_metadata"
    stitched_root = data_root / "built_image_data"
    output_root = data_root / "experiment_metadata"

    generate_image_manifest(
        metadata_root=metadata_root,
        stitched_root=stitched_root,
        output_root=output_root,
        exp_name="20240314",
        microscope_id="YX1",
    )
