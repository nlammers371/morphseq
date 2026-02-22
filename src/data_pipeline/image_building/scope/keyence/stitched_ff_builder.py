"""
Keyence Image Building: Stitched FF Image Organization

This module handles Keyence microscope data processing:
- Reads already focus-fused Keyence TIFFs
- Performs tile stitching using stitch2d
- Writes stitched images to built_image_data/{exp}/stitched_ff_images/{well}/{channel}/

MVP Requirements:
- Read Keyence TIFFs (already focus-fused)
- Stitch tiles into single images
- Organize into standardized directory structure
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Sequence
import numpy as np
from tqdm import tqdm
import skimage.io as skio
from stitch2d import StructuredMosaic

# Import shared utilities
from src.build.export_utils import trim_to_shape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
logging.getLogger("stitch2d").setLevel(logging.ERROR)

# Shift tolerance thresholds for QC
SHIFT_TOL = {
    2: {"vertical": 1, "horizontal": 1},
    3: {"vertical": 2, "horizontal": 2},
}


def align_with_qc(mosaic, n_tiles: int, orientation: str) -> bool:
    """
    Perform alignment with quality control checks.

    Returns:
        True if fallback needed, False if alignment succeeded
    """
    fallback_flag = False

    try:
        mosaic.align()
    except Exception as e:
        log.debug("align() raised %s – will fall back", e)
        return True

    coords = mosaic.params.get("coords", {})
    if len(coords) != n_tiles:
        log.debug("Only %d/%d tiles aligned – falling back", len(coords), n_tiles)
        return True

    # Shift QC: Δx for vertical stacks, Δy for horizontal
    arr = np.array([coords[i] for i in range(n_tiles)])
    axis = 1 if orientation == "vertical" else 0
    if np.abs(arr[:, axis]).max() > SHIFT_TOL[n_tiles][orientation]:
        log.debug("Shifts exceed tolerance – falling back")
        return True

    return fallback_flag


def stitch_keyence_tiles(
    ff_tile_dir: Path,
    output_path: Path,
    orientation: str,
    n_tiles: int,
    master_params_path: Path | None = None,
    target_shape: tuple[int, int] | None = None,
    overwrite: bool = False,
) -> bool:
    """
    Stitch Keyence FF tiles into a single image.

    Args:
        ff_tile_dir: Directory containing FF tile images (im_p*.jpg)
        output_path: Path to save stitched image
        orientation: 'vertical' or 'horizontal'
        n_tiles: Number of tiles (2 or 3)
        master_params_path: Path to master stitching parameters (fallback)
        target_shape: Target output shape (height, width)
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful, False otherwise
    """
    if output_path.exists() and not overwrite:
        return True

    try:
        # Initialize mosaic
        mosaic = StructuredMosaic(
            str(ff_tile_dir),
            dim=n_tiles,
            origin="upper left",
            direction=orientation,
            pattern="raster"
        )

        # Attempt alignment with QC
        fallback_flag = align_with_qc(mosaic, n_tiles=n_tiles, orientation=orientation)

        # Use master params if alignment failed
        if fallback_flag and master_params_path and master_params_path.exists():
            log.debug("Using master params for %s", ff_tile_dir.name)
            mosaic.load_params(str(master_params_path))

        # Reset tiles and stitch
        mosaic.reset_tiles()
        mosaic.smooth_seams()
        stitched = mosaic.stitch()

        # Transpose if horizontal orientation
        if orientation == "horizontal":
            stitched = stitched.T

        # Trim to target shape if specified
        if target_shape is not None:
            stitched = trim_to_shape(stitched, target_shape)

        # Invert (Keyence convention)
        maxv = np.iinfo(stitched.dtype).max
        stitched = maxv - stitched

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        skio.imsave(output_path, stitched, check_contrast=False)

        return True

    except Exception as e:
        log.error("Failed to stitch %s: %s", ff_tile_dir.name, e)
        return False


def build_master_params(
    sample_dirs: List[Path],
    orientation: str,
    n_tiles: int,
    outfile: Path,
):
    """
    Build master stitching parameters from sample directories.

    Samples multiple stitching attempts and computes median parameters
    for use as fallback when individual alignment fails.
    """
    align_array = []
    last_good_mosaic = None

    for fld in sample_dirs:
        try:
            mosaic = StructuredMosaic(
                str(fld), dim=n_tiles,
                origin="upper left", direction=orientation,
                pattern="raster"
            )
            mosaic.align()
            if len(mosaic.params["coords"]) == n_tiles:
                coords = mosaic.params["coords"]
                arr = np.array([coords[i] for i in range(n_tiles)])
                align_array.append(arr)
                last_good_mosaic = mosaic
        except Exception as e:
            log.debug("Sample alignment failed for %s: %s", fld, e)

    if not align_array or last_good_mosaic is None:
        log.warning("No good samples – master params not written")
        return

    # Median across all sampled alignments
    med_coords = np.nanmedian(np.stack(align_array), axis=0)
    coords_dict = {i: med_coords[i].tolist() for i in range(n_tiles)}

    # Use real params dict as template
    master_params = last_good_mosaic.params.copy()
    master_params["coords"] = coords_dict

    # Write JSON
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(master_params))
    log.info("Wrote master params to %s", outfile)


def compile_keyence_data(
    raw_data_root: Path,
    output_root: Path,
    exp_name: str,
    ff_tile_root: Path,  # Where FF tiles were saved (intermediate step)
    orientation: str,
    size_factor: float = 1.0,
    overwrite: bool = False,
    n_stitch_samples: int = 50,
):
    """
    Compile Keyence stitched FF images from pre-computed tiles.

    Args:
        raw_data_root: Path to raw_image_data/Keyence/
        output_root: Path to built_image_data/
        exp_name: Experiment name
        ff_tile_root: Path to FF tile directory (intermediate)
        orientation: Tile orientation ('vertical' or 'horizontal')
        size_factor: Scaling factor for target shape
        overwrite: Whether to overwrite existing files
        n_stitch_samples: Number of samples for master params

    Output structure:
        built_image_data/{exp_name}/stitched_ff_images/{well}/{channel}/{well}_{channel}_t{time_int:04d}.tif

    Note: This assumes FF tiles have already been generated in ff_tile_root.
          Full integration with FF generation will come in later pipeline stages.
    """
    log.info("Processing Keyence data: %s", exp_name)

    output_dir = output_root / exp_name / "stitched_ff_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find FF tile directories
    ff_folders = sorted(ff_tile_root.glob("ff_*"))
    if not ff_folders:
        log.warning("No FF tile folders found in %s", ff_tile_root)
        return

    n_tiles = len(list(ff_folders[0].glob("*.jpg")))
    log.info("Found %d FF folders, %d tiles per folder", len(ff_folders), n_tiles)

    # Build or load master params
    master_params_path = ff_tile_root / "master_params.json"
    if overwrite or not master_params_path.exists():
        sample_dirs = np.random.choice(
            ff_folders,
            min(n_stitch_samples, len(ff_folders)),
            replace=False
        )
        build_master_params(list(sample_dirs), orientation, n_tiles, master_params_path)

    # Determine target shape
    target_shapes = {
        2: np.array([800, 630]),
        3: np.array([1140, 630]) if orientation == "vertical" else np.array([1140, 480])
    }
    target = tuple((target_shapes[n_tiles] * size_factor).astype(int))

    # Process each folder
    processed = 0
    skipped = 0
    failed = 0

    for ff_folder in tqdm(ff_folders, desc="Stitching"):
        # Parse folder name: ff_{well}_t{time:04d}
        folder_name = ff_folder.name  # e.g., ff_A01_t0000
        parts = folder_name.replace("ff_", "").split("_t")
        if len(parts) != 2:
            log.warning("Unexpected folder name format: %s", folder_name)
            continue

        well_name = parts[0]
        try:
            time_int = int(parts[1])
        except ValueError:
            log.warning("Could not parse time from: %s", folder_name)
            continue

        # Determine channel (assume BF for MVP, extend later for multi-channel)
        channel_name = "BF"

        # Output path
        output_path = output_dir / well_name / channel_name / f"{well_name}_{channel_name}_t{time_int:04d}.tif"

        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        # Stitch
        success = stitch_keyence_tiles(
            ff_tile_dir=ff_folder,
            output_path=output_path,
            orientation=orientation,
            n_tiles=n_tiles,
            master_params_path=master_params_path,
            target_shape=target,
            overwrite=overwrite,
        )

        if success:
            processed += 1
        else:
            failed += 1

    log.info("Keyence stitching complete: processed=%d, skipped=%d, failed=%d",
             processed, skipped, failed)


if __name__ == "__main__":
    # Example usage (not executed in pipeline)
    from pathlib import Path

    data_root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
    raw_root = data_root / "raw_image_data" / "Keyence"
    built_root = data_root / "built_image_data"
    ff_tile_root = built_root / "Keyence" / "FF_images" / "20240915_keyence"

    compile_keyence_data(
        raw_data_root=raw_root,
        output_root=built_root,
        exp_name="20240915_keyence",
        ff_tile_root=ff_tile_root,
        orientation="vertical",
        size_factor=1.0,
        overwrite=False,
    )
