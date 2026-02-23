"""
YX1 Image Building: Focus Stacking and Stitched FF Image Generation

This module handles YX1 microscope data processing:
- Reads ND2 files
- Focus stacking using LoG (Laplacian of Gaussian) method
- Writes stitched FF images to built_image_data/{exp}/stitched_ff_images/{well}/{channel}/

MVP Requirements:
- Read ND2 file
- Focus stack (LoG method)
- Write stitched TIFFs
- GPU support with proper device parameter
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from tqdm import tqdm
import nd2
import skimage
import skimage.io as skio

# Import shared utilities from existing codebase
from src.build.export_utils import LoG_focus_stacker, im_rescale

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def _read_nd2(path: Path) -> nd2.ND2File:
    """Read ND2 file from directory."""
    nd2_files = list(path.glob("*.nd2"))
    if not nd2_files:
        raise FileNotFoundError(f"No nd2 in {path}")
    if len(nd2_files) > 1:
        raise RuntimeError(f"Multiple nd2 files in {path}")
    return nd2.ND2File(nd2_files[0])


def _get_stack(
    dask_arr, t: int, w: int, n_z_keep: int | None = None
) -> np.ndarray:
    """Return Z×Y×X BF stack (no channels)."""
    nz = dask_arr.shape[2]
    buf = max((nz - n_z_keep) // 2, 0) if n_z_keep else 0
    return (
        dask_arr[t, w, buf : nz - buf, :, :].compute()
        if buf or n_z_keep
        else dask_arr[t, w, :, :, :].compute()
    )


def _focus_stack(
    stack_zyx: np.ndarray,
    device: str,
    filter_size: int = 3
) -> np.ndarray:
    """Apply LoG focus stacking to Z-stack."""
    # Normalize and convert to tensor
    norm, _, _ = im_rescale(stack_zyx)
    norm = norm.astype(np.float32)
    tensor = torch.from_numpy(norm).to(device)

    # Apply focus stacking
    ff_t, _ = LoG_focus_stacker(tensor, filter_size, device)
    arr = ff_t.cpu().numpy()
    arr_clipped = np.clip(arr, 0, 65535)
    ff_i = arr_clipped.astype(np.uint16)

    # Convert to 8 bit
    ff_8 = skimage.util.img_as_ubyte(ff_i)

    return ff_8


def _write_stitched_ff(
    output_dir: Path,
    well_name: str,
    channel_name: str,
    time_int: int,
    image: np.ndarray,
    overwrite: bool = False
):
    """
    Write stitched FF image to standardized location.

    Output structure:
    built_image_data/{exp}/stitched_ff_images/{well}/{channel}/{well}_{channel}_t{time_int:04d}.tif
    """
    well_dir = output_dir / well_name / channel_name
    well_dir.mkdir(parents=True, exist_ok=True)

    output_path = well_dir / f"{well_name}_{channel_name}_t{time_int:04d}.tif"

    if output_path.exists() and not overwrite:
        return

    skio.imsave(output_path, image, check_contrast=False)


def compile_yx1_data(
    raw_data_root: Path,
    output_root: Path,
    exp_name: str,
    well_series_mapping: dict[str, int],  # well_name -> series_number (1-based)
    overwrite: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",  # GPU BUG FIX: NOT commented out
    n_workers: int = 1,
    z_buffer: bool = False,
):
    """
    Compile YX1 ND2 data into stitched FF images.

    Args:
        raw_data_root: Path to raw_image_data/YX1/
        output_root: Path to built_image_data/
        exp_name: Experiment name
        well_series_mapping: Dict mapping well names (e.g., 'A01') to ND2 series numbers (1-based)
        overwrite: Whether to overwrite existing files
        device: PyTorch device for focus stacking ('cuda' or 'cpu')
        n_workers: Number of workers (not used in MVP, kept for compatibility)
        z_buffer: Whether to trim Z-stack (specific to exp 20231206)

    Output structure:
        built_image_data/{exp_name}/stitched_ff_images/{well}/{channel}/{well}_{channel}_t{time_int:04d}.tif
    """

    exp_path = raw_data_root / exp_name
    output_dir = output_root / exp_name / "stitched_ff_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Processing YX1 data: %s (device=%s)", exp_name, device)

    if device == "cpu":
        log.warning("Using CPU. This may be slow. GPU recommended.")

    # Read ND2 file
    nd = _read_nd2(exp_path)
    shape_twzcxy = nd.shape  # T,W,Z,C,Y,X
    n_t, n_w, n_z = shape_twzcxy[:3]
    log.info("ND2 shape: n_t=%d, n_w=%d, n_z=%d", n_t, n_w, n_z)

    dask_arr = nd.to_dask()  # (T,W,Z,C,Y,X)
    channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]

    # Determine BF channel index
    bf_idx = _determine_bf_channel(channel_names)
    log.info("Using BF channel index: %d (%s)", bf_idx, channel_names[bf_idx] if bf_idx < len(channel_names) else "unknown")

    # Select BF channel if multi-channel
    if len(shape_twzcxy) == 6:
        dask_arr = dask_arr[:, :, :, bf_idx, :, :]

    # Build lookup of ND2 well index -> well name
    well_name_lookup = {int(series)-1: name for name, series in well_series_mapping.items()}

    log.info("Processing %d wells, %d timepoints", len(well_name_lookup), n_t)

    # Z-stack buffer for specific experiment
    n_z_keep = 12 if z_buffer else None

    # Process each well and timepoint
    total_frames = len(well_name_lookup) * n_t
    processed = 0
    skipped = 0

    for nd2_idx, well_name in sorted(well_name_lookup.items()):
        for t in range(n_t):
            # Check if already exists
            output_path = output_dir / well_name / "BF" / f"{well_name}_BF_t{t:04d}.tif"
            if output_path.exists() and not overwrite:
                skipped += 1
                continue

            try:
                # Get Z-stack for this well and timepoint
                stack = _get_stack(dask_arr, t, nd2_idx, n_z_keep=n_z_keep)

                # Apply focus stacking
                ff = _focus_stack(stack, device, filter_size=3)

                # Write output
                _write_stitched_ff(output_dir, well_name, "BF", t, ff, overwrite)

                processed += 1

                if processed % 50 == 0:
                    log.info("Processed %d/%d frames", processed, total_frames - skipped)

            except Exception as e:
                log.error("Failed processing well=%s, t=%d: %s", well_name, t, e)
                continue

    nd.close()

    log.info("YX1 processing complete: processed=%d, skipped=%d, total=%d",
             processed, skipped, total_frames)


def _determine_bf_channel(channel_names: list[str]) -> int:
    """
    Determine BF channel index from channel names.

    Handles various YX1 naming conventions:
    - 'BF' (standard)
    - 'EYES - Dia' (common alternative)
    - 'Empty' (mislabeled)
    - Single channel (assume BF)
    - Environment variable override: YX1_BF_CHANNEL_INDEX
    """
    # Check environment variable override
    import os
    env_bf = os.environ.get("YX1_BF_CHANNEL_INDEX")
    if env_bf is not None:
        try:
            return int(env_bf)
        except Exception:
            raise ValueError(f"Invalid YX1_BF_CHANNEL_INDEX env var: {env_bf}")

    # Try exact 'BF' (case-insensitive)
    lower = [str(n).lower() for n in channel_names]
    if "bf" in lower:
        return lower.index("bf")

    # Try known single-channel labels
    try_labels = ["eyes - dia", "empty"]
    for i, name in enumerate(lower):
        if name in try_labels:
            if name != "bf":
                log.warning("Using non-standard BF channel: '%s'", channel_names[i])
            return i

    # Fallback: if only one channel, use it
    if len(channel_names) == 1:
        log.warning("Single channel detected, assuming BF: '%s'", channel_names[0])
        return 0

    # Give up
    raise ValueError(
        f"Could not locate BF channel. Available: {channel_names}. "
        f"Set YX1_BF_CHANNEL_INDEX to override."
    )


if __name__ == "__main__":
    # Example usage (not executed in pipeline)
    from pathlib import Path

    data_root = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
    raw_root = data_root / "raw_image_data" / "YX1"
    built_root = data_root / "built_image_data"

    # Example well mapping (normally comes from series_well_mapper)
    example_mapping = {
        "A01": 1,
        "A02": 2,
        "B01": 3,
        "B02": 4,
    }

    compile_yx1_data(
        raw_data_root=raw_root,
        output_root=built_root,
        exp_name="20240314",
        well_series_mapping=example_mapping,
        overwrite=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
