"""
YX1 microscope metadata extraction.

Extracts scope metadata from YX1 ND2 files and produces validated scope_metadata.csv.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import nd2

from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA
from data_pipeline.schemas.channel_normalization import CHANNEL_NORMALIZATION_MAP
from data_pipeline.io.validators import validate_dataframe_schema

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _find_nd2_file(raw_data_dir: Path) -> Path:
    """Find the ND2 file in the raw data directory."""
    nd2_files = list(raw_data_dir.glob("*.nd2"))
    if not nd2_files:
        raise FileNotFoundError(f"No ND2 files found in {raw_data_dir}")
    if len(nd2_files) > 1:
        raise RuntimeError(f"Multiple ND2 files found in {raw_data_dir}: {nd2_files}")
    return nd2_files[0]


def _extract_timestamps(nd: nd2.ND2File, n_t: int, n_w: int, n_z: int) -> np.ndarray:
    """
    Extract timestamps from ND2 file with gap imputation.

    Args:
        nd: Open ND2File object
        n_t: Number of timepoints
        n_w: Number of wells
        n_z: Number of Z slices

    Returns:
        Array of timestamps in seconds (length n_t)
    """
    # Extract timestamps from first well
    times = np.full((n_t,), np.nan, dtype=float)

    for t in range(n_t):
        seq = t * n_w * n_z  # First well, first z-slice
        try:
            times[t] = nd.frame_metadata(seq).channels[0].time.relativeTimeMs / 1000.0
        except Exception:
            continue  # Leave as NaN

    valid_count = (~np.isnan(times)).sum()
    log.info(f"Extracted {valid_count}/{n_t} valid timestamps from ND2")

    # Calculate cycle time from valid data
    s = pd.Series(times)
    original_valid = s.dropna()

    if len(original_valid) >= 2:
        original_diffs = original_valid.diff().dropna()
        cycle_time = original_diffs.median()
        log.info(f"Calculated cycle time: {cycle_time:.2f}s")
    else:
        cycle_time = 1800.0  # 30 minutes default
        log.info(f"Using default cycle time: {cycle_time:.2f}s")

    # Impute missing values
    if s.isna().any():
        missing_count = s.isna().sum()
        log.info(f"Imputing {missing_count} missing timestamps...")

        if len(original_valid) > 0:
            first_valid_idx = s.first_valid_index()
            last_valid_idx = s.last_valid_index()

            # Fill backwards from first valid
            if first_valid_idx > 0:
                first_time = s.iloc[first_valid_idx]
                for i in range(first_valid_idx - 1, -1, -1):
                    s.iloc[i] = first_time - (first_valid_idx - i) * cycle_time

            # Fill forwards from last valid
            if last_valid_idx < len(s) - 1:
                last_time = s.iloc[last_valid_idx]
                for i in range(last_valid_idx + 1, len(s)):
                    s.iloc[i] = last_time + (i - last_valid_idx) * cycle_time

            # Fill middle gaps
            for i in range(len(s)):
                if pd.isna(s.iloc[i]):
                    s.iloc[i] = s.iloc[0] + i * cycle_time

    # Last resort
    if s.isna().all():
        log.warning("No valid timestamps - using default intervals")
        s = pd.Series(np.arange(n_t, dtype=float) * 1800.0)

    # Ensure monotonic
    s = s.cummax()

    return s.to_numpy()


def _normalize_channel_name(raw_name: str) -> str:
    """
    Normalize YX1 channel name to standard name.

    Args:
        raw_name: Raw channel name from ND2

    Returns:
        Normalized channel name (e.g., 'BF', 'GFP')
    """
    # Try direct mapping first
    if raw_name in CHANNEL_NORMALIZATION_MAP:
        return CHANNEL_NORMALIZATION_MAP[raw_name]

    # Try case-insensitive match for common patterns
    raw_lower = raw_name.lower()

    # Check for BF variations
    if any(x in raw_lower for x in ['dia', 'empty', 'brightfield', 'bf']):
        return 'BF'

    # Check for fluorescence channels
    if 'gfp' in raw_lower:
        return 'GFP'
    if 'rfp' in raw_lower or 'mcherry' in raw_lower:
        return 'RFP'

    # Default: return as-is and warn
    log.warning(f"Unknown channel name '{raw_name}' - using as-is")
    return raw_name


def extract_yx1_scope_metadata(
    raw_data_dir: Path,
    output_csv: Path,
    experiment_id: str
) -> pd.DataFrame:
    """
    Extract YX1 scope metadata from ND2 file.

    Args:
        raw_data_dir: Directory containing ND2 file
        output_csv: Output path for scope_metadata.csv
        experiment_id: Experiment identifier

    Returns:
        DataFrame with validated scope metadata
    """
    log.info(f"Extracting YX1 scope metadata for {experiment_id}")

    # Find and open ND2 file
    nd2_path = _find_nd2_file(raw_data_dir)
    log.info(f"Reading ND2 file: {nd2_path}")

    with nd2.ND2File(nd2_path) as nd:
        # Get dimensions
        shape = nd.shape  # (T, W, Z, C, Y, X)
        n_t, n_w, n_z = shape[:3]
        log.info(f"ND2 shape: T={n_t}, W={n_w}, Z={n_z}")

        # Get spatial calibration
        voxel_size = nd.voxel_size()
        micrometers_per_pixel = voxel_size[0]  # X dimension
        image_height_px = shape[4]  # Y
        image_width_px = shape[5]   # X

        # Get channel names
        channel_names = [c.channel.name for c in nd.frame_metadata(0).channels]
        log.info(f"Raw channel names: {channel_names}")

        # Get objective info
        try:
            objective = nd.frame_metadata(0).channels[0].microscope.objectiveName
        except:
            objective = "Unknown"

        # Extract timestamps
        timestamps = _extract_timestamps(nd, n_t, n_w, n_z)

        # Calculate frame interval
        if len(timestamps) >= 2:
            frame_interval_s = float(np.median(np.diff(timestamps)))
        else:
            frame_interval_s = 1800.0  # Default 30 min

        log.info(f"Frame interval: {frame_interval_s:.2f}s")

        # Build metadata rows (one per well, timepoint, channel)
        rows = []

        for w_idx in range(n_w):
            well_index = f"{w_idx:02d}"  # Will be mapped to well ID later

            for t_idx in range(n_t):
                time_s = timestamps[t_idx]

                for c_idx, raw_channel in enumerate(channel_names):
                    # Normalize channel name
                    channel = _normalize_channel_name(raw_channel)

                    # Build IDs (temporary - will be refined by series mapper)
                    well_id = f"{experiment_id}_{well_index}"
                    image_id = f"{well_id}_{channel}_t{t_idx:04d}"

                    row = {
                        'experiment_id': experiment_id,
                        'well_id': well_id,
                        'well_index': well_index,
                        'image_id': image_id,
                        'time_int': t_idx,
                        'micrometers_per_pixel': micrometers_per_pixel,
                        'image_width_px': image_width_px,
                        'image_height_px': image_height_px,
                        'objective_magnification': objective,
                        'frame_interval_s': frame_interval_s,
                        'absolute_start_time': timestamps[0],
                        'experiment_time_s': time_s,
                        'microscope_id': 'YX1',
                        'channel': channel,
                        'z_position': 0,  # Z-stacked, so single plane
                        'frame_index': t_idx,
                    }
                    rows.append(row)

    # Build DataFrame
    df = pd.DataFrame(rows)

    log.info(f"Created metadata with {len(df)} rows")
    log.info(f"Wells: {df['well_index'].nunique()}, Timepoints: {df['time_int'].nunique()}, Channels: {df['channel'].nunique()}")

    # Validate schema
    validate_dataframe_schema(df, REQUIRED_COLUMNS_SCOPE_METADATA, "YX1 scope metadata")

    # Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info(f"Wrote scope metadata to {output_csv}")

    return df
