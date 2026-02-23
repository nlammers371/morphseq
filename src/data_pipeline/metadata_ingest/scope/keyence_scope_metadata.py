"""
Keyence microscope metadata extraction.

Extracts scope metadata from Keyence BZ-X TIFF files and validates against schema.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import logging

from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA
from data_pipeline.schemas.channel_normalization import CHANNEL_NORMALIZATION_MAP
from data_pipeline.io.validators import validate_dataframe_schema

log = logging.getLogger(__name__)


def _scrape_keyence_metadata(tiff_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from Keyence TIFF file.

    Keyence BZ-X microscopes embed XML metadata in TIFF files between <Data> tags.

    Args:
        tiff_path: Path to Keyence TIFF file

    Returns:
        Dictionary with metadata fields
    """
    def _findnth(haystack, needle, n):
        """Find the nth occurrence of needle in haystack."""
        parts = haystack.split(needle, n + 1)
        if len(parts) <= n + 1:
            return -1
        return len(haystack) - len(parts[-1]) - len(needle)

    with open(tiff_path, 'rb') as f:
        fulldata = f.read()

    # Extract XML metadata between <Data> tags
    metadata = fulldata.partition(b'<Data>')[2].partition(b'</Data>')[0].decode()

    meta_dict = {}
    keyword_list = ['ShootingDateTime', 'LensName', 'Observation Type', 'Width', 'Height', 'Width', 'Height']
    outname_list = ['Time (s)', 'Objective', 'Channel', 'Width (px)', 'Height (px)', 'Width (um)', 'Height (um)']

    for k in range(len(keyword_list)):
        param_string = keyword_list[k]
        name = outname_list[k]

        if (param_string == 'Width') or (param_string == 'Height'):
            if 'um' in name:
                ind1 = _findnth(metadata, param_string + ' Type', 2)
                ind2 = _findnth(metadata, '/' + param_string, 2)
            else:
                ind1 = _findnth(metadata, param_string + ' Type', 1)
                ind2 = _findnth(metadata, '/' + param_string, 1)
        else:
            ind1 = metadata.find(param_string)
            ind2 = metadata.find('/' + param_string)

        long_string = metadata[ind1:ind2]
        subind1 = long_string.find(">")
        subind2 = long_string.find("<")
        param_val = long_string[subind1+1:subind2]

        sysind = long_string.find("System.")
        dtype = long_string[sysind+7:subind1-1]
        if 'Int' in dtype:
            param_val = int(param_val)

        if param_string == "ShootingDateTime":
            # Convert from 100 nanoseconds to seconds
            param_val = float(param_val) / 10 / 1000 / 1000
        elif "um" in name:
            # Convert from nanometers to micrometers
            param_val = float(param_val) / 1000

        meta_dict[name] = param_val

    return meta_dict


def _normalize_channel_name(raw_channel: str) -> str:
    """
    Normalize Keyence channel name to standard name.

    Args:
        raw_channel: Raw channel name from microscope

    Returns:
        Normalized channel name (e.g., "BF", "GFP", "RFP")
    """
    # Try direct lookup first
    if raw_channel in CHANNEL_NORMALIZATION_MAP:
        return CHANNEL_NORMALIZATION_MAP[raw_channel]

    # Try case-insensitive match
    raw_lower = raw_channel.lower()
    for key, value in CHANNEL_NORMALIZATION_MAP.items():
        if key.lower() == raw_lower:
            return value

    # Common Keyence variations
    if 'bright' in raw_lower or 'bf' in raw_lower or 'phase' in raw_lower:
        return 'BF'
    elif 'gfp' in raw_lower or 'green' in raw_lower:
        return 'GFP'
    elif 'rfp' in raw_lower or 'red' in raw_lower or 'cherry' in raw_lower:
        return 'RFP'

    # Default: return as-is with warning
    log.warning(f"Unknown channel name '{raw_channel}', using as-is")
    return raw_channel


def _discover_keyence_files(raw_data_dir: Path, experiment_id: str) -> List[Path]:
    """
    Discover Keyence TIFF files in raw data directory.

    Keyence file structure can vary:
    - {exp}/XY##/{files}  (multi-well format)
    - {exp}/W0##/{files}  (cytometer format)
    - {exp}/{files}       (flat format)

    Args:
        raw_data_dir: Root directory containing raw Keyence data
        experiment_id: Experiment identifier

    Returns:
        List of TIFF file paths
    """
    exp_dir = raw_data_dir / experiment_id

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Try different Keyence file patterns
    tiff_files = []

    # Pattern 1: Files with CH (channel) indicator
    tiff_files = list(exp_dir.rglob("*CH*.tif"))

    if not tiff_files:
        # Pattern 2: Any TIFF files
        tiff_files = list(exp_dir.rglob("*.tif"))

    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {exp_dir}")

    log.info(f"Discovered {len(tiff_files)} Keyence TIFF files in {exp_dir}")
    return sorted(tiff_files)


def _extract_well_from_path(file_path: Path) -> str:
    """
    Extract well identifier from Keyence file path.

    Common patterns:
    - XY##a/... → "A##" format
    - W0##/... → well index
    - Filename contains well info

    Args:
        file_path: Path to Keyence TIFF file

    Returns:
        Well identifier (e.g., "A01", "B12")
    """
    path_str = str(file_path)

    # Check for XY pattern in path (e.g., XY01a → A01)
    for part in file_path.parts:
        if part.startswith('XY'):
            # XY01a format
            well_num = part[2:4]  # Get "01"
            well_letter = part[-1].upper()  # Get "a" → "A"
            # Convert letter to row (a=A, b=B, etc.)
            if well_letter.isalpha():
                return f"{well_letter}{well_num}"

    # Check for W0 pattern (W001 → A01)
    for part in file_path.parts:
        if part.startswith('W0'):
            well_idx = int(part[1:])
            # Convert to row/col (1-indexed, 12 cols per row)
            row = (well_idx - 1) // 12
            col = (well_idx - 1) % 12 + 1
            return f"{chr(65 + row)}{col:02d}"

    # Fallback: extract from filename
    filename = file_path.name
    # Look for patterns like "A01", "B12", etc.
    import re
    match = re.search(r'[A-H](0[1-9]|1[0-2])', filename)
    if match:
        return match.group(0)

    log.warning(f"Could not extract well from path: {file_path}")
    return "unknown"


def extract_keyence_scope_metadata(
    raw_data_dir: Path,
    experiment_id: str,
    output_csv: Path
) -> pd.DataFrame:
    """
    Extract Keyence scope metadata from raw TIFF files.

    Reads Keyence BZ-X microscope TIFF files, extracts embedded metadata,
    normalizes channel names, and validates against schema.

    Args:
        raw_data_dir: Root directory containing raw Keyence data
        experiment_id: Experiment identifier
        output_csv: Path to write validated scope_metadata.csv

    Returns:
        Validated DataFrame with scope metadata

    Raises:
        FileNotFoundError: If data directory or files not found
        ValueError: If validation fails
    """
    log.info(f"Extracting Keyence scope metadata for {experiment_id}")

    # Discover TIFF files
    tiff_files = _discover_keyence_files(raw_data_dir, experiment_id)

    # Extract metadata from each file
    rows = []
    for tiff_path in tiff_files:
        try:
            meta = _scrape_keyence_metadata(tiff_path)

            # Extract well from path
            well_index = _extract_well_from_path(tiff_path)

            # Normalize channel name
            raw_channel = meta.get('Channel', 'unknown')
            normalized_channel = _normalize_channel_name(raw_channel)

            # Compute micrometers per pixel
            width_um = meta.get('Width (um)', 0)
            width_px = meta.get('Width (px)', 1)
            micrometers_per_pixel = width_um / width_px if width_px > 0 else 0

            # Extract timepoint index from filename or path
            # Common patterns: T0001, t0001, _t01, etc.
            import re
            time_match = re.search(r'[Tt](\d+)', tiff_path.name)
            time_int = int(time_match.group(1)) if time_match else 0

            # Build row
            row = {
                'experiment_id': experiment_id,
                'well_index': well_index,
                'well_id': f"{experiment_id}_{well_index}",
                'time_int': time_int,
                'frame_index': time_int,  # For Keyence, time_int == frame_index
                'image_id': f"{experiment_id}_{well_index}_{normalized_channel}_t{time_int:04d}",

                # Spatial calibration
                'micrometers_per_pixel': micrometers_per_pixel,
                'image_width_px': meta.get('Width (px)', 0),
                'image_height_px': meta.get('Height (px)', 0),
                'objective_magnification': meta.get('Objective', 'unknown'),

                # Temporal calibration
                'absolute_start_time': meta.get('Time (s)', 0),
                'experiment_time_s': meta.get('Time (s)', 0),
                'frame_interval_s': 0,  # Will compute after sorting

                # Acquisition metadata
                'microscope_id': 'Keyence',
                'channel': normalized_channel,
                'z_position': 0,  # Keyence FF images are single Z

                # Provenance
                'raw_channel_name': raw_channel,
                'source_file': str(tiff_path),
            }

            rows.append(row)

        except Exception as e:
            log.warning(f"Failed to extract metadata from {tiff_path}: {e}")
            continue

    if not rows:
        raise ValueError(f"No valid metadata extracted for {experiment_id}")

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Sort by well and time
    df = df.sort_values(['well_index', 'time_int']).reset_index(drop=True)

    # Compute frame_interval_s
    # Group by well and compute time differences
    def compute_intervals(group):
        if len(group) > 1:
            # Compute median interval for this well
            times = group['experiment_time_s'].values
            intervals = np.diff(times)
            median_interval = np.median(intervals) if len(intervals) > 0 else 0
            group['frame_interval_s'] = median_interval
        else:
            group['frame_interval_s'] = 0
        return group

    df = df.groupby('well_index', group_keys=False).apply(compute_intervals)

    # Adjust experiment_time_s to be relative to experiment start
    min_time = df['experiment_time_s'].min()
    df['experiment_time_s'] = df['experiment_time_s'] - min_time

    # Validate against schema
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_SCOPE_METADATA,
        stage_name="Keyence scope metadata extraction"
    )

    # Write output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info(f"Wrote Keyence scope metadata to {output_csv}")

    return df
