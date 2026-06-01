"""
Keyence-specific series to well mapping.

Maps Keyence microscope series numbers to well positions based on file structure.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
import logging

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.shared.identifiers import build_well_id

log = logging.getLogger(__name__)

# Series mapping schema
REQUIRED_COLUMNS_SERIES_MAPPING = [
    'experiment_id',
    'well_index',
    'well_id',
    'series_number',
    'mapping_method',
]


def _discover_keyence_wells(raw_data_dir: Path, experiment_id: str) -> list:
    """
    Discover wells in Keyence experiment directory.

    Args:
        raw_data_dir: Root directory containing raw Keyence data
        experiment_id: Experiment identifier

    Returns:
        List of (well_index, well_path) tuples sorted by well
    """
    exp_dir = raw_data_dir / experiment_id

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    wells = []

    # Check for XY pattern (e.g., XY01, XY16)
    xy_dirs = sorted(exp_dir.glob("XY*"))
    if xy_dirs:
        for well_dir in xy_dirs:
            well_name = well_dir.name  # e.g., "XY01a"
            suffix = well_name[2:]
            if suffix.isdigit():
                xy_idx = int(suffix)
                row = (xy_idx - 1) // 12
                col = (xy_idx - 1) % 12 + 1
                well_index = f"{chr(65 + row)}{col:02d}"
                wells.append((well_index, well_dir))
                continue

            # Legacy format with trailing row letter (e.g. XY01a)
            well_num = well_name[2:4]
            well_letter = well_name[-1].upper()
            if well_num.isdigit() and well_letter.isalpha():
                well_index = f"{well_letter}{well_num}"
                wells.append((well_index, well_dir))

    # Check for W0 pattern (W001 → A01)
    w0_dirs = sorted(exp_dir.glob("W0*"))
    if w0_dirs and not xy_dirs:
        for well_dir in w0_dirs:
            well_name = well_dir.name  # e.g., "W001"
            well_num = int(well_name[1:])
            # Convert to row/col (1-indexed, 12 cols per row)
            row = (well_num - 1) // 12
            col = (well_num - 1) % 12 + 1
            well_index = f"{chr(65 + row)}{col:02d}"
            wells.append((well_index, well_dir))

    if not wells:
        raise ValueError(f"No Keyence well directories found in {exp_dir}")

    log.info(f"Discovered {len(wells)} Keyence wells in {exp_dir}")
    return sorted(wells, key=lambda x: x[0])


def _count_positions_per_well(well_path: Path) -> int:
    """
    Count number of positions/series in a Keyence well directory.

    Keyence can have:
    - Single position (files directly in well dir)
    - Multiple positions (P* subdirectories)

    Args:
        well_path: Path to well directory

    Returns:
        Number of positions/series in well
    """
    # Check for P* subdirectories
    pos_dirs = sorted(well_path.glob("P*"))
    if pos_dirs:
        return len(pos_dirs)

    # Check for direct image files (single position)
    image_files = list(well_path.glob("*CH*.tif"))
    if image_files:
        return 1

    # No images found
    log.warning(f"No images found in {well_path}")
    return 0


def map_series_to_wells_keyence(
    raw_data_dir: Path,
    scope_metadata_csv: Path,
    output_mapping_csv: Path,
    output_provenance_json: Path,
    experiment_id: str,
) -> pd.DataFrame:
    """
    Map Keyence series to wells (plate-free).

    For Keyence microscopes, the mapping strategy is:
    1. Discover well directories (XY## or W0##)
    2. Count positions per well (P* subdirs or single position)
    3. Assign sequential series numbers
    4. Cross-reference with scope metadata to flag missing wells

    Args:
        raw_data_dir: Root directory containing raw Keyence data
        scope_metadata_csv: Path to validated scope_series_metadata_raw.csv
        output_mapping_csv: Path to write series_well_mapping.csv
        output_provenance_json: Path to write mapping_provenance.json
        experiment_id: Experiment identifier (used for composing well_id)

    Returns:
        DataFrame with series-to-well mapping

    Raises:
        FileNotFoundError: If required files not found
        ValueError: If mapping fails validation
    """
    log.info(f"Mapping Keyence series to wells for {experiment_id}")

    # Load scope metadata to cross-reference
    scope_df = pd.read_csv(scope_metadata_csv)

    # Discover wells from raw data directory
    wells = _discover_keyence_wells(raw_data_dir, experiment_id)

    # Build mapping
    rows = []
    series_number = 0
    warnings = []

    for well_index, well_path in wells:
        # Count positions in this well
        n_positions = _count_positions_per_well(well_path)

        if n_positions == 0:
            warnings.append(f"Well {well_index}: No images found")
            continue

        # For each position, create a series mapping
        for pos_idx in range(n_positions):
            series_number += 1

            scope_has_well = well_index in scope_df['well_index'].values

            if not scope_has_well:
                warnings.append(f"Well {well_index}: Not found in scope metadata")

            row = {
                'experiment_id': experiment_id,
                'well_index': well_index,
                'well_id': build_well_id(well_index),
                'series_number': series_number,
                'position_index': pos_idx,
                'mapping_method': 'keyence_directory_structure',
                'n_positions_in_well': n_positions,
                'source_directory': str(well_path),
            }

            rows.append(row)

    if not rows:
        raise ValueError(f"No valid series-to-well mappings found for {experiment_id}")

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Validate against schema
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_SERIES_MAPPING,
        stage_name="Keyence series-to-well mapping"
    )

    # Write output CSV
    output_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_mapping_csv, index=False)
    log.info(f"Wrote Keyence series mapping to {output_mapping_csv}")

    # Write provenance JSON
    provenance = {
        'experiment_id': experiment_id,
        'microscope': 'Keyence',
        'mapping_method': 'keyence_directory_structure',
        'n_series': int(len(df)),
        'n_wells': int(df['well_index'].nunique()),
        'mapping_summary': {
            'total_series': int(len(df)),
            'total_wells': int(df['well_index'].nunique()),
            'wells_with_multiple_positions': int((df['n_positions_in_well'] > 1).sum()),
        },
        'warnings': warnings if warnings else None,
        'source_files': {
            'scope_metadata': str(scope_metadata_csv),
        },
    }

    output_provenance_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_provenance_json, 'w') as f:
        json.dump(provenance, f, indent=2)
    log.info(f"Wrote mapping provenance to {output_provenance_json}")

    if warnings:
        log.warning(f"Mapping completed with {len(warnings)} warnings")
        for warn in warnings[:5]:  # Show first 5 warnings
            log.warning(f"  {warn}")
        if len(warnings) > 5:
            log.warning(f"  ... and {len(warnings) - 5} more warnings")

    return df


def load_series_well_mapping(mapping_csv: Path) -> pd.DataFrame:
    """
    Load and validate series-to-well mapping.

    Args:
        mapping_csv: Path to series_well_mapping.csv

    Returns:
        Validated DataFrame

    Raises:
        FileNotFoundError: If file not found
        ValueError: If validation fails
    """
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Series mapping not found: {mapping_csv}")

    df = pd.read_csv(mapping_csv)

    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_SERIES_MAPPING,
        stage_name="Series-to-well mapping"
    )

    return df



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-keyence-experiment-dir", type=Path, required=True)
    p.add_argument("--scope-metadata-csv", type=Path, required=True)
    p.add_argument("--output-mapping-csv", type=Path, required=True)
    p.add_argument("--output-provenance-json", type=Path, required=True)
    p.add_argument("--experiment-id", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    map_series_to_wells_keyence(
        raw_data_dir=args.raw_keyence_experiment_dir,
        scope_metadata_csv=args.scope_metadata_csv,
        output_mapping_csv=args.output_mapping_csv,
        output_provenance_json=args.output_provenance_json,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
