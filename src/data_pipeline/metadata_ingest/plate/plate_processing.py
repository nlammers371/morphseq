"""
Process and normalize plate layout metadata from Excel/CSV files.

This module reads raw plate layout files, normalizes column names to match
the schema, and validates the output.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
from data_pipeline.io.validators import validate_dataframe_schema


def process_plate_layout(
    input_file: Path,
    experiment_id: str,
    output_csv: Path
) -> pd.DataFrame:
    """
    Normalize and validate plate metadata from Excel or CSV file.

    Supports two formats:
    1. Multi-sheet Excel format (standard MorphSeq format with 8x12 plate sheets)
    2. Single CSV/sheet format (flattened table)

    Args:
        input_file: Path to input Excel (.xlsx) or CSV file
        experiment_id: Experiment identifier to add to each row
        output_csv: Path where validated CSV will be written

    Returns:
        Validated DataFrame with normalized column names

    Raises:
        ValueError: If required columns are missing or contain null values
        FileNotFoundError: If input file doesn't exist
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Parse file based on format
    if input_file.suffix.lower() in ['.xlsx', '.xls']:
        # Try multi-sheet format first
        df, series_map = _parse_plate_metadata_excel(input_file, experiment_id)

        # If series_number_map exists, save it alongside plate metadata
        if series_map is not None:
            series_map_path = output_csv.parent / "series_number_map.csv"
            series_map.to_csv(series_map_path, index=False)
            print(f"✅ Saved series_number_map: {series_map_path}")

    elif input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
        df = _normalize_column_names(df)
        if 'experiment_id' not in df.columns:
            df['experiment_id'] = experiment_id
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}. Use .xlsx, .xls, or .csv")

    # Generate well_id (format: experiment_id_well_index)
    if 'well_id' not in df.columns:
        df['well_id'] = df['experiment_id'] + '_' + df['well_index']

    # Validate against schema
    validate_dataframe_schema(df, REQUIRED_COLUMNS_PLATE_METADATA, "plate_metadata")

    # Write validated CSV
    df.to_csv(output_csv, index=False)

    return df


def _parse_plate_metadata_excel(xlsx_path: Path, exp_name: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Flatten all 8×12 plate layout sheets into a long DataFrame.

    Dynamically reads all sheets in the Excel file that have the standard
    8×12 plate format (rows A-H, columns 1-12). Also extracts series_number_map
    if present (critical for YX1 data).

    Args:
        xlsx_path: Path to multi-sheet Excel file
        exp_name: Experiment identifier

    Returns:
        Tuple of (plate_df, series_map_df)
        - plate_df: Long-format DataFrame with one row per well
        - series_map_df: Series number mapping (or None if not present)

    Raises:
        RuntimeError: If parsing fails or no valid plate sheets found
    """
    # Generate all 96 well positions (A01-H12)
    wells = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]
    plate_df = pd.DataFrame({"well": wells, "experiment_date": exp_name})

    # Sheets to skip (not 8×12 plate format)
    skip_sheets = ['Export Summary']

    # Numeric sheets (will be parsed as float)
    numeric_sheets = ['temperature', 'embryos_per_well']

    series_map_df = None

    try:
        with pd.ExcelFile(xlsx_path) as xlf:
            valid_sheets = []

            for sheet_name in xlf.sheet_names:
                # Handle series_number_map separately (not 8×12 format)
                if sheet_name == 'series_number_map':
                    try:
                        series_map_df = xlf.parse(sheet_name)
                        print(f"✅ Found series_number_map with {len(series_map_df)} rows")
                    except Exception as e:
                        print(f"⚠️  Could not parse series_number_map: {e}")
                    continue

                # Skip other non-plate sheets
                if sheet_name in skip_sheets:
                    continue

                try:
                    # Try to read sheet as 8×12 plate format
                    df = xlf.parse(sheet_name, header=0)

                    # Check if it has the right shape (at least 8 rows, 13 columns including row labels)
                    if df.shape[0] < 8 or df.shape[1] < 13:
                        print(f"⚠️  Skipping sheet '{sheet_name}': wrong shape {df.shape}, expected at least (8, 13)")
                        continue

                    # Extract 8×12 plate data (rows 0-7, columns 1-12)
                    plate_data = df.iloc[:8, 1:13]

                    # Convert to appropriate dtype and flatten
                    if sheet_name in numeric_sheets:
                        arr = plate_data.to_numpy(dtype=float).ravel()
                    else:
                        arr = plate_data.to_numpy(dtype=str).ravel()

                    plate_df[sheet_name] = arr
                    valid_sheets.append(sheet_name)

                except Exception as e:
                    print(f"⚠️  Skipping sheet '{sheet_name}': {e}")
                    continue

            if not valid_sheets:
                raise RuntimeError(f"No valid 8×12 plate sheets found in {xlsx_path}")

            print(f"✅ Loaded {len(valid_sheets)} plate sheets: {', '.join(valid_sheets)}")

        # Remove wells lacking start_age_hpf (empty string or NaN)
        if 'start_age_hpf' in plate_df.columns:
            mask = plate_df["start_age_hpf"].astype(str).str.strip() != ""
            plate_df = plate_df.loc[mask].reset_index(drop=True)

        # Normalize column names
        plate_df = _normalize_column_names(plate_df)

        return plate_df, series_map_df

    except Exception as e:
        raise RuntimeError(
            f"❌ Error reading or parsing Excel file: {xlsx_path}\n"
            f"Original error:\n{e}"
        ) from e


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to match schema expectations.

    Handles common variations in plate layout files:
    - well → well_index
    - experiment_date → experiment_id
    - chem_perturbation → treatment
    - temperature → temperature_c

    Args:
        df: Raw DataFrame from input file

    Returns:
        DataFrame with normalized column names
    """
    # Define column mappings (input_name → schema_name)
    column_mappings = {
        # Well identifiers
        'well': 'well_index',
        'Well': 'well_index',
        'well_name': 'well_index',

        # Experiment ID
        'experiment_date': 'experiment_id',
        'experiment': 'experiment_id',
        'exp_id': 'experiment_id',

        # Treatment
        'chem_perturbation': 'treatment',
        'chemical_perturbation': 'treatment',
        'drug': 'treatment',

        # Temperature (keep as 'temperature' per schema)
        'temp': 'temperature',
        'temp_c': 'temperature',
        'temperature_c': 'temperature',

        # Age
        'start_age': 'start_age_hpf',
        'age_hpf': 'start_age_hpf',
        'age': 'start_age_hpf',

        # Embryos
        'embryos': 'embryos_per_well',
        'n_embryos': 'embryos_per_well',
    }

    # Apply mappings
    df = df.rename(columns=column_mappings)

    return df
