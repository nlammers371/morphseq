"""
Align and merge validated scope and plate metadata.

This module joins microscope-extracted metadata with plate layout annotations
to create a unified metadata table for downstream processing.
"""

from pathlib import Path
import pandas as pd

from data_pipeline.schemas.scope_and_plate_metadata import REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA
from data_pipeline.io.validators import validate_dataframe_schema


def align_scope_and_plate_metadata(
    plate_metadata_csv: Path,
    scope_metadata_csv: Path,
    output_csv: Path
) -> pd.DataFrame:
    """
    Join plate and scope metadata on experiment_id and well_id.

    Both input CSVs should already be validated against their respective schemas.
    This function performs a left join (keeping all scope metadata rows) and
    validates the result.

    Args:
        plate_metadata_csv: Path to validated plate_metadata.csv
        scope_metadata_csv: Path to validated scope_metadata.csv
        output_csv: Path where merged CSV will be written

    Returns:
        Merged and validated DataFrame

    Raises:
        ValueError: If required columns are missing or contain null values
        FileNotFoundError: If input files don't exist
    """
    # Load validated inputs
    if not plate_metadata_csv.exists():
        raise FileNotFoundError(f"Plate metadata file not found: {plate_metadata_csv}")
    if not scope_metadata_csv.exists():
        raise FileNotFoundError(f"Scope metadata file not found: {scope_metadata_csv}")

    plate_df = pd.read_csv(plate_metadata_csv)
    scope_df = pd.read_csv(scope_metadata_csv)

    # Merge on experiment_id and well_id
    # Use left join to keep all scope metadata (images)
    # Each image should match exactly one well from the plate layout
    merged_df = scope_df.merge(
        plate_df,
        on=['experiment_id', 'well_id'],
        how='left',
        validate='many_to_one',  # Many images per well
        suffixes=('', '_plate')
    )

    # Handle potential duplicate columns from merge
    # Keep scope version for well_index if both exist
    if 'well_index_plate' in merged_df.columns:
        merged_df = merged_df.drop(columns=['well_index_plate'])

    # Check if any scope rows failed to match plate data
    unmatched = merged_df[merged_df['genotype'].isna()]
    if len(unmatched) > 0:
        unmatched_wells = unmatched['well_id'].unique()
        raise ValueError(
            f"Found {len(unmatched)} scope metadata rows with no matching plate data. "
            f"Unmatched well_ids: {list(unmatched_wells)[:10]}"
        )

    # Validate against schema
    validate_dataframe_schema(merged_df, REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA, "scope_and_plate_metadata")

    # Write validated CSV
    merged_df.to_csv(output_csv, index=False)

    return merged_df
