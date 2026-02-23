"""
Consolidate all feature extraction outputs into single table.

Merges mask geometry, pose/kinematics, fraction alive, and stage predictions
into consolidated_snip_features.csv with schema validation.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import warnings

from ..schemas.features import REQUIRED_COLUMNS_FEATURES


def consolidate_snip_features(
    tracking_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    kinematics_df: pd.DataFrame,
    fraction_alive_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Consolidate all feature tables into single DataFrame.

    Args:
        tracking_df: Segmentation tracking with snip_id, embryo_id, frame_index
        geometry_df: Mask geometry metrics (area, perimeter, etc.)
        kinematics_df: Pose and kinematics metrics
        fraction_alive_df: Viability metrics
        stage_df: Developmental stage predictions
        metadata_df: Optional plate/scope metadata

    Returns:
        Consolidated features DataFrame with schema validation
    """
    # Start with tracking base
    consolidated = tracking_df.copy()

    # Merge geometry metrics
    consolidated = consolidated.merge(
        geometry_df,
        on='snip_id',
        how='left',
        suffixes=('', '_geometry')
    )

    # Merge kinematics
    consolidated = consolidated.merge(
        kinematics_df,
        on='snip_id',
        how='left',
        suffixes=('', '_kinematics')
    )

    # Merge fraction alive
    consolidated = consolidated.merge(
        fraction_alive_df,
        on='snip_id',
        how='left',
        suffixes=('', '_via')
    )

    # Merge stage predictions
    consolidated = consolidated.merge(
        stage_df,
        on='snip_id',
        how='left',
        suffixes=('', '_stage')
    )

    # Merge metadata if provided
    if metadata_df is not None:
        # Determine merge key (well_id or experiment_id)
        if 'well_id' in metadata_df.columns:
            merge_key = 'well_id'
        elif 'experiment_id' in metadata_df.columns:
            merge_key = 'experiment_id'
        else:
            warnings.warn("Metadata missing well_id or experiment_id, skipping metadata merge")
            metadata_df = None

        if metadata_df is not None:
            consolidated = consolidated.merge(
                metadata_df,
                on=merge_key,
                how='left',
                suffixes=('', '_metadata')
            )

    # Validate schema
    validate_feature_schema(consolidated)

    return consolidated


def validate_feature_schema(df: pd.DataFrame) -> None:
    """
    Validate consolidated features against REQUIRED_COLUMNS_FEATURES.

    Raises:
        ValueError: If required columns missing or contain all-null values
    """
    missing_cols = []
    null_cols = []

    for col in REQUIRED_COLUMNS_FEATURES:
        if col not in df.columns:
            missing_cols.append(col)
        elif df[col].isna().all():
            null_cols.append(col)

    if missing_cols:
        raise ValueError(
            f"Consolidated features missing required columns: {missing_cols}"
        )

    if null_cols:
        warnings.warn(
            f"Consolidated features contain all-null required columns: {null_cols}. "
            "This may indicate missing upstream data."
        )


def save_consolidated_features(
    consolidated_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save consolidated features with validation.

    Args:
        consolidated_df: Consolidated features DataFrame
        output_path: Path to save CSV

    Raises:
        ValueError: If schema validation fails
    """
    # Validate before saving
    validate_feature_schema(consolidated_df)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_df.to_csv(output_path, index=False)

    print(f"✅ Saved consolidated features: {output_path}")
    print(f"   {len(consolidated_df)} snips with {len(consolidated_df.columns)} columns")


def load_and_consolidate_features(
    tracking_path: Path,
    geometry_path: Path,
    kinematics_path: Path,
    fraction_alive_path: Path,
    stage_path: Path,
    metadata_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load individual feature CSVs and consolidate into single table.

    Args:
        tracking_path: Path to segmentation_tracking.csv
        geometry_path: Path to mask_geometry_metrics.csv
        kinematics_path: Path to pose_kinematics_metrics.csv
        fraction_alive_path: Path to fraction_alive.csv
        stage_path: Path to stage_predictions.csv
        metadata_path: Optional path to scope_and_plate_metadata.csv
        output_path: Optional path to save consolidated CSV

    Returns:
        Consolidated features DataFrame
    """
    # Load all feature tables
    tracking_df = pd.read_csv(tracking_path)
    geometry_df = pd.read_csv(geometry_path)
    kinematics_df = pd.read_csv(kinematics_path)
    fraction_alive_df = pd.read_csv(fraction_alive_path)
    stage_df = pd.read_csv(stage_path)

    metadata_df = None
    if metadata_path and metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path)

    # Consolidate
    consolidated = consolidate_snip_features(
        tracking_df,
        geometry_df,
        kinematics_df,
        fraction_alive_df,
        stage_df,
        metadata_df,
    )

    # Save if output path provided
    if output_path:
        save_consolidated_features(consolidated, output_path)

    return consolidated
