#!/usr/bin/env python3
"""
QC Consolidation Module

Merges all QC flags from different modules and computes the final use_embryo gate.

Inputs:
- segmentation_quality_qc.csv (edge, discontinuous, overlapping_mask flags)
- auxiliary_mask_qc.csv (yolk, focus, bubble flags)
- embryo_death_qc.csv (dead_flag, death_inflection_time_int, death_predicted_stage_hpf)
- surface_area_outliers_qc.csv (sa_outlier_flag)

Output:
- consolidated_qc_flags.csv with all flags + use_embryo column

Logic:
- use_embryo = NOT (any QC_FAIL_FLAG is True)
- QC_FAIL_FLAGS: dead_flag, sa_outlier_flag, yolk_flag, edge_flag,
                discontinuous_mask_flag, overlapping_mask_flag, focus_flag, bubble_flag

Authors: Wave 6 Implementation
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

# Import schema for validation
from src.data_pipeline.schemas.quality_control import REQUIRED_COLUMNS_QC, QC_FAIL_FLAGS


def consolidate_qc_flags(
    segmentation_qc_df: pd.DataFrame,
    imaging_qc_df: pd.DataFrame,
    death_qc_df: pd.DataFrame,
    size_qc_df: pd.DataFrame,
    fraction_alive_df: Optional[pd.DataFrame] = None,
    merge_on: str = 'snip_id'
) -> pd.DataFrame:
    """
    Merge all QC flags and compute final use_embryo gate.

    Parameters
    ----------
    segmentation_qc_df : pd.DataFrame
        Segmentation QC flags:
        [snip_id, edge_flag, discontinuous_mask_flag, overlapping_mask_flag, mask_quality_flag]
    imaging_qc_df : pd.DataFrame
        Imaging QC flags:
        [snip_id, yolk_flag, focus_flag, bubble_flag]
    death_qc_df : pd.DataFrame
        Death detection flags:
        [snip_id, dead_flag2, dead_inflection_time_int]
    size_qc_df : pd.DataFrame
        Size validation flags:
        [snip_id, sa_outlier_flag]
    fraction_alive_df : pd.DataFrame, optional
        Fraction alive data for dead_flag computation:
        [snip_id, fraction_alive]
        If None, dead_flag will be computed from dead_flag2
    merge_on : str, default 'snip_id'
        Column to merge on

    Returns
    -------
    pd.DataFrame
        Consolidated QC flags with use_embryo column

    Notes
    -----
    - use_embryo = NOT (any flag in QC_FAIL_FLAGS is True)
    - QC_FAIL_FLAGS defined in schemas/quality_control.py
    - All flags default to False if column missing
    - Schema validation ensures REQUIRED_COLUMNS_QC are present
    """
    print(f"🔍 Consolidating QC flags...")

    # Start with segmentation QC (has core IDs)
    consolidated = segmentation_qc_df.copy()

    # Merge imaging QC
    consolidated = consolidated.merge(
        imaging_qc_df,
        on=merge_on,
        how='left',
        suffixes=('', '_imaging')
    )

    # Merge death QC
    consolidated = consolidated.merge(
        death_qc_df,
        on=merge_on,
        how='left',
        suffixes=('', '_death')
    )

    # Merge size QC
    consolidated = consolidated.merge(
        size_qc_df[[merge_on, 'sa_outlier_flag']],
        on=merge_on,
        how='left',
        suffixes=('', '_size')
    )

    # Compute dead_flag from fraction_alive if provided
    if fraction_alive_df is not None:
        consolidated = consolidated.merge(
            fraction_alive_df[[merge_on, 'fraction_alive']],
            on=merge_on,
            how='left'
        )
        # dead_flag = fraction_alive < 0.9 (UNet viability threshold)
        consolidated['dead_flag'] = consolidated['fraction_alive'] < 0.9
    else:
        # Fallback: use dead_flag2 as dead_flag if fraction_alive not available
        if 'dead_flag2' in consolidated.columns:
            consolidated['dead_flag'] = consolidated['dead_flag2'].fillna(False)
        else:
            consolidated['dead_flag'] = False

    # Ensure all QC_FAIL_FLAGS exist and fill NaN with False
    for flag_col in QC_FAIL_FLAGS:
        if flag_col not in consolidated.columns:
            consolidated[flag_col] = False
        consolidated[flag_col] = consolidated[flag_col].fillna(False).astype(bool)

    # Compute use_embryo flag: NOT (any fail flag is True)
    # Use .any(axis=1) to check if ANY flag is True for each row
    qc_fail_any = consolidated[QC_FAIL_FLAGS].any(axis=1)
    consolidated['use_embryo'] = ~qc_fail_any

    # Add death metadata columns if available
    if 'dead_inflection_time_int' in death_qc_df.columns:
        # Already merged, ensure it's in output
        pass

    # Compute death_predicted_stage_hpf (stage at death inflection)
    if 'predicted_stage_hpf' in consolidated.columns and 'dead_inflection_time_int' in consolidated.columns:
        # For each embryo, find the stage_hpf at death_inflection_time_int
        # This is already done in death_detection.py, so we just need to ensure it's present
        # If not present, we can compute it here
        if 'death_predicted_stage_hpf' not in consolidated.columns:
            consolidated['death_predicted_stage_hpf'] = np.nan
            # For rows where dead_inflection_time_int is set, copy predicted_stage_hpf
            # This is a simplification - proper implementation would look up the exact time_int
            has_inflection = ~consolidated['dead_inflection_time_int'].isna()
            if has_inflection.any():
                # Simplified: use current predicted_stage_hpf for now
                # TODO: Lookup exact stage_hpf at inflection time_int from features table
                consolidated.loc[has_inflection, 'death_predicted_stage_hpf'] = \
                    consolidated.loc[has_inflection, 'predicted_stage_hpf']

    # Validate against schema
    validate_qc_schema(consolidated)

    # Summary statistics
    print_qc_summary(consolidated)

    return consolidated


def validate_qc_schema(df: pd.DataFrame):
    """
    Validate consolidated QC dataframe against REQUIRED_COLUMNS_QC schema.

    Raises
    ------
    ValueError
        If required columns are missing or contain all-null values
    """
    missing_cols = [col for col in REQUIRED_COLUMNS_QC if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Consolidated QC missing required columns: {missing_cols}\n"
            f"Expected schema: {REQUIRED_COLUMNS_QC}"
        )

    # Check for all-null columns in critical fields
    critical_cols = ['snip_id', 'embryo_id', 'use_embryo']
    for col in critical_cols:
        if col in df.columns and df[col].isna().all():
            raise ValueError(f"Critical column '{col}' is all-null")

    print(f"✅ Schema validation passed ({len(REQUIRED_COLUMNS_QC)} required columns)")


def print_qc_summary(df: pd.DataFrame):
    """Print summary of QC flagging statistics."""
    total = len(df)
    usable = df['use_embryo'].sum()
    flagged = total - usable

    print(f"\n📊 QC Consolidation Summary:")
    print(f"   Total snips: {total}")
    print(f"   Usable (use_embryo=True): {usable} ({100*usable/total:.1f}%)")
    print(f"   Flagged (use_embryo=False): {flagged} ({100*flagged/total:.1f}%)")
    print(f"\n   Flag breakdown:")

    for flag_col in QC_FAIL_FLAGS:
        if flag_col in df.columns:
            flag_count = df[flag_col].sum()
            if flag_count > 0:
                print(f"      {flag_col}: {flag_count} ({100*flag_count/total:.1f}%)")


def save_consolidated_qc(
    consolidated_df: pd.DataFrame,
    output_path: Path,
    validate_schema: bool = True
):
    """
    Save consolidated QC flags with optional schema validation.

    Parameters
    ----------
    consolidated_df : pd.DataFrame
        Consolidated QC dataframe
    output_path : Path
        Output CSV path
    validate_schema : bool, default True
        Whether to validate against REQUIRED_COLUMNS_QC
    """
    if validate_schema:
        validate_qc_schema(consolidated_df)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    consolidated_df.to_csv(output_path, index=False)
    print(f"💾 Saved consolidated QC: {output_path}")
    print(f"   Rows: {len(consolidated_df)}")
    print(f"   Columns: {len(consolidated_df.columns)}")


def main():
    """Example usage and documentation."""
    print("QC Consolidation Module")
    print("=" * 50)
    print("Merges all QC flags and computes use_embryo gate")
    print("\nInputs:")
    print("  - segmentation_quality_qc.csv")
    print("  - auxiliary_mask_qc.csv")
    print("  - embryo_death_qc.csv")
    print("  - surface_area_outliers_qc.csv")
    print("\nOutput:")
    print("  - consolidated_qc_flags.csv")
    print("\nLogic:")
    print("  use_embryo = NOT (any QC_FAIL_FLAG is True)")
    print(f"  QC_FAIL_FLAGS: {QC_FAIL_FLAGS}")
    print("\nUsage:")
    print("  from quality_control.consolidation import consolidate_qc_flags")
    print("  consolidated_df = consolidate_qc_flags(")
    print("      segmentation_qc_df, imaging_qc_df, death_qc_df, size_qc_df")
    print("  )")


if __name__ == "__main__":
    main()
