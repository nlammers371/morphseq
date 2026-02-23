#!/usr/bin/env python3
"""
Analysis-Ready Table Assembly

Assembles the final analysis-ready table by joining:
- consolidated_snip_features.csv (morphology + kinematics + stage)
- consolidated_qc_flags.csv (all QC flags + use_embryo gate)
- embedding latents (optional, stub for MVP)
- plate/scope metadata (from features table)

Output:
- features_qc_embeddings.csv with embedding_calculated column

This table is the single source for all downstream analysis and notebooks.

Authors: Wave 6 Implementation
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

# Import schema for validation
from src.data_pipeline.schemas.analysis_ready import REQUIRED_COLUMNS_ANALYSIS_READY


def assemble_features_qc_embeddings(
    features_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    embeddings_df: Optional[pd.DataFrame] = None,
    merge_on: str = 'snip_id'
) -> pd.DataFrame:
    """
    Assemble final analysis-ready table with features, QC, and embeddings.

    Parameters
    ----------
    features_df : pd.DataFrame
        consolidated_snip_features.csv with columns:
        [snip_id, embryo_id, time_int, well_id, well_index, experiment_id,
         area_um2, perimeter_um, length_um, width_um, centroid_x, centroid_y,
         orientation, displacement_um, speed_um_per_s, predicted_stage_hpf,
         fraction_alive, + plate/scope metadata columns]
    qc_df : pd.DataFrame
        consolidated_qc_flags.csv with columns:
        [snip_id, use_embryo, dead_flag, sa_outlier_flag, edge_flag,
         discontinuous_mask_flag, overlapping_mask_flag, yolk_flag,
         focus_flag, bubble_flag, mask_quality_flag, + death metadata]
    embeddings_df : pd.DataFrame, optional
        Latent embeddings with columns:
        [snip_id, embedding_model, z0, z1, ..., z{dim-1}]
        If None, embedding_calculated will be False for all rows
    merge_on : str, default 'snip_id'
        Column to merge on

    Returns
    -------
    pd.DataFrame
        Analysis-ready table with embedding_calculated column

    Notes
    -----
    - embedding_calculated = True if embeddings present for that snip_id
    - Validates against REQUIRED_COLUMNS_ANALYSIS_READY schema
    - Preserves all feature columns, QC flags, and optional embeddings
    - Suitable for direct use in analysis notebooks and metric learning
    """
    print(f"🔍 Assembling analysis-ready table...")
    print(f"   Features: {len(features_df)} snips")
    print(f"   QC flags: {len(qc_df)} snips")
    if embeddings_df is not None:
        print(f"   Embeddings: {len(embeddings_df)} snips")

    # Start with features (has most columns)
    analysis_df = features_df.copy()

    # Merge QC flags
    # Only merge QC-specific columns (avoid duplicating metadata)
    qc_cols_to_merge = [
        merge_on,
        'use_embryo',
        'dead_flag',
        'sa_outlier_flag',
        'edge_flag',
        'discontinuous_mask_flag',
        'overlapping_mask_flag',
        'yolk_flag',
        'focus_flag',
        'bubble_flag',
        'mask_quality_flag',
        'death_inflection_time_int',
        'death_predicted_stage_hpf'
    ]
    # Filter to only columns that exist in qc_df
    qc_cols_available = [col for col in qc_cols_to_merge if col in qc_df.columns]

    analysis_df = analysis_df.merge(
        qc_df[qc_cols_available],
        on=merge_on,
        how='left',
        suffixes=('', '_qc')
    )

    # Add embeddings if provided
    if embeddings_df is not None:
        # Merge embeddings
        analysis_df = analysis_df.merge(
            embeddings_df,
            on=merge_on,
            how='left',
            suffixes=('', '_emb')
        )

        # embedding_calculated = True if any embedding column (z0...z{n}) is non-null
        embedding_cols = [col for col in embeddings_df.columns if col.startswith('z')]
        if embedding_cols:
            # Check if at least one embedding dimension is non-null
            analysis_df['embedding_calculated'] = ~analysis_df[embedding_cols[0]].isna()
        else:
            # No embedding columns found
            analysis_df['embedding_calculated'] = False
    else:
        # No embeddings provided - all False
        analysis_df['embedding_calculated'] = False

    # Fill missing QC flags with False (for rows not in qc_df)
    qc_flag_cols = [
        'use_embryo',
        'dead_flag',
        'sa_outlier_flag',
        'edge_flag',
        'discontinuous_mask_flag',
        'overlapping_mask_flag',
        'yolk_flag',
        'focus_flag',
        'bubble_flag',
        'mask_quality_flag'
    ]
    for col in qc_flag_cols:
        if col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].fillna(False).astype(bool)

    # Validate schema
    validate_analysis_ready_schema(analysis_df, require_embeddings=embeddings_df is not None)

    # Summary statistics
    print_analysis_ready_summary(analysis_df)

    return analysis_df


def validate_analysis_ready_schema(
    df: pd.DataFrame,
    require_embeddings: bool = False
):
    """
    Validate analysis-ready table against schema.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready dataframe
    require_embeddings : bool, default False
        If True, require embedding columns (z0...z{n}) to be present
        If False, only check core columns

    Raises
    ------
    ValueError
        If required columns are missing
    """
    # Check core required columns (excluding embeddings)
    core_required = [col for col in REQUIRED_COLUMNS_ANALYSIS_READY if not col.startswith('z')]
    missing_cols = [col for col in core_required if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Analysis-ready table missing required columns: {missing_cols}\n"
            f"Expected schema: {REQUIRED_COLUMNS_ANALYSIS_READY}"
        )

    # Check for embedding columns if required
    if require_embeddings:
        embedding_cols = [col for col in df.columns if col.startswith('z') and col[1:].isdigit()]
        if not embedding_cols:
            raise ValueError("Embeddings required but no embedding columns (z0...z{n}) found")

    # Check critical fields are not all-null
    critical_cols = ['snip_id', 'embryo_id', 'experiment_id', 'embedding_calculated']
    for col in critical_cols:
        if col in df.columns and df[col].isna().all():
            raise ValueError(f"Critical column '{col}' is all-null")

    print(f"✅ Schema validation passed")


def print_analysis_ready_summary(df: pd.DataFrame):
    """Print summary statistics for analysis-ready table."""
    total = len(df)
    usable = df['use_embryo'].sum() if 'use_embryo' in df.columns else 0
    has_embeddings = df['embedding_calculated'].sum() if 'embedding_calculated' in df.columns else 0

    print(f"\n📊 Analysis-Ready Table Summary:")
    print(f"   Total snips: {total}")
    print(f"   Usable (use_embryo=True): {usable} ({100*usable/total:.1f}%)")
    print(f"   Has embeddings: {has_embeddings} ({100*has_embeddings/total:.1f}%)")
    print(f"   Columns: {len(df.columns)}")

    # Count embedding dimensions
    embedding_cols = [col for col in df.columns if col.startswith('z') and col[1:].isdigit()]
    if embedding_cols:
        print(f"   Embedding dimensions: {len(embedding_cols)}")

    # Experiment breakdown
    if 'experiment_id' in df.columns:
        n_experiments = df['experiment_id'].nunique()
        print(f"   Experiments: {n_experiments}")


def save_analysis_ready(
    analysis_df: pd.DataFrame,
    output_path: Path,
    validate_schema: bool = True
):
    """
    Save analysis-ready table with optional schema validation.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Analysis-ready dataframe
    output_path : Path
        Output CSV path
    validate_schema : bool, default True
        Whether to validate against schema
    """
    if validate_schema:
        validate_analysis_ready_schema(analysis_df, require_embeddings=False)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    analysis_df.to_csv(output_path, index=False)
    print(f"💾 Saved analysis-ready table: {output_path}")
    print(f"   Rows: {len(analysis_df)}")
    print(f"   Columns: {len(analysis_df.columns)}")


def filter_for_analysis(
    analysis_df: pd.DataFrame,
    use_embryo_only: bool = True,
    require_embeddings: bool = False
) -> pd.DataFrame:
    """
    Filter analysis-ready table for downstream use.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        Analysis-ready table
    use_embryo_only : bool, default True
        If True, keep only rows with use_embryo=True
    require_embeddings : bool, default False
        If True, keep only rows with embedding_calculated=True

    Returns
    -------
    pd.DataFrame
        Filtered analysis table
    """
    filtered = analysis_df.copy()

    if use_embryo_only and 'use_embryo' in filtered.columns:
        filtered = filtered[filtered['use_embryo'] == True]

    if require_embeddings and 'embedding_calculated' in filtered.columns:
        filtered = filtered[filtered['embedding_calculated'] == True]

    print(f"📊 Filtered analysis table: {len(filtered)}/{len(analysis_df)} rows")

    return filtered


def main():
    """Example usage and documentation."""
    print("Analysis-Ready Table Assembly")
    print("=" * 50)
    print("Assembles final analysis table with features, QC, and embeddings")
    print("\nInputs:")
    print("  - consolidated_snip_features.csv")
    print("  - consolidated_qc_flags.csv")
    print("  - {experiment_id}_latents.csv (optional)")
    print("\nOutput:")
    print("  - features_qc_embeddings.csv")
    print("\nColumns:")
    print("  - All feature columns (morphology, kinematics, stage)")
    print("  - All QC flags (dead, sa_outlier, edge, etc.)")
    print("  - use_embryo gate (for filtering)")
    print("  - Embedding columns z0...z{dim-1} (if available)")
    print("  - embedding_calculated (boolean helper)")
    print("\nUsage:")
    print("  from analysis_ready import assemble_features_qc_embeddings")
    print("  analysis_df = assemble_features_qc_embeddings(")
    print("      features_df, qc_df, embeddings_df")
    print("  )")
    print("\n  # Filter for analysis")
    print("  from analysis_ready import filter_for_analysis")
    print("  usable_df = filter_for_analysis(")
    print("      analysis_df, use_embryo_only=True, require_embeddings=False")
    print("  )")


if __name__ == "__main__":
    main()
