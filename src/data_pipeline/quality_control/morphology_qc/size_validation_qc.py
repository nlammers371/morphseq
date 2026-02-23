#!/usr/bin/env python3
"""
Size Validation QC Module

Detects surface area outliers using two-sided thresholding against
global reference curves derived from wild-type control embryos.

Flags embryos with:
- SA > k_upper × p95 (too large: segmentation artifacts, debris)
- SA < k_lower × p5 (too small: incomplete masks, dead/dying embryos)

This module wraps compute_sa_outlier_flag() from surface_area_outlier_detection.py
to provide a consistent QC module interface.

Authors: Wave 6 Implementation
Date: 2025-11-06
"""

import pandas as pd
from pathlib import Path
from typing import Optional

# Import core SA outlier detection function
from src.data_pipeline.quality_control.surface_area_outlier_detection import compute_sa_outlier_flag


def compute_size_validation_qc(
    features_df: pd.DataFrame,
    sa_reference_path: Path,
    k_upper: float = 1.2,
    k_lower: float = 0.9,
    stage_col: str = "predicted_stage_hpf",
    sa_col: str = "surface_area_um"
) -> pd.DataFrame:
    """
    Compute surface area outlier flags using global reference curves.

    This is a thin wrapper around compute_sa_outlier_flag() that provides
    a consistent interface for the QC pipeline.

    Parameters
    ----------
    features_df : pd.DataFrame
        consolidated_snip_features.csv with columns:
        [snip_id, embryo_id, predicted_stage_hpf, surface_area_um, ...]
    sa_reference_path : Path
        Path to metadata/sa_reference_curves.csv
    k_upper : float, default 1.2
        Upper threshold multiplier (SA > k_upper × p95 flagged)
    k_lower : float, default 0.9
        Lower threshold multiplier (SA < k_lower × p5 flagged)
    stage_col : str, default "predicted_stage_hpf"
        Column name for developmental stage
    sa_col : str, default "surface_area_um"
        Column name for surface area

    Returns
    -------
    pd.DataFrame
        QC flags with columns:
        [snip_id, embryo_id, sa_outlier_flag]

    Notes
    -----
    - Requires sa_reference_curves.csv with [stage_hpf, p5, p50, p95, n]
    - Reference curves built from wild-type control embryos across all experiments
    - Default thresholds (k_upper=1.2, k_lower=0.9) validated on test cases
    - See surface_area_outlier_detection.py for detailed documentation

    Examples
    --------
    >>> from pathlib import Path
    >>> ref_path = Path('metadata/sa_reference_curves.csv')
    >>> qc_df = compute_size_validation_qc(features_df, ref_path)
    >>> qc_df['sa_outlier_flag'].sum()  # Count flagged snips
    """
    print(f"🔍 Computing size validation QC for {len(features_df)} snips...")

    # Check required columns
    required_cols = ['snip_id', 'embryo_id', stage_col, sa_col]
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Call core SA outlier detection
    result_df = compute_sa_outlier_flag(
        df=features_df,
        sa_reference_path=sa_reference_path,
        k_upper=k_upper,
        k_lower=k_lower,
        stage_col=stage_col,
        sa_col=sa_col
    )

    # Extract QC columns
    qc_df = result_df[['snip_id', 'embryo_id', 'sa_outlier_flag']].copy()

    # Summary already printed by compute_sa_outlier_flag()
    print(f"✅ Size validation QC complete")

    return qc_df


def main():
    """Example usage and documentation."""
    print("Size Validation QC Module")
    print("=" * 50)
    print("Detects surface area outliers using reference curves")
    print("- Flags too large: SA > 1.2 × p95 (artifacts)")
    print("- Flags too small: SA < 0.9 × p5 (incomplete/dead)")
    print("\nUsage:")
    print("  from quality_control.morphology_qc import compute_size_validation_qc")
    print("  from pathlib import Path")
    print("  ref_path = Path('metadata/sa_reference_curves.csv')")
    print("  qc_df = compute_size_validation_qc(features_df, ref_path)")


if __name__ == "__main__":
    main()
