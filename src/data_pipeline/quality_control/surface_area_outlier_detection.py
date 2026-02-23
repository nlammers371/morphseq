#!/usr/bin/env python3
"""
Two-Sided Surface Area Outlier Detection

This module implements two-sided outlier detection for embryo surface area
measurements using global reference curves derived from wild-type control embryos.

Key Features:
- Flags embryos that are too large (SA > k_upper × p95) - segmentation artifacts
- Flags embryos that are too small (SA < k_lower × p5) - incomplete masks, dead embryos
- Uses global reference curves aggregated across all experiments
- Tuned thresholds: k_upper=1.4, k_lower=0.7 (validated on test cases)

Algorithm:
1. Load global reference curves (p5, p50, p95 vs stage_hpf)
2. For each embryo timepoint, interpolate expected p5 and p95 at current stage
3. Flag if SA > k_upper × p95 OR SA < k_lower × p5
4. Return dataframe with sa_outlier_flag column added/updated

Reference Generation:
- See: src/data_pipeline/quality_control/generate_references/build_sa_reference.py
- Reference file: metadata/sa_reference_curves.csv
- Regenerate quarterly or after major pipeline changes

Authors: SA Outlier Detection Team
Date: 2025-10-08
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def compute_sa_outlier_flag(
    df: pd.DataFrame,
    sa_reference_path: Path,
    k_upper: float = 1.2,
    k_lower: float = 0.9,
    stage_col: str = "predicted_stage_hpf",
    sa_col: str = "surface_area_um",
) -> pd.DataFrame:
    """
    Two-sided surface area outlier detection using global reference curves.

    Flags embryos with:
    - SA > k_upper × p95 (too large: segmentation artifacts, debris)
    - SA < k_lower × p5 (too small: incomplete masks, dead/dying embryos)

    The reference curves (p5, p50, p95) are derived from wild-type control embryos
    across all experiments, providing robust stage-specific thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with stage and surface area columns
    sa_reference_path : Path
        Path to sa_reference_curves.csv containing columns:
        [stage_hpf, p5, p50, p95, n]
    k_upper : float, default 1.2
        Upper threshold multiplier (SA > k_upper × p95 flagged)
    k_lower : float, default 0.9
        Lower threshold multiplier (SA < k_lower × p5 flagged)
    stage_col : str, default "predicted_stage_hpf"
        Column name for developmental stage in hours post-fertilization
    sa_col : str, default "surface_area_um"
        Column name for surface area in square micrometers

    Returns
    -------
    pd.DataFrame
        Input df with 'sa_outlier_flag' column added (True if outlier)

    Raises
    ------
    FileNotFoundError
        If sa_reference_path does not exist
    ValueError
        If required columns are missing from df or reference file

    Examples
    --------
    >>> from pathlib import Path
    >>> df = pd.DataFrame({
    ...     'predicted_stage_hpf': [30.0, 50.0, 70.0],
    ...     'surface_area_um': [400000, 1200000, 1500000]
    ... })
    >>> ref_path = Path('metadata/sa_reference_curves.csv')
    >>> df = compute_sa_outlier_flag(df, ref_path)
    >>> df['sa_outlier_flag']
    0    True   # Too small
    1    False
    2    False

    Notes
    -----
    - Reference curves are pre-smoothed with Savitzky-Golay filter
    - Gaps and edge bins are filled during reference generation
    - Default thresholds (k_upper=1.2, k_lower=0.9) validated on test cases:
      20250711_F03_e01, F06_e01, H07_e01 (all caught, including early stages <50 hpf)

    See Also
    --------
    generate_references.build_sa_reference : Script to regenerate reference curves
    """
    df = df.copy()

    # Validate inputs
    if not Path(sa_reference_path).exists():
        raise FileNotFoundError(
            f"SA reference file not found: {sa_reference_path}\n"
            f"Generate it using: src/data_pipeline/quality_control/generate_references/build_sa_reference.py"
        )

    if stage_col not in df.columns:
        raise ValueError(f"Missing required column: {stage_col}")
    if sa_col not in df.columns:
        raise ValueError(f"Missing required column: {sa_col}")

    # Load reference curves
    try:
        ref_df = pd.read_csv(sa_reference_path)
    except Exception as e:
        raise ValueError(f"Failed to load SA reference file: {e}")

    required_cols = ['stage_hpf', 'p5', 'p95']
    missing_cols = [c for c in required_cols if c not in ref_df.columns]
    if missing_cols:
        raise ValueError(
            f"SA reference file missing required columns: {missing_cols}\n"
            f"Expected columns: {required_cols}"
        )

    # Interpolate reference values at each embryo's stage
    # Reference curves are pre-filled/extrapolated, so simple interp is sufficient
    df['_ref_p5'] = np.interp(df[stage_col], ref_df['stage_hpf'], ref_df['p5'])
    df['_ref_p95'] = np.interp(df[stage_col], ref_df['stage_hpf'], ref_df['p95'])

    # Calculate thresholds
    upper_threshold = k_upper * df['_ref_p95']
    lower_threshold = k_lower * df['_ref_p5']

    # Two-sided flagging
    too_large = df[sa_col] > upper_threshold
    too_small = df[sa_col] < lower_threshold
    df['sa_outlier_flag'] = too_large | too_small

    # Clean up temporary columns
    df.drop(columns=['_ref_p5', '_ref_p95'], inplace=True)

    # Summary stats
    n_flagged = df['sa_outlier_flag'].sum()
    n_too_large = too_large.sum()
    n_too_small = too_small.sum()

    print(f"✅ SA QC (two-sided): {n_flagged} frames flagged")
    print(f"   Too large (>{k_upper}×p95): {n_too_large}")
    print(f"   Too small (<{k_lower}×p5): {n_too_small}")

    return df


def validate_sa_reference(sa_reference_path: Path) -> dict:
    """
    Validate SA reference file format and content.

    Parameters
    ----------
    sa_reference_path : Path
        Path to sa_reference_curves.csv

    Returns
    -------
    dict
        Validation results with keys: valid (bool), errors (list), warnings (list), stats (dict)
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Check file exists
    if not Path(sa_reference_path).exists():
        result['valid'] = False
        result['errors'].append(f"File not found: {sa_reference_path}")
        return result

    # Load and check columns
    try:
        ref_df = pd.read_csv(sa_reference_path)
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Failed to load file: {e}")
        return result

    required_cols = ['stage_hpf', 'p5', 'p50', 'p95', 'n']
    missing_cols = [c for c in required_cols if c not in ref_df.columns]
    if missing_cols:
        result['valid'] = False
        result['errors'].append(f"Missing columns: {missing_cols}")
        return result

    # Check for NaN values
    for col in ['p5', 'p50', 'p95']:
        n_nan = ref_df[col].isna().sum()
        if n_nan > 0:
            result['warnings'].append(f"Column '{col}' has {n_nan} NaN values")

    # Check stage coverage
    stage_min = ref_df['stage_hpf'].min()
    stage_max = ref_df['stage_hpf'].max()
    result['stats']['stage_range'] = (stage_min, stage_max)

    if stage_min > 10:
        result['warnings'].append(f"Reference starts at {stage_min:.1f} hpf (late)")
    if stage_max < 100:
        result['warnings'].append(f"Reference ends at {stage_max:.1f} hpf (early)")

    # Check sample sizes
    low_n_bins = (ref_df['n'] < 5).sum()
    if low_n_bins > 0:
        result['warnings'].append(f"{low_n_bins} bins have n<5 samples")

    result['stats']['n_bins'] = len(ref_df)
    result['stats']['median_n'] = ref_df['n'].median()
    result['stats']['total_samples'] = ref_df['n'].sum()

    return result
