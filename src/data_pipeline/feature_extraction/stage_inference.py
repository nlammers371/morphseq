"""
Developmental stage (HPF) prediction from embryo features.

Predicts hours post-fertilization using Kimmel et al. (1995) temperature formula.
Extracted from build03A_process_images.py segment_wells_sam2_csv (lines 984-989).
"""

import numpy as np
import pandas as pd
from typing import Optional


def predict_stage_hpf(
    start_age_hpf: float,
    elapsed_time_s: float,
    temperature_c: float,
) -> float:
    """
    Predict developmental stage using Kimmel et al. (1995) formula.

    Formula: stage_hpf = start_age + time_hours * (0.055 * temp - 0.57)

    Args:
        start_age_hpf: Starting age at fertilization (hours post-fertilization)
        elapsed_time_s: Elapsed time since start (seconds)
        temperature_c: Incubation temperature (Celsius)

    Returns:
        Predicted stage in hours post-fertilization (HPF)
    """
    time_hours = elapsed_time_s / 3600.0

    # Kimmel et al. 1995 developmental rate formula
    developmental_rate = 0.055 * temperature_c - 0.57

    predicted_hpf = start_age_hpf + (time_hours * developmental_rate)

    return float(predicted_hpf)


def infer_stage_from_area(
    area_um2: float,
    reference_curve: Optional[dict] = None,
) -> tuple:
    """
    Infer developmental stage from surface area using reference curve.

    This is a STUB for MVP. Full implementation would use empirical
    surface area vs. stage curves from reference data.

    Args:
        area_um2: Embryo surface area in square micrometers
        reference_curve: Reference area vs. stage mapping (future)

    Returns:
        Tuple of (predicted_stage_hpf, confidence)
    """
    # STUB: Return NaN for now
    # Full implementation would interpolate from reference curves
    return np.nan, np.nan


def compute_stage_predictions_batch(
    tracking_df: pd.DataFrame,
    start_age_col: str = 'start_age_hpf',
    time_col: str = 'time_s',
    temp_col: str = 'temperature_c',
) -> pd.DataFrame:
    """
    Compute stage predictions for batch of snips.

    Args:
        tracking_df: DataFrame with temporal and temperature metadata
        start_age_col: Column name for starting age (HPF)
        time_col: Column name for elapsed time (seconds)
        temp_col: Column name for temperature (Celsius)

    Returns:
        DataFrame with snip_id and predicted_stage_hpf
    """
    results = []

    for idx, row in tracking_df.iterrows():
        snip_id = row['snip_id']

        # Extract metadata
        start_age = row.get(start_age_col, np.nan)
        elapsed_time = row.get(time_col, np.nan)
        temperature = row.get(temp_col, np.nan)

        # Handle alternative column names
        if pd.isna(temperature) and 'temperature' in row:
            temperature = row['temperature']

        # Predict stage
        if pd.notna(start_age) and pd.notna(elapsed_time) and pd.notna(temperature):
            predicted_hpf = predict_stage_hpf(
                start_age,
                elapsed_time,
                temperature
            )
        else:
            predicted_hpf = np.nan

        results.append({
            'snip_id': snip_id,
            'predicted_stage_hpf': predicted_hpf,
            'stage_confidence': 1.0 if pd.notna(predicted_hpf) else 0.0,
        })

    return pd.DataFrame(results)
